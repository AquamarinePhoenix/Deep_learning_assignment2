import os
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import _modules.config as cfg
from _modules.write import write_to_file
from _modules.dataset import CaptionDataset, collate_fn
import time
import re
from collections import Counter

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    tokens = text.split()
    return tokens

def token_f1(pred, gt):
    pred_tokens = normalize_text(pred)
    gt_tokens = normalize_text(gt)

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    # intersection (min counts)
    common = pred_counter & gt_counter
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / sum(pred_counter.values())
    recall = overlap / sum(gt_counter.values())
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def _generation_kwargs():
    gen_kwargs = {
        "max_new_tokens": 50,
        "num_beams": cfg.NUM_BEAMS,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    if getattr(cfg, "DO_SAMPLE", True):
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = cfg.TEMPERATURE

    return gen_kwargs

def evaluate_epoch_metrics(model, processor, data_dict, device):
    if not data_dict:
        return None

    dataset = CaptionDataset(data_dict, cfg.IMAGE_DIR, processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    count = 0

    model.eval()
    with th.no_grad():
        for batch in loader:
            image = batch["image"][0]
            caption = batch["caption"][0]

            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model.generate(**inputs, **_generation_kwargs())
            pred_lt = processor.decode(out[0], skip_special_tokens=True)

            precision, recall, f1 = token_f1(pred_lt, caption)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            count += 1

    if count == 0:
        return None

    return total_precision / count, total_recall / count, total_f1 / count

def train(model, processor, optimizer, data_dict, device, results_file, val_data=None, save_best=cfg.SAVE_BEST):
    loss_history = []
    epoch_times = []
    val_precision_history = []
    val_recall_history = []
    val_f1_history = []
    first_epoch_sample_index = 0

    model.to(device)

    dataset = CaptionDataset(data_dict, cfg.IMAGE_DIR, processor)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    best_loss = float("inf") if save_best else None
    best_run_root = None
    if save_best:
        best_run_root = os.path.join(cfg.BEST_MODEL_SAVE_DIR, f"run_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(best_run_root, exist_ok=True)

    for epoch in range(cfg.EPOCHS):
        epoch_start_time = time.time()
        model.train()

        total_loss = 0
        count = 0

        write_to_file(results_file, f"\nEpoch {epoch + 1}")

        for batch in loader:
            images = batch["image"]
            captions = batch["caption"]
            image_names = batch["image_name"]

            inputs = processor(
                images=images,
                text=captions,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            count += 1

            # log first epoch samples
            if epoch == 0:
                for i in range(len(image_names)):
                    first_epoch_sample_index += 1
                    write_to_file(
                        results_file,
                        f"{first_epoch_sample_index}. {image_names[i]} | {captions[i]}"
                    )

        avg_loss = total_loss / count if count > 0 else 0
        loss_history.append(avg_loss)

        val_metrics = evaluate_epoch_metrics(model, processor, val_data, device) if val_data else None
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        if val_metrics:
            val_precision, val_recall, val_f1 = val_metrics
        else:
            # ensure zeros are recorded so plotting shows the lines from epoch 1
            val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0

        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)
        val_f1_history.append(val_f1)

        write_to_file(
            results_file,
            f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Time: {epoch_time:.2f}s"
        )

        # save best-performing model by lowest average loss
        if save_best:
            try:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_dir = os.path.join(best_run_root, f"epoch_{epoch + 1:03d}") if best_run_root else cfg.BEST_MODEL_SAVE_DIR
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)
                    processor.save_pretrained(best_dir)
                    os.makedirs(cfg.BEST_MODEL_SAVE_DIR, exist_ok=True)
                    with open(os.path.join(cfg.BEST_MODEL_SAVE_DIR, "latest.txt"), "w", encoding="utf-8") as latest_file:
                        latest_file.write(best_dir)
                    write_to_file(results_file, f"New best model saved (Epoch {epoch + 1})\
                        with avg_loss {avg_loss:.4f} -> {best_dir}")
            except Exception as e:
                write_to_file(results_file, f"Failed to save best model: {e}")

    # always save final model
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    model.save_pretrained(cfg.MODEL_SAVE_DIR)
    processor.save_pretrained(cfg.MODEL_SAVE_DIR)

    return model, loss_history, epoch_times, (val_precision_history, val_recall_history, val_f1_history)

def evaluate(model, processor, data_dict, device, results_file, clip_model, clip_processor):
    model.to(device)
    model.eval()

    evaluated_count = 0
    skipped_count = 0
    
    total_f1 = 0
    count = 0

    for item in data_dict:
        img_name = item["image"]
        test_path = os.path.join(cfg.IMAGE_DIR, "test", os.path.basename(img_name))

        if not os.path.exists(test_path):
            fallback_path = os.path.join(cfg.IMAGE_DIR, "test", img_name)
            if os.path.exists(fallback_path):
                test_path = fallback_path
            else:
                skipped_count += 1
                write_to_file(results_file, f"\nTEST {img_name}")
                write_to_file(results_file, "Skipped: test image not found")
                continue

        image = Image.open(test_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with th.no_grad():
            out = model.generate(
                **inputs,
                **_generation_kwargs(),
            )

        pred_lt = processor.decode(out[0], skip_special_tokens=True)
        
        # prepare inputs for CLIP
        clip_inputs = clip_processor(
            text=[pred_lt, item["caption_lt"]],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with th.no_grad():
            outputs = clip_model(**clip_inputs)

        # embeddings
        image_embeds = outputs.image_embeds      # (1, D)
        text_embeds = outputs.text_embeds        # (2, D)

        # normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # compute cosine similarities
        sim_pred = th.matmul(image_embeds, text_embeds[0].unsqueeze(0).T).item()
        sim_gt   = th.matmul(image_embeds, text_embeds[1].unsqueeze(0).T).item()
        
        precision, recall, f1 = token_f1(pred_lt, item["caption_lt"])
        
        real_lt = item.get("caption_lt", "")

        write_to_file(results_file, f"\nTEST {img_name}")
        write_to_file(results_file, f"Pred_LT: {pred_lt}")
        write_to_file(results_file, f"Real_LT: {real_lt}")
        write_to_file(results_file, f"CLIP_sim_pred: {sim_pred:.4f}")
        write_to_file(results_file, f"CLIP_sim_gt:   {sim_gt:.4f}")
        write_to_file(results_file, f"Precision: {precision:.4f}")
        write_to_file(results_file, f"Recall:    {recall:.4f}")
        write_to_file(results_file, f"F1:        {f1:.4f}")
        total_f1 += f1
        count += 1
        evaluated_count += 1

    write_to_file(results_file, f"TEST summary -> evaluated: {evaluated_count}, skipped: {skipped_count}")
    avg_f1 = total_f1 / count if count > 0 else 0
    write_to_file(results_file, f"\nAverage F1: {avg_f1:.4f}")
    return model
