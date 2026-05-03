import os
import random
import torch as th
from torch.utils.data import DataLoader
from PIL import Image
import _modules.config as cfg
from _modules.write import write_to_file
from _modules.dataset import CaptionDataset, collate_fn
from _modules.plots import plot_loss_history
import time
import re

def f1_score(pred, real):
    pred_tokens = set(re.findall(r"\w+", pred.lower()))
    real_tokens = set(re.findall(r"\w+", real.lower()))

    if not pred_tokens or not real_tokens:
        return 0

    tp = len(pred_tokens & real_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(real_tokens)

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)

def bleu_score(pred, real):
    # Simple BLEU-1 with brevity penalty
    pred_tokens = re.findall(r"\\w+", pred.lower())
    real_tokens = re.findall(r"\\w+", real.lower())
    if not pred_tokens or not real_tokens:
        return 0
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    match = 0
    for t in set(real_tokens):
        match += min(pred_counts.get(t, 0), real_tokens.count(t))
    precision = match / len(pred_tokens)
    # brevity penalty
    ref_len = len(real_tokens)
    cand_len = len(pred_tokens)
    bp = 1.0
    if cand_len == 0:
        bp = 0.0
    elif cand_len < ref_len:
        bp = pow(2.718281828, 1 - ref_len / cand_len)
    return bp * precision

def compute_val_metrics(model, processor, data_dict, device, image_split="train"):
    model.eval()
    total_f1 = 0
    total_bleu = 0
    count = 0
    with th.no_grad():
        for item in data_dict:
            img_name = item["image"]
            test_path = cfg.IMAGE_DIR + image_split + "/" + img_name
            if not os.path.exists(test_path):
                continue
            image = Image.open(test_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
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
            out = model.generate(**inputs, **gen_kwargs)
            pred_lt = processor.decode(out[0], skip_special_tokens=True)
            total_f1 += f1_score(pred_lt, item["caption_lt"])
            total_bleu += bleu_score(pred_lt, item["caption_lt"])
            count += 1
    avg_f1 = total_f1 / count if count > 0 else 0
    avg_bleu = total_bleu / count if count > 0 else 0
    return avg_f1, avg_bleu

def train(model, processor, optimizer, data_dict, device, results_file, val_data=None, save_best=cfg.SAVE_BEST):
    loss_history = []
    epoch_times = []
    first_epoch_sample_index = 0

    model.to(device)

    dataset = CaptionDataset(data_dict, cfg.IMAGE_DIR, processor)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    best_loss = float("inf") if save_best else None

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
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        write_to_file(results_file, f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        # save best-performing model by lowest average loss
        if save_best:
            try:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_dir = cfg.BEST_MODEL_SAVE_DIR
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)
                    processor.save_pretrained(best_dir)
                    write_to_file(results_file, f"New best model saved (Epoch {epoch + 1})\
                        with avg_loss {avg_loss:.4f} -> {best_dir}")
            except Exception as e:
                write_to_file(results_file, f"Failed to save best model: {e}")

    # always save final model
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    model.save_pretrained(cfg.MODEL_SAVE_DIR)
    processor.save_pretrained(cfg.MODEL_SAVE_DIR)

    return model, loss_history, epoch_times

def validate(model, processor, data_dict, device, results_file, train_loss=None, epoch_times=None, plot_path=None):
    model.to(device)
    model.eval()

    total_f1 = 0
    total_bleu = 0
    count = 0

    for item in data_dict:
        img_name = item["image"]
        test_path = cfg.IMAGE_DIR + "test/" + img_name

        if not os.path.exists(test_path):
            continue

        image = Image.open(test_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with th.no_grad():
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

            out = model.generate(
                **inputs,
                **gen_kwargs,
            )

        pred_lt = processor.decode(out[0], skip_special_tokens=True)
        real_lt = item.get("caption_lt", "")

        write_to_file(results_file, f"\nVALIDATION {img_name}")
        write_to_file(results_file, f"Pred_LT: {pred_lt}")
        write_to_file(results_file, f"Real_LT: {real_lt}")

        if real_lt:
            total_f1 += f1_score(pred_lt, real_lt)
            total_bleu += bleu_score(pred_lt, real_lt)
            count += 1

    avg_f1 = total_f1 / count if count > 0 else 0
    avg_bleu = total_bleu / count if count > 0 else 0
    write_to_file(results_file, f"Validation summary -> F1: {avg_f1:.4f}, BLEU: {avg_bleu:.4f}")

    if train_loss:
        plot_loss_history(train_loss, epoch_times=epoch_times, save_path=plot_path)

    return avg_f1, avg_bleu

def evaluate(model, processor, data_dict, device, results_file):
    model.to(device)
    model.eval()

    for item in data_dict:
        img_name = item["image"]
        test_path = cfg.IMAGE_DIR + "test/" + img_name

        if not os.path.exists(test_path):
            continue

        image = Image.open(test_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with th.no_grad():
            gen_kwargs = {
                "max_new_tokens": 50,
                "num_beams": cfg.NUM_BEAMS,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
            }

            # `temperature` only applies to sampling-based decoding.
            if getattr(cfg, "DO_SAMPLE", True):
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = cfg.TEMPERATURE

            out = model.generate(
                **inputs,
                **gen_kwargs,
            )

        pred_lt = processor.decode(out[0], skip_special_tokens=True)

        write_to_file(results_file, f"\nTEST {img_name}")
        write_to_file(results_file, f"Pred_LT: {pred_lt}")
    return model

def split_train_validation(train_data, results_file):
    val_split = getattr(cfg, "VAL_SPLIT", None)
    if val_split is not None and val_split > 0:
        train_items = list(train_data)
        random.Random(42).shuffle(train_items)
        val_size = max(1, int(len(train_items) * val_split))
        val_data = train_items[:val_size]
        train_data = train_items[val_size:]
        write_to_file(results_file, f"Using {len(train_data)} training samples and {len(val_data)} validation samples (VAL_SPLIT={val_split})")
    else:
        val_data = None
        write_to_file(results_file, f"Using {len(train_data)} training samples (no validation split)")
    return train_data, val_data