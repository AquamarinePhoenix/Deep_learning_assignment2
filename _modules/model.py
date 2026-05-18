import os
import torch as th
from torch.utils.data import DataLoader
from PIL import Image
import _modules.config as cfg
from _modules.write import write_to_file
from _modules.dataset import CaptionDataset, collate_fn
import time
import re
from collections import Counter
from functools import partial
import math

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    tokens = text.split()
    return tokens

def _extract_ngrams(tokens, n):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def _metric_suffix(metric_n):
    return f"{metric_n}"

def bleu_score(pred, gt, max_n=4):
    pred_tokens = normalize_text(pred)
    ref_tokens = normalize_text(gt)

    if not pred_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(_extract_ngrams(pred_tokens, n))
        ref_ngrams = Counter(_extract_ngrams(ref_tokens, n))

        if not pred_ngrams:
            precisions.append(1e-12)
            continue

        overlap = sum((pred_ngrams & ref_ngrams).values())
        total_pred = sum(pred_ngrams.values())
        precisions.append(max(overlap / total_pred, 1e-12))

    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    if pred_len == 0:
        return 0.0

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / pred_len))
    geo_mean = math.exp(sum((1.0 / max_n) * math.log(p) for p in precisions))
    return brevity_penalty * geo_mean

def bleu4_score(pred, gt):
    return bleu_score(pred, gt, max_n=4)

def _build_idf(reference_texts, max_n=4):
    num_docs = max(1, len(reference_texts))
    idf = {}

    for n in range(1, max_n + 1):
        df = Counter()
        for text in reference_texts:
            tokens = normalize_text(text)
            ngrams = set(_extract_ngrams(tokens, n))
            for ng in ngrams:
                df[ng] += 1

        for ng, freq in df.items():
            idf[ng] = math.log((num_docs + 1) / (freq + 1)) + 1.0

    return idf

def cider_score(pred, gt, idf_map, max_n=4):
    pred_tokens = normalize_text(pred)
    ref_tokens = normalize_text(gt)

    per_n_scores = []
    for n in range(1, max_n + 1):
        pred_counts = Counter(_extract_ngrams(pred_tokens, n))
        ref_counts = Counter(_extract_ngrams(ref_tokens, n))

        if not pred_counts or not ref_counts:
            per_n_scores.append(0.0)
            continue

        pred_total = sum(pred_counts.values())
        ref_total = sum(ref_counts.values())

        pred_vec = {
            ng: (count / pred_total) * idf_map.get(ng, 0.0)
            for ng, count in pred_counts.items()
        }
        ref_vec = {
            ng: (count / ref_total) * idf_map.get(ng, 0.0)
            for ng, count in ref_counts.items()
        }

        common = set(pred_vec.keys()) & set(ref_vec.keys())
        dot = sum(pred_vec[ng] * ref_vec[ng] for ng in common)
        pred_norm = math.sqrt(sum(v * v for v in pred_vec.values()))
        ref_norm = math.sqrt(sum(v * v for v in ref_vec.values()))

        if pred_norm == 0.0 or ref_norm == 0.0:
            per_n_scores.append(0.0)
            continue

        per_n_scores.append(dot / (pred_norm * ref_norm))

    # Keep the common CIDEr scaling while using 1..4 gram agreement.
    return 10.0 * (sum(per_n_scores) / max_n)

def cider4_score(pred, gt, idf_map):
    return cider_score(pred, gt, idf_map, max_n=4)

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

def evaluate_epoch_metrics(model, processor, data_dict, device, metric_n=4):
    if not data_dict:
        return None

    dataset = CaptionDataset(data_dict, cfg.IMAGE_DIR, processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    reference_texts = [item["caption_lt"] for item in data_dict if item.get("caption_lt")]
    idf_map = _build_idf(reference_texts, max_n=metric_n)

    total_bleu4 = 0.0
    total_cider4 = 0.0
    count = 0

    model.eval()
    with th.no_grad():
        for batch in loader:
            image = batch["image"][0]
            caption = batch["caption"][0]

            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model.generate(**inputs, **_generation_kwargs())
            pred_lt = processor.decode(out[0], skip_special_tokens=True)

            bleu_score_value = bleu_score(pred_lt, caption, max_n=metric_n)
            cider_score_value = cider_score(pred_lt, caption, idf_map, max_n=metric_n)
            total_bleu4 += bleu_score_value
            total_cider4 += cider_score_value
            count += 1

    if count == 0:
        return None

    return total_bleu4 / count, total_cider4 / count

def train(model, processor, optimizer, data_dict, device, results_file, val_data=None, save_best=cfg.SAVE_BEST, metric_n=4):
    loss_history = []
    epoch_times = []
    val_bleu4_history = []
    val_cider4_history = []
    first_epoch_sample_index = 0

    model.to(device)

    dataset = CaptionDataset(data_dict, cfg.IMAGE_DIR, processor)
    # Use mosaic augmentation: 50% probability of creating 2x2 image mosaics during training
    collate_with_mosaic = partial(collate_fn, apply_mosaic=True, mosaic_probability=0.5)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_with_mosaic)

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

        val_metrics = evaluate_epoch_metrics(model, processor, val_data, device, metric_n=metric_n) if val_data else None
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        if val_metrics:
            val_bleu4, val_cider4 = val_metrics
        else:
            # ensure zeros are recorded so plotting shows the lines from epoch 1
            val_bleu4, val_cider4 = 0.0, 0.0

        val_bleu4_history.append(val_bleu4)
        val_cider4_history.append(val_cider4)

        write_to_file(
            results_file,
            f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}, Val BLEU-{_metric_suffix(metric_n)}: {val_bleu4:.4f}, Val CIDEr-{_metric_suffix(metric_n)}: {val_cider4:.4f}, Time: {epoch_time:.2f}s"
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

    return model, loss_history, epoch_times, (val_bleu4_history, val_cider4_history)

def _resolve_image_path(item):
    image_path = item.get("image_path")
    if image_path:
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(cfg.IMAGE_DIR, image_path)

    img_name = item["image"]
    test_path = os.path.join(cfg.IMAGE_DIR, "test", os.path.basename(img_name))

    if os.path.exists(test_path):
        return test_path

    fallback_path = os.path.join(cfg.IMAGE_DIR, "test", img_name)
    if os.path.exists(fallback_path):
        return fallback_path

    train_path = os.path.join(cfg.IMAGE_DIR, "train", img_name)
    if os.path.exists(train_path):
        return train_path

    if os.path.exists(img_name):
        return img_name

    return test_path

def evaluate(model, processor, data_dict, device, results_file, metric_n=4):
    model.to(device)
    model.eval()

    evaluated_count = 0
    skipped_count = 0
    
    reference_texts = [item.get("caption_lt", item.get("caption", "")) for item in data_dict if item.get("caption_lt") or item.get("caption")]
    idf_map = _build_idf(reference_texts, max_n=metric_n)

    total_bleu4 = 0.0
    total_cider4 = 0.0
    count = 0

    for item in data_dict:
        img_name = item.get("image", os.path.basename(item.get("image_path", "unknown")))
        test_path = _resolve_image_path(item)

        if not os.path.exists(test_path):
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

        ground_truth = item.get("caption_lt", item.get("caption", ""))
        bleu4 = bleu_score(pred_lt, ground_truth, max_n=metric_n)
        cider4 = cider_score(pred_lt, ground_truth, idf_map, max_n=metric_n)
        
        real_lt = ground_truth

        write_to_file(results_file, f"\nTEST {img_name}")
        write_to_file(results_file, f"Pred_LT: {pred_lt}")
        write_to_file(results_file, f"Real_LT: {real_lt}")
        write_to_file(results_file, f"BLEU-{_metric_suffix(metric_n)}:  {bleu4:.4f}")
        write_to_file(results_file, f"CIDEr-{_metric_suffix(metric_n)}: {cider4:.4f}")
        total_bleu4 += bleu4
        total_cider4 += cider4
        count += 1
        evaluated_count += 1

    write_to_file(results_file, f"TEST summary -> evaluated: {evaluated_count}, skipped: {skipped_count}")
    avg_bleu4 = total_bleu4 / count if count > 0 else 0.0
    avg_cider4 = total_cider4 / count if count > 0 else 0.0
    write_to_file(results_file, f"\nAverage BLEU-{_metric_suffix(metric_n)}: {avg_bleu4:.4f}")
    write_to_file(results_file, f"Average CIDEr-{_metric_suffix(metric_n)}: {avg_cider4:.4f}")
    return model
