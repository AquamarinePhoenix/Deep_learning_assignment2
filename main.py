#%% Run the __main__ script

import os
import json
import random
import torch as th
from dotenv import load_dotenv
import _modules.config as cfg
from _modules.VLM import load_VLM
from _modules.write import clear_file, write_to_file
from _modules.model import train, evaluate
from _modules.plots import plot_training_curves
from _modules.experiment import build_openimages_dataset

load_dotenv()
token = os.getenv("HF_TOKEN")
temperature_tag = str(cfg.TEMPERATURE).replace(".", "p")
results_file = os.path.join(cfg.OUTPUT_DIR, f"results_temp_{temperature_tag}.txt")
clear_file(results_file)


def _load_mutual_test_records():
    mutual_dir = os.path.join(os.path.dirname(cfg.CAPTIONS_TEST), "mutual")
    if not os.path.isdir(mutual_dir):
        return []

    caption_map = {
        "dog.jpg": ("a dog", "šuo"),
        "horse.jpg": ("a horse", "arklys"),
        "horse_dog.jpg": ("a horse and a dog", "arklys ir šuo"),
        "park.jpg": ("a park", "parkas"),
        "random.jpg": ("a colorful grid", "spalvų tinklelis"),
    }

    mutual_records = []
    for filename in sorted(os.listdir(mutual_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        caption_en, caption_lt = caption_map.get(
            filename,
            (os.path.splitext(filename)[0].replace("_", " "), os.path.splitext(filename)[0].replace("_", " ")),
        )
        absolute_image_path = os.path.abspath(os.path.join(mutual_dir, filename))
        mutual_records.append(
            {
                "image": filename,
                "image_path": absolute_image_path,
                "caption_en": caption_en,
                "caption_lt": caption_lt,
                "label": "mutual",
                "split": "test",
            }
        )

    return mutual_records


def _select_coco_training_records(records, target_total, label_ratios=None, seed=42):
    if not records or target_total <= 0:
        return []

    label_ratios = label_ratios or {"horse": 0.45, "dog": 0.45, "background": 0.10}
    label_order = ["horse", "dog", "background"]
    grouped = {label: [] for label in label_order}
    for record in records:
        label = str(record.get("label", "")).lower()
        if label in grouped:
            grouped[label].append(record)

    rng = random.Random(seed)
    for label_records in grouped.values():
        rng.shuffle(label_records)

    raw_targets = {label: float(target_total) * float(label_ratios.get(label, 0.0)) for label in label_order}
    target_counts = {label: int(raw_targets[label]) for label in label_order}
    remainder = target_total - sum(target_counts.values())
    if remainder > 0:
        for label in sorted(label_order, key=lambda name: (raw_targets[name] - target_counts[name]), reverse=True):
            if remainder == 0:
                break
            target_counts[label] += 1
            remainder -= 1

    selected = []
    for label in label_order:
        pool = grouped[label]
        if not pool:
            continue

        needed = target_counts[label]
        if needed <= len(pool):
            selected.extend(pool[:needed])
            continue

        selected.extend(pool)
        index = 0
        while len([item for item in selected if str(item.get("label", "")).lower() == label]) < needed:
            selected.append(pool[index % len(pool)])
            index += 1

    if len(selected) < target_total:
        leftovers = [record for record in records if record not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target_total - len(selected)])

    return selected[:target_total]

write_to_file(results_file, f"TEMPERATURE: {cfg.TEMPERATURE}")
write_to_file(results_file, f"COCO_TRAIN_RATIO: {getattr(cfg, 'COCO_TRAIN_RATIO', 0.5)}")

with open(cfg.CAPTIONS_TRAIN, "r", encoding="utf-8") as f:
    train_data = json.load(f)

num_all_train_samples = len(train_data)
train_subset_ratio = float(getattr(cfg, "TRAIN_SUBSET_RATIO", 1.0))
subset_size = int(train_subset_ratio * num_all_train_samples)

if train_subset_ratio < 1.0:
    train_data = train_data[:subset_size]

if num_all_train_samples > 0 and len(train_data) == 0:
    raise ValueError(
        f"TRAIN_SUBSET_RATIO={train_subset_ratio} selected 0 samples from {num_all_train_samples}. "
        "Increase TRAIN_SUBSET_RATIO for a non-empty training set."
    )

openimages_train_data = []
openimages_test_data = []
if getattr(cfg, "USE_OPENIMAGES_EXPERIMENT", True):
    openimages_train_data, openimages_test_data = build_openimages_dataset()
    coco_train_target = max(1, int(round(num_all_train_samples * float(getattr(cfg, "COCO_TRAIN_RATIO", 0.5)))))
    openimages_train_data = _select_coco_training_records(
        openimages_train_data,
        coco_train_target,
        label_ratios=getattr(cfg, "COCO_TRAIN_LABEL_RATIOS", None),
        seed=42,
    )

if openimages_train_data:
    train_data = train_data + openimages_train_data

val_split = float(getattr(cfg, "VAL_SPLIT", 0.0))
val_data = []
if 0.0 < val_split < 1.0 and len(train_data) > 1:
    shuffled_train_data = train_data[:]
    random.Random(42).shuffle(shuffled_train_data)
    val_size = int(round(len(shuffled_train_data) * val_split))
    val_size = max(1, min(len(shuffled_train_data) - 1, val_size))
    val_data = shuffled_train_data[-val_size:]
    train_data = shuffled_train_data[:-val_size]
elif val_split > 0.0:
    write_to_file(results_file, "Validation split requested but not enough samples to create a hold-out set")

write_to_file(results_file, f"TRAIN set JSON: {cfg.CAPTIONS_TRAIN}")
write_to_file(results_file, f"Total TRAIN samples available: {num_all_train_samples}")
write_to_file(results_file, f"TRAIN_SUBSET_RATIO: {train_subset_ratio}")
write_to_file(results_file, f"COCO train target: {max(1, int(round(num_all_train_samples * float(getattr(cfg, 'COCO_TRAIN_RATIO', 0.5)))))}")
write_to_file(results_file, f"COCO train samples added: {len(openimages_train_data)}")
write_to_file(results_file, f"Combined train samples used: {len(train_data)}")
write_to_file(results_file, f"Loaded {len(train_data)} TRAIN samples")
write_to_file(results_file, f"VAL_SPLIT: {val_split}")
write_to_file(results_file, f"Loaded {len(val_data)} VALIDATION samples")

load_dir = None
if getattr(cfg, 'USE_BEST_MODEL', False):
    latest_path = os.path.join(cfg.BEST_MODEL_SAVE_DIR, "latest.txt")
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as latest_file:
            candidate_dir = latest_file.read().strip()
        if candidate_dir:
            load_dir = candidate_dir
    elif os.path.exists(cfg.BEST_MODEL_SAVE_DIR):
        load_dir = cfg.BEST_MODEL_SAVE_DIR

if load_dir:
    write_to_file(results_file, f"Loading best model from {cfg.BEST_MODEL_SAVE_DIR}")

model, processor, device = load_VLM(cfg.VLM_NAME, cfg.DEVICE, load_dir=load_dir)

for param in model.vision_model.parameters():
        param.requires_grad = False

optimizer = th.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.LEARNING_RATE
)

if load_dir:
    write_to_file(results_file, "USE_BEST_MODEL is enabled; skipping training and evaluating the loaded checkpoint only")
else:
    model, loss_history, epoch_times, val_metrics = train(model, processor, optimizer, train_data, device, results_file, val_data=val_data)
    write_to_file(results_file, f"Total training time: {sum(epoch_times):.2f}s")

    plot_training_curves(loss_history, epoch_times, val_metrics=val_metrics, temperature=cfg.TEMPERATURE)

with open(cfg.CAPTIONS_TEST, "r", encoding="utf-8") as f:
    test_data = json.load(f)

mutual_test_data = _load_mutual_test_records()
if mutual_test_data:
    test_data.extend(mutual_test_data)

write_to_file(results_file, f"TEST set JSON: {cfg.CAPTIONS_TEST}")
write_to_file(results_file, f"Loaded {len(test_data)} TEST samples")
if mutual_test_data:
    write_to_file(results_file, f"Added {len(mutual_test_data)} mutual test samples from data/mutual")

model = evaluate(model, processor, test_data, device, results_file)

if openimages_test_data:
    write_to_file(results_file, "---test_on_coco----")
    write_to_file(results_file, f"Loaded {len(openimages_test_data)} COCO test samples")
    model = evaluate(model, processor, openimages_test_data, device, results_file, metric_n=getattr(cfg, "OPENIMAGES_METRIC_N", 4))
# %%
