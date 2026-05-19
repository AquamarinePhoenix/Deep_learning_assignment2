import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
from PIL import Image

import _modules.config as cfg
from _modules.VLM import load_VLM
from _modules.model import evaluate, train
from _modules.plots import plot_training_curves
from _modules.write import clear_file, write_to_file


@dataclass
class OpenImagesExperimentConfig:
    train_split: float = 0.8
    val_split: float = 0.1
    split_seed: int = 42
    max_samples_per_class: int = 100
    target_classes: tuple[str, ...] = ("horse", "dog", "background")
    target_captions_en: dict[str, str] = field(
        default_factory=lambda: {
            "horse": "a horse",
            "dog": "a dog",
            "background": "a background",
        }
    )
    target_captions_lt: dict[str, str] = field(
        default_factory=lambda: {
            "horse": "arklys",
            "dog": "šuo",
            "background": "fonas",
        }
    )
    dataset_dir: str = cfg.OPENIMAGES_DATA_DIR
    results_file: str = cfg.OPENIMAGES_RESULTS_FILE
    metric_n: int = cfg.OPENIMAGES_METRIC_N


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _load_fiftyone_zoo_dataset(max_samples):
    try:
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError(
            "The COCO experiment requires the 'fiftyone' package. Install it before running this path."
        ) from exc

    dataset_name = getattr(cfg, "OPENIMAGES_SOURCE_DATASET", "coco-2017")

    return foz.load_zoo_dataset(
        dataset_name,
        split="train",
        label_types=["detections"],
        shuffle=True,
        seed=cfg.OPENIMAGES_SPLIT_SEED,
        max_samples=max(1, max_samples),
    )


def _extract_sample_labels(sample):
    sample_dict = sample.to_dict() if hasattr(sample, "to_dict") else {}
    labels = []
    for field_name in ("ground_truth", "detections", "objects", "labels"):
        field_value = sample_dict.get(field_name)
        if isinstance(field_value, dict):
            detections = field_value.get("detections", [])
            for detection in detections:
                if isinstance(detection, dict):
                    label = detection.get("label")
                else:
                    label = getattr(detection, "label", None)
                if label:
                    labels.append(str(label).lower())
    return labels


def _sample_to_record(sample, caption_en, caption_lt, label, split_name, dataset_root):
    image_dir = os.path.join(dataset_root, "images", split_name, label)
    _ensure_dir(image_dir)

    source_path = getattr(sample, "filepath", None)
    if not source_path or not os.path.exists(source_path):
        raise FileNotFoundError(f"Source image missing for {label}: {source_path}")

    safe_name = f"{split_name}_{label}_{getattr(sample, 'id', random.randint(0, 10**9))}.jpg"
    target_path = os.path.join(image_dir, safe_name)
    if not os.path.exists(target_path):
        shutil.copy2(source_path, target_path)

    relative_image_path = os.path.relpath(target_path, os.path.join(dataset_root, "images"))
    return {
        "image": relative_image_path,
        "image_path": target_path,
        "caption_en": caption_en,
        "caption_lt": caption_lt,
        "label": label,
        "split": split_name,
    }


def _build_class_records(dataset, label_name, max_samples, target_labels=None):
    records = []
    target_labels = {label.lower() for label in (target_labels or [label_name])}

    for sample in dataset:
        sample_labels = set(_extract_sample_labels(sample))
        if label_name == "background":
            if sample_labels.intersection(target_labels):
                continue
        else:
            if label_name not in sample_labels:
                continue

        records.append(sample)
        if len(records) >= max_samples:
            break

    return records


def _balanced_split(records_by_label, train_split, seed):
    rng = random.Random(seed)
    train_records = []
    test_records = []

    for label, records in records_by_label.items():
        shuffled = records[:]
        rng.shuffle(shuffled)
        split_index = int(round(len(shuffled) * train_split))
        if len(shuffled) > 1:
            split_index = max(1, min(len(shuffled) - 1, split_index))
        else:
            split_index = len(shuffled)

        train_records.extend([(label, sample) for sample in shuffled[:split_index]])
        test_records.extend([(label, sample) for sample in shuffled[split_index:]])

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def build_openimages_dataset(experiment_cfg=None, force_rebuild=False):
    experiment_cfg = experiment_cfg or OpenImagesExperimentConfig()
    dataset_root = experiment_cfg.dataset_dir
    train_json = cfg.OPENIMAGES_TRAIN_JSON
    test_json = cfg.OPENIMAGES_TEST_JSON

    if not force_rebuild and os.path.exists(train_json) and os.path.exists(test_json):
        with open(train_json, "r", encoding="utf-8") as train_file:
            train_records = json.load(train_file)
        with open(test_json, "r", encoding="utf-8") as test_file:
            test_records = json.load(test_file)
        return train_records, test_records

    _ensure_dir(dataset_root)
    _ensure_dir(os.path.join(dataset_root, "images"))

    sample_pool_size = int(getattr(cfg, "OPENIMAGES_SOURCE_SAMPLE_POOL", max(5000, experiment_cfg.max_samples_per_class * 50)))
    zoo_dataset = _load_fiftyone_zoo_dataset(sample_pool_size)

    records_by_label = defaultdict(list)

    horse_records = _build_class_records(
        zoo_dataset,
        "horse",
        experiment_cfg.max_samples_per_class,
        target_labels={"horse"},
    )
    dog_records = _build_class_records(
        zoo_dataset,
        "dog",
        experiment_cfg.max_samples_per_class,
        target_labels={"dog"},
    )

    background_records = _build_class_records(
        zoo_dataset,
        "background",
        experiment_cfg.max_samples_per_class,
        target_labels={"horse", "dog"},
    )

    records_by_label["horse"] = horse_records
    records_by_label["dog"] = dog_records
    records_by_label["background"] = background_records

    for label_name, records in records_by_label.items():
        if not records:
            raise RuntimeError(
                f"COCO experiment could not collect any '{label_name}' samples. Check the FiftyOne label fields or target class names."
            )
        if len(records) < experiment_cfg.max_samples_per_class:
            raise RuntimeError(
                f"COCO experiment collected too few '{label_name}' samples: {len(records)} / {experiment_cfg.max_samples_per_class}. "
                f"Increase OPENIMAGES_SOURCE_SAMPLE_POOL (currently {sample_pool_size}) or reduce OPENIMAGES_MAX_SAMPLES_PER_CLASS."
            )

    train_records_raw, test_records_raw = _balanced_split(
        records_by_label,
        train_split=experiment_cfg.train_split,
        seed=experiment_cfg.split_seed,
    )

    if not train_records_raw or not test_records_raw:
        raise RuntimeError("COCO experiment split produced an empty train or test set.")

    train_records = []
    test_records = []
    for label, sample in train_records_raw:
        train_records.append(
            _sample_to_record(
                sample,
                experiment_cfg.target_captions_en[label],
                experiment_cfg.target_captions_lt[label],
                label,
                "train",
                dataset_root,
            )
        )
    for label, sample in test_records_raw:
        test_records.append(
            _sample_to_record(
                sample,
                experiment_cfg.target_captions_en[label],
                experiment_cfg.target_captions_lt[label],
                label,
                "test",
                dataset_root,
            )
        )

    _ensure_dir(os.path.dirname(train_json))
    with open(train_json, "w", encoding="utf-8") as train_file:
        json.dump(train_records, train_file, ensure_ascii=False, indent=2)
    with open(test_json, "w", encoding="utf-8") as test_file:
        json.dump(test_records, test_file, ensure_ascii=False, indent=2)

    return train_records, test_records


def _split_validation(train_records, val_split, seed):
    if not train_records or val_split <= 0.0 or len(train_records) < 2:
        return train_records, []

    shuffled = train_records[:]
    random.Random(seed).shuffle(shuffled)
    val_size = int(round(len(shuffled) * val_split))
    val_size = max(1, min(len(shuffled) - 1, val_size))
    return shuffled[:-val_size], shuffled[-val_size:]


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


def evaluate_main_model_on_openimages(model, processor, device, results_file):
    """Run inference using the main trained model on OpenImages test set."""
    experiment_cfg = OpenImagesExperimentConfig()
    train_records, test_records = build_openimages_dataset(experiment_cfg)
    _, val_records = _split_validation(
        train_records,
        val_split=experiment_cfg.val_split,
        seed=experiment_cfg.split_seed,
    )

    # Ensure test images are available
    test_out_dir = os.path.join(cfg.IMAGE_DIR, "test")
    os.makedirs(test_out_dir, exist_ok=True)
    for rec in test_records:
        src = rec.get("image_path")
        if not src or not os.path.exists(src):
            continue
        dst = os.path.join(test_out_dir, os.path.basename(src))
        try:
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            rec["image"] = os.path.basename(dst)
            rec["image_path"] = dst
        except Exception:
            continue

    write_to_file(results_file, "")
    write_to_file(results_file, "=" * 80)
    write_to_file(results_file, "MAIN MODEL INFERENCE ON OPENIMAGES TEST SET")
    write_to_file(results_file, "=" * 80)
    from _modules.model import evaluate as model_evaluate
    model_evaluate(model, processor, test_records, device, results_file, metric_n=experiment_cfg.metric_n)


def _save_test_preview(model, processor, test_records, device, results_file, metric_n, figure_path, sample_count=15):
    if not test_records:
        return

    def _pick_preview_records(records, target_count):
        return records[:target_count]

    preview_count = min(sample_count, len(test_records)) if test_records else 0
    preview_records = _pick_preview_records(test_records, preview_count)
    reference_texts = [item.get("caption_lt", item.get("caption", "")) for item in test_records if item.get("caption_lt") or item.get("caption")]
    idf_map = None
    if reference_texts:
        from _modules.model import _build_idf, bleu_score, cider_score

        idf_map = _build_idf(reference_texts, max_n=metric_n)
    else:
        from _modules.model import bleu_score, cider_score

    columns = 3
    rows = max(1, math.ceil(preview_count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    write_to_file(results_file, "")
    write_to_file(results_file, f"Saving {preview_count} test previews to {figure_path}")

    for index, (axis, item) in enumerate(zip(axes, preview_records), start=1):
        image_path = item.get("image_path")
        if image_path and os.path.exists(image_path):
            pass  # Use the provided image_path
        else:
            # Try multiple fallback locations
            basename = os.path.basename(item.get("image_path", item.get("image", "")))
            candidates = [
                os.path.join(cfg.IMAGE_DIR, "test", basename),
                os.path.join(cfg.OPENIMAGES_DATA_DIR, "images", "test", basename),
                os.path.join(cfg.OPENIMAGES_DATA_DIR, "images", item.get("image", "")),
                item.get("image", "")
            ]
            image_path = None
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    image_path = candidate
                    break
            if not image_path:
                write_to_file(results_file, f"  Warning: Could not find image for TEST PREVIEW {index}: {item.get('image')}")
                continue
        
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, **_generation_kwargs())

        predicted = processor.decode(outputs[0], skip_special_tokens=True)
        ground_truth = item.get("caption_lt", item.get("caption", ""))
        bleu_value = bleu_score(predicted, ground_truth, max_n=metric_n)
        cider_value = cider_score(predicted, ground_truth, idf_map, max_n=metric_n) if idf_map is not None else 0.0

        write_to_file(results_file, f"TEST PREVIEW {index}")
        write_to_file(results_file, f"Image: {item.get('image', os.path.basename(image_path))}")
        write_to_file(results_file, f"Predicted: {predicted}")
        write_to_file(results_file, f"Ground truth: {ground_truth}")
        write_to_file(results_file, f"BLEU-{metric_n}: {bleu_value:.4f}")
        write_to_file(results_file, f"CIDEr-{metric_n}: {cider_value:.4f}")

        axis.imshow(image)
        axis.set_title(
            f"GT: {ground_truth}\nPred: {predicted}",
            fontsize=9,
            wrap=True,
        )
        axis.axis("off")

    for axis in axes[len(preview_records):]:
        axis.axis("off")

    plt.tight_layout()
    _ensure_dir(os.path.dirname(figure_path))
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_openimages_experiment():
    experiment_cfg = OpenImagesExperimentConfig()
    train_records, test_records = build_openimages_dataset(experiment_cfg)
    train_records, val_records = _split_validation(
        train_records,
        val_split=experiment_cfg.val_split,
        seed=experiment_cfg.split_seed,
    )

    results_file = experiment_cfg.results_file
    clear_file(results_file)

    label_counts = Counter(record["label"] for record in train_records + val_records + test_records)
    write_to_file(results_file, f"OpenImages dataset dir: {experiment_cfg.dataset_dir}")
    write_to_file(results_file, f"Training photo count: {len(train_records)}")
    write_to_file(results_file, f"Validation photo count: {len(val_records)}")
    write_to_file(results_file, f"Test photo count: {len(test_records)}")
    write_to_file(results_file, f"Epochs: {cfg.EPOCHS}")
    write_to_file(results_file, f"Label counts: {dict(label_counts)}")

    # Ensure test images are available under the main IMAGE_DIR/test/ folder
    test_out_dir = os.path.join(cfg.IMAGE_DIR, "test")
    os.makedirs(test_out_dir, exist_ok=True)
    copied = 0
    for rec in test_records:
        src = rec.get("image_path")
        if not src or not os.path.exists(src):
            continue
        dst = os.path.join(test_out_dir, os.path.basename(src))
        try:
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            # update in-memory record so evaluation finds the copied file
            rec["image"] = os.path.basename(dst)
            rec["image_path"] = dst
            copied += 1
        except Exception:
            continue

    write_to_file(results_file, f"Copied {copied} test images to {test_out_dir}")

    load_dir = None
    if getattr(cfg, "USE_BEST_MODEL", False):
        latest_path = os.path.join(cfg.BEST_MODEL_SAVE_DIR, "latest.txt")
        if os.path.exists(latest_path):
            with open(latest_path, "r", encoding="utf-8") as latest_file:
                candidate_dir = latest_file.read().strip()
            if candidate_dir:
                load_dir = candidate_dir
        elif os.path.exists(cfg.BEST_MODEL_SAVE_DIR):
            load_dir = cfg.BEST_MODEL_SAVE_DIR

    model, processor, device = load_VLM(cfg.VLM_NAME, cfg.DEVICE, load_dir=load_dir)

    for param in model.vision_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
    )

    if load_dir:
        write_to_file(results_file, f"Loading best model from {cfg.BEST_MODEL_SAVE_DIR}")
        write_to_file(results_file, "USE_BEST_MODEL is enabled; skipping training and evaluating the loaded checkpoint only")
        loss_history = []
        val_metrics = None
    else:
        model, loss_history, epoch_times, val_metrics = train(
            model,
            processor,
            optimizer,
            train_records,
            device,
            results_file,
            val_data=val_records,
            metric_n=experiment_cfg.metric_n,
        )
        write_to_file(results_file, f"Total training time: {sum(epoch_times):.2f}s")
        if loss_history:
            final_loss = loss_history[-1]
            best_bleu = max(val_metrics[0]) if val_metrics and val_metrics[0] else 0.0
            best_cider = max(val_metrics[1]) if val_metrics and val_metrics[1] else 0.0
            write_to_file(results_file, f"Training summary -> epochs: {len(loss_history)}, final train loss: {final_loss:.4f}, best BLEU-{experiment_cfg.metric_n}: {best_bleu:.4f}, best CIDEr-{experiment_cfg.metric_n}: {best_cider:.4f}")
        plot_training_curves(loss_history, epoch_times, val_metrics=val_metrics, metric_n=experiment_cfg.metric_n)

    write_to_file(results_file, f"TEST set JSON: {cfg.OPENIMAGES_TEST_JSON}")
    write_to_file(results_file, f"Loaded {len(test_records)} TEST samples")
    evaluate(model, processor, test_records, device, results_file, metric_n=experiment_cfg.metric_n)

    preview_path = os.path.join(cfg.OUTPUT_DIR, "openimages_test_preview.png")
    _save_test_preview(
        model,
        processor,
        test_records,
        device,
        results_file,
        experiment_cfg.metric_n,
        preview_path,
        sample_count=15,
    )

