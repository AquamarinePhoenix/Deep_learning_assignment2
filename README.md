# Deep Learning Assignment 2: Lithuanian Image Captioning with Vision-Language Models

## Problem Definition

**Motivation:** Vision-language models (VLMs) like BLIP are trained on large English-centric datasets. They work well for English but often perform poorly on non-dominant languages like Lithuanian due to limited representation in pre-training data.

**The Problem:** How can we adapt a pre-trained VLM to generate **accurate image captions in Lithuanian**?

**Why This Matters:** Lithuanian is a low-resource language. By creating a curated dataset and fine-tuning a VLM on Lithuanian captions paired with images, we can teach the model to describe visual content in Lithuanian, enabling accessibility for Lithuanian speakers and advancing multilingual AI research.

## Goals & Objectives

### Core Task

1. **Build an Original Dataset**
   - Collect and organize images across diverse semantic categories
   - Provide dual-language captions (English and Lithuanian) for each image
   - Ensure the dataset is manually curated and representative of real-world visual content

2. **Fine-Tune a Pre-trained VLM**
   - Adapt `Salesforce/blip-image-captioning-base` using our Lithuanian dataset
   - Use strategic fine-tuning (freeze vision encoder, train text decoder) to preserve visual understanding
   - Save the best-performing checkpoint based on training metrics

3. **Demonstrate & Validate**
   - Evaluate the model on held-out test images
   - Generate Lithuanian captions for unseen images

---
The second individual task requires creating a new, unique small dataset and using it to fine-tune a large language model (LLM) or vision–language model (VLM) for Lithuanian or another non-dominant language.

Main requirements
Dataset creation (core focus)

Create a new, original dataset (not directly reused from existing benchmarks).
The dataset must target Lithuanian or another low-resource language.
Clearly describe:
data collection or annotation process,
dataset structure and size,
why the dataset is useful.
Model fine-tuning

Fine-tune a pretrained LLM or VLM using the created dataset.
Document the training setup and chosen fine-tuning method.
Demonstration

During assessment, demonstrate the model on new test inputs provided by the instructor.
Explain how the dataset influenced the model’s behavior.
You must be able to clearly explain how the dataset was created, how it was used, and why it is suitable for a non-dominant language.

---

## Dataset Overview

### What We Built

A **Lithuanian Image Captioning Dataset** with:
- **51 training images** across 13 semantic categories
- **3 test images** for independent evaluation
- **Dual captions**: English descriptions and Lithuanian translations for each image
- **Category-based organization**: Images organized by semantic content for better learning signal

### Dataset Categories & Statistics

| Category | Count | Examples |
|----------|-------|----------|
| `people/` | 10 | girl, family, cyclist, dancers, photographer, rider |
| `animals/` | 5 | horse, dog, cat, sheep, snail |
| `nature/` | 5 | flowers, fireplace, trees, water, pine_cone |
| `objects/` | 5 | car, glasses, money, sculpture, garbage |
| `landscapes/` | 4 | ark, ice, nature, space |
| `miscellaneous/` | 4 | bubbles, star_wars, structure, game_cones |
| `furniture/` | 2 | bench, sofa |
| `tools/` | 3 | cuttlery, pencils, tools |
| `activities/` | 2 | sport, firebreathing |
| `body_part/` | 2 | eye, feet_nails |
| `clothing/` | 2 | hat, socks, dresses |
| `complex/` | 1 | people_car |
| `food/` | 1 | peas |

### Example Dataset Entries

**Example 1: Animals - Simple Object**
```json
{
  "image": "animals/cyclist.jpg",
  "caption_en": "a woman riding a bicycle",
  "caption_lt": "moteris važiuoja dviračiu"
}
```

**Example 2: Nature - Scene Description**
```json
{
  "image": "nature/flowers.jpg",
  "caption_en": "wildflowers in a field at sunset",
  "caption_lt": "laukinės gėlės lauke saulėlydžio metu"
}
```

**Example 3: Complex Scenes - Multi-object Relationships**
```json
{
  "image": "complex/people_car.jpg",
  "caption_en": "two women lying on top of a car",
  "caption_lt": "dvi moterys guli ant automobilio"
}
```

### Why This Dataset Is Useful

1. **Low-Resource Language Focus**: Specifically targets Lithuanian, a language with minimal representation in large pre-training datasets
2. **Semantic Diversity**: Spans 13 categories covering everyday objects, people, nature, and complex scenes
3. **Dual-Language Annotations**: Enables evaluation of both English and Lithuanian caption quality
4. **Manual Curation**: Every caption is carefully written to be accurate and natural, not auto-translated
5. **Category Structure**: Semantic organization helps the model learn relationships between similar visual concepts

---

## What We've Accomplished

### Dataset Preparation & Validation
- **Organized 54 images** (51 train + 3 test) into semantic category subfolders
- **Created structured JSON captions** with category-relative image paths for both train and test
- **Validated coverage**: 100% of train/test images have corresponding caption entries
- **Ensured consistency**: No duplicate image keys, no missing file references
- **Verified split integrity**: No overlapping image paths between `captions_train.json` and `captions_test.json` (0 train-test collisions)

### Dataset Structure

```
data/
├── images/
│   ├── train/
│   │   ├── animals/        (5 images)
│   │   ├── people/         (10 images)
│   │   ├── nature/         (5 images)
│   │   ├── objects/        (5 images)
│   │   ├── [9 other categories]
│   │   └── ...
│   └── test/
│       ├── food/           (egg.jpg)
│       ├── objects/        (train.jpg)
│       └── people/         (girls_umbrella.jpg)
├── captions_train.json     (51 entries with category-relative image paths)
└── captions_test.json      (3 entries for evaluation)
```

### Training Infrastructure
- **Model initialized**: BLIP (`Salesforce/blip-image-captioning-base`)
- **Fine-tuning strategy**: Vision encoder frozen, text decoder trained
- **Training loop implemented**: Batch processing, loss tracking, epoch-based training
- **Best model tracking**: Automatic checkpoint saving on minimum loss
- **Evaluation pipeline ready**: Test set prepared for caption generation and comparison

---

## How to Run

### 1. Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure (Optional)
Edit `_modules/config.py` to adjust:
- `EPOCHS`: training iterations
- `LEARNING_RATE`: optimizer step size
- `BATCH_SIZE`: samples per batch
- `NUM_BEAMS`: beam width for stronger candidate search during generation
- `DO_SAMPLE`: enables sampling mode for generation (`True` by default)
- `TEMPERATURE`: controls caption diversity when sampling is enabled

### Why These Constants Were Chosen

- `LEARNING_RATE = 3e-5`: conservative fine-tuning rate to avoid damaging pre-trained BLIP weights on a small dataset.
- `BATCH_SIZE = 4`: fits typical student GPU memory while keeping gradient estimates reasonably stable.
- `EPOCHS = 2`: practical baseline to verify learning signal quickly before longer runs.
- `NUM_BEAMS = 5`: balances quality and speed for short caption generation.
- `DO_SAMPLE = True` with `TEMPERATURE = 0.7`: keeps outputs temperature-sensitive while limiting overly random captions.

Practical tuning guideline:
- Lower `TEMPERATURE` (e.g., `0.4-0.6`) for safer, more deterministic captions.
- Higher `TEMPERATURE` (e.g., `0.8-1.0`) for more diverse but less stable phrasing.

### 3. Run Training & Evaluation
```bash
python main.py
```

Output logged to `output/results.txt`.

---

## Data Format Reference

Caption files are JSON arrays with this structure:

```json
{
  "image": "category/filename.jpg",
  "caption_en": "English description",
  "caption_lt": "Lithuanian description"
}
```

**Image paths** are relative to `data/images/train/` (training) or `data/images/test/` (evaluation).

---

## Project Structure

```
_modules/
  ├── config.py           # Hyperparameters and paths
  ├── VLM.py              # Model and processor loader
  ├── model.py            # Training and evaluation logic
  ├── dataset.py          # PyTorch Dataset
  └── write.py            # Logging utilities
data/
  ├── images/train/       # 51 training images in category subfolders
  ├── images/test/        # 3 test images
  ├── captions_train.json
  └── captions_test.json
output/
  ├── results.txt         # Training logs
  └── models/
      ├── final/          # Final model
      └── best/           # Best checkpoint
main.py                    # Entry point
```

---

## Key Implementation Details

### Fine-Tuning Strategy

```python
# Freeze vision encoder to preserve pre-trained visual features
for param in model.vision_model.parameters():
    param.requires_grad = False

# Train only the text decoder for Lithuanian captions
optimizer = th.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.LEARNING_RATE
)
```

This approach:
- **Preserves visual understanding** from pre-training
- **Adapts language generation** to Lithuanian
- **Prevents catastrophic forgetting** on small datasets

---

## Warnings & Cautions

> [!WARNING]
> **GPU Highly Recommended:** Fine-tuning without GPU will be extremely slow and may cause out-of-memory errors. Training on CPU: 2–4 hours/epoch. Training on GPU: 1–5 minutes/epoch.

> [!WARNING]
> **Disk Space Required:** Model checkpoints are 400+ MB. Ensure 2 GB available before training with `SAVE_BEST = True`.

> [!CAUTION]
> **Reproducibility:** Pin PyTorch, Transformers, and dependency versions for reproducible results across runs.

> [!CAUTION]
> **Never Commit Secrets:** Do not commit `.env` files or hardcoded tokens. Use GitHub Secrets or local environment variables.

---

## Troubleshooting

**"BlipProcessor.from_pretrained" fails during model loading**  
→ Ensure internet access to Hugging Face hub, or download the model manually and provide a local path.

**"Image file not found" error during training**  
→ Verify that image paths in JSON are relative to `data/images/train/` or `data/images/test/`, and that category subfolders exist.

**Training loss doesn't decrease**  
→ Check learning rate (default `3e-5` is safe), JSON formatting, and dataset size (51 images is minimal).

---

## Next Steps & Extensions

- Add more images to increase dataset diversity
- Implement evaluation metrics (BLEU, CIDEr, METEOR)
- Support multi-reference captions per image
- Test on other low-resource languages
- Compare fine-tuned vs. base model performance

---

## License

See the `LICENSE` file in the repository root.