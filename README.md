# Deep Learning Assignment 2: Lithuanian Image Captioning with Vision-Language Models

## Problem Definition

**Motivation:** Vision-language models (VLMs) like BLIP are trained on large English-centric datasets. They work well for English but often perform poorly on non-dominant languages like Lithuanian due to limited representation in pre-training data.

**The Problem:** How can we adapt a pre-trained VLM to generate **accurate image captions in Lithuanian**?

**Why This Matters:** Lithuanian is a low-resource language. By creating a curated dataset and fine-tuning a VLM on Lithuanian captions paired with images, we can teach the model to describe visual content in Lithuanian, enabling accessibility for Lithuanian speakers and advancing multilingual AI research.

## Goals & Objectives

### Core Tasks

1. **Dataset Preparation**
   - Collect and organize images across diverse semantic categories
   - Provide dual-language captions (English and Lithuanian) for each image
   - Ensure manual curation and representation of real-world visual content

2. **Model Fine-Tuning**
   - Adapt `Salesforce/blip-image-captioning-base` using the Lithuanian dataset
   - Use strategic fine-tuning: freeze vision encoder, train text decoder
   - Save best-performing checkpoint based on training metrics

3. **Model Validation**
   - Evaluate on held-out test images
   - Generate Lithuanian captions for unseen images
   - Report metrics: token F1-score and CLIP similarity

---
This task requires creating a new, unique small dataset and using it to fine-tune a VLM for Lithuanian language.

Main requirements
Dataset creation

Create a new, original dataset, not directly reused from existing benchmarks.
The dataset must target Lithuanian language.
Clearly describe:
data collection or annotation process,
dataset structure and size,
why the dataset is useful.
Model fine-tuning

Fine-tune a pretrained VLM using the created dataset.
Document the training setup and chosen fine-tuning method.
Demonstration

During assessment, demonstrate the model on new test inputs provided by the instructor.
Explain how the dataset influenced the model’s behavior.
You must be able to clearly explain how the dataset was created, how it was used, and why it is suitable for a lithuanian language.

---

## Dataset Overview

### Content

A **Lithuanian Image Captioning Dataset** with:
- **79 training images** across 13 semantic categories
- **13 test images** for each category
- **Dual captions**: English descriptions and Lithuanian translations for each image
- **Category-based organization**: Images organized by semantic content for better learning signal

### Dataset Categories

> [!CAUTION]
> **Class Imbalance Alert**: Dataset has significant category imbalance. `furniture/` and `complex/` have only 2 samples each, while `objects/` has 16. Consider this when interpreting per-category results.

| Category | Count | Examples |
|----------|-------|----------|
| `people/` | 14 | hairdresser, knight, family, cyclist |
| `animals/` | 7 | horse, dog, cat, sheep, snail |
| `nature/` | 5 | flowers, fireplace, trees, water, pine_cone |
| `objects/` | 16 | car, glasses, money, sculpture, garbage |
| `landscapes/` | 5 | ark, ice, nature, space |
| `miscellaneous/` | 8 | bubbles, star_wars, structure, game_cones |
| `furniture/` | 2 | bench, sofa |
| `tools/` | 6 | cutlery, pencils, tools |
| `activities/` | 3 | sport, firebreathing |
| `body_part/` | 3 | eye, feet_nails |
| `clothing/` | 3 | hat, socks, dresses |
| `complex/` | 2 | people_car, woman_horse |
| `food/` | 5 | peas, egg |

### Example Entries

**Example 1: People**
```json
{
  "image": "people/cyclist.jpg",
  "caption_en": "a girl riding a bicycle with a virtual reality glasses on",
  "caption_lt": "mergaitė važiuoja dviračiu, užsidėjusi virtualios realybės akinius"
}
```

**Example 2: Nature**
```json
{
  "image": "nature/flowers.jpg",
  "caption_en": "wildflowers in a field on the higher ground at sunset",
  "caption_lt": "laukinės gėlės lauke aukštumoje saulėlydžio metu"
}
```

**Example 3: Complex Scenes**
```json
{
  "image": "complex/people_car.jpg",
  "caption_en": "two women lying hair down on top of a car hood",
  "caption_lt": "dvi moterys guli, pasileidusios plaukus, ant automobilio kapoto"
}
```

### Why This Dataset Is Useful

> [!WARNING]
> **Small Dataset**: 79 training samples is limited for deep learning. Model may overfit or underfit. Increasing dataset size to 200+ samples recommended for robust performance.

- **Low-resource language**: Specifically targets Lithuanian with minimal pre-training representation
- **Semantic diversity**: 13 categories spanning everyday objects, people, nature, and complex scenes
- **Manual curation**: Accurate, natural captions not auto-translated
- **Organized structure**: Categorical organization helps model learn visual relationships

---

## Project Structure

```
Deep_learning_assignment2/
├── main.py                      # Main training and evaluation script
├── README.md                    # This file
├── LICENSE                      # Project license
├── _modules/                    # Core module package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration and hyperparameters
│   ├── dataset.py               # Dataset class and utilities
│   ├── model.py                 # Training and evaluation functions
│   ├── plots.py                 # Visualization and plotting
│   ├── VLM.py                   # Vision-Language Model loading
│   └── write.py                 # File I/O utilities
├── data/                        # Dataset directory
│   ├── captions_train.json      # Training captions (79 samples)
│   ├── captions_test.json       # Test captions (13 samples)
│   └── images/
│       ├── train/               # Training images (13 categories)
│       └── test/                # Test images (13 categories)
└── output/                      # Results and model checkpoints
    ├── results.txt              # Training and evaluation logs
    ├── training_loss.png        # Loss and time per epoch
    ├── training_metrics.png     # Validation metrics (if VAL_SPLIT > 0)
    └── models/                  # Saved model checkpoints
        ├── best/                # Best checkpoint by loss
        └── config.json, model.safetensors, etc.
```

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
  └── plots.py            # Visualizing the results
data/
  ├── images/train/       # 79 training images in category subfolders
  ├── images/test/        # 13 test images in category subfolders
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

> [!NOTE]
> **Low-Resource Language Challenge**: Lithuanian has minimal representation in BLIP's pre-training data. Performance will be inherently lower than English. This is expected behavior, not a bug.

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

### Hyperparameter Tuning Reference

| Parameter | Config Variable | Typical Default | Current Value | Description |
|-----------|-----------------|-----------------|----------------|-------------|
| **Training Epochs** | `EPOCHS` | 5-10 | 10 | Number of complete passes through training data |
| **Learning Rate** | `LEARNING_RATE` | 1e-4 to 1e-5 | 3e-5 | AdamW optimizer step size (conservative for fine-tuning) |
| **Batch Size** | `BATCH_SIZE` | 16-32 | 8 | Samples per gradient update (tuned for GPU memory) |
| **Validation Split** | `VAL_SPLIT` | 0.2 | 0.1 | Fraction of training data reserved for validation |
| **Train Subset Ratio** | `TRAIN_SUBSET_RATIO` | 1.0 | 1.0 | Fraction of training data to use (1.0 = all data) |
| **Beam Search Width** | `NUM_BEAMS` | 1 | 5 | Number of beams for beam search (1 = greedy) |
| **Sampling Mode** | `DO_SAMPLE` | False | True | Enable temperature-based sampling for diversity |
| **Temperature** | `TEMPERATURE` | 1.0 | 0.7 | Sampling temperature (lower = deterministic) |
| **Max New Tokens** | `max_new_tokens` | 50-128 | 50 | Maximum caption length (tokens) |
| **Repetition Penalty** | `repetition_penalty` | 1.0 | 1.2 | Penalty for repeated n-grams (>1.0 discourages repetition) |
| **No Repeat N-gram Size** | `no_repeat_ngram_size` | 0 | 3 | Block repeated n-grams of this size |
| **Early Stopping** | `early_stopping` | False | True | Stop search when no improvements found |

**Configuration Location:** [`_modules/config.py`](_modules/config.py)

**Key Tuning Guidelines:**
- ↑ **Increase `EPOCHS`** if model is underfitting (improving loss)
- ↓ **Decrease `LEARNING_RATE`** if training is unstable (spikes/NaN)
- ↓ **Decrease `BATCH_SIZE`** if out of memory; ↑ if training is slow
- ↑ **Increase `TEMPERATURE`** for more diverse captions; ↓ for more deterministic
- ↑ **Increase `NUM_BEAMS`** for higher quality (slower); ↓ for faster inference
- ↓ **Decrease `repetition_penalty`** if model avoids needed repetition

---

## Example Pipeline Results

![Training loss](output/training_loss.png)

We can observe the decline in loss which indicates a proper learning (learning parameter is set correctly) and at the end starting stabilizing, indicating a more probable overfitting for even higher epoch numbers.

<img src="data/images/test/activities/sport_run.jpg" alt="Sport run" width="320">

TEST activities/sport_run.jpg
Pred_LT: the men ' s 100m hurdles at the iaafc
Real_LT: keli sportininkai begioja arba šoka per kartį prie vandens
CLIP_sim_pred: 0.2111
CLIP_sim_gt:   0.2486
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/animals/sheep.jpg" alt="Sheep" width="320">

TEST animals/sheep.jpg
Pred_LT: eziukas ir baltas sukrautos ant grindu
Real_LT: trys baltos mažos avys stovi arba sėdi ant žalios trumpos žolės
CLIP_sim_pred: 0.2001
CLIP_sim_gt:   0.2019
Precision: 0.1667
Recall:    0.0909
F1:        0.1176

<img src="data/images/test/body_part/hands_feet.jpg" alt="Hands feet" width="320">

TEST body_part/hands_feet.jpg
Pred_LT: hands holding a baby ' s foot
Real_LT: dvi poros tamsiaodžių rankų, laikančių baltaodes mažas pėdas
CLIP_sim_pred: 0.3152
CLIP_sim_gt:   0.1945
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/clothing/sweater.jpg" alt="Sweater" width="320">

TEST clothing/sweater.jpg
Pred_LT: ralpho sweaters, blau
Real_LT: mėlynas megztinis baltame fone
CLIP_sim_pred: 0.3346
CLIP_sim_gt:   0.2056
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/complex/mosaic.jpg" alt="Mosaic" width="320">

TEST complex/mosaic.jpg
Pred_LT: vyras, leidziantis dumus is savo burnos
Real_LT: mozaika žmonių skirtingais kampais ir formomis, sudėtais ratu
CLIP_sim_pred: 0.2130
CLIP_sim_gt:   0.2024
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/food/lemons.jpg" alt="Lemons" width="320">

TEST food/lemons.jpg
Pred_LT: sliced lemons on a white background
Real_LT: dvi perpjautos ir viena pilna geltona citrina baltame fone
CLIP_sim_pred: 0.3240
CLIP_sim_gt:   0.1987
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/furniture/bed.jpg" alt="Bed" width="320">

TEST furniture/bed.jpg
Pred_LT: bed is made of wood
Real_LT: baltos patalynės lova kambaryje su lempa
CLIP_sim_pred: 0.3162
CLIP_sim_gt:   0.2021
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/landscapes/nature.jpg" alt="Nature" width="320">

TEST landscapes/nature.jpg
Pred_LT: medinis, laikantys ezero su keliu ir zaluma
Real_LT: natūralus kraštovaizdis, kurį sudaro tiltas ir tolumoje esanti žaluma
CLIP_sim_pred: 0.2042
CLIP_sim_gt:   0.1705
Precision: 0.1429
Recall:    0.1111
F1:        0.1250

<img src="data/images/test/miscellaneous/rocks.jpg" alt="Rocks" width="320">

TEST miscellaneous/rocks.jpg
Pred_LT: vyras, sedintys ant zemes
Real_LT: tamsūs akmenys, pažerti ant grindų su maža mėlyna sraige viduryje
CLIP_sim_pred: 0.2439
CLIP_sim_gt:   0.2493
Precision: 0.2500
Recall:    0.1000
F1:        0.1429

<img src="data/images/test/nature/forest.jpg" alt="Forest" width="320">

TEST nature/forest.jpg
Pred_LT: sunlight shining through the trees in the forest
Real_LT: medžiai miške su blankia saulės šviesa
CLIP_sim_pred: 0.2900
CLIP_sim_gt:   0.2153
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/objects/tram.jpg" alt="Tram" width="320">

TEST objects/tram.jpg
Pred_LT: blue and white train
Real_LT: mėlynos ir baltos spalvų tramvajus ant bėgių
CLIP_sim_pred: 0.2871
CLIP_sim_gt:   0.3009
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/people/woman.jpg" alt="Woman" width="320">

TEST people/woman.jpg
Pred_LT: a woman in a black dress is smiling at the camera
Real_LT: tamsiaplaukė mergina, besišypsanti su viena akimi užmerkta
CLIP_sim_pred: 0.2770
CLIP_sim_gt:   0.2194
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

<img src="data/images/test/tools/drill.jpg" alt="Drill" width="320">

TEST tools/drill.jpg
Pred_LT: a driller on a white background
Real_LT: geltonas grąžtas baltame fone
CLIP_sim_pred: 0.2573
CLIP_sim_gt:   0.2009
Precision: 0.0000
Recall:    0.0000
F1:        0.0000
TEST summary -> evaluated: 13, skipped: 0

Average F1: 0.0297

The average F1 score suggests that the predictions are way off and need better generalization - bigger dataset, less complex caption database for images.

> [!CAUTION]
> **English Dominance in Predictions**: Some generated captions still contain English words, causing CLIP to rate them higher than pure Lithuanian ground truth. This occurs because CLIP is trained on English-heavy datasets. Expect lower F1 scores but don't discard the model—it's a known limitation of the evaluation pipeline, not necessarily poor caption quality.

Interestingly enough, sometimes the predicted cosine similarity is higher than the ground truth's. The reason behind that is the dominance of english words in the prediction: if there are still english words dominant in the predicted caption, CLIP model fancies the predicted over the lithuanian.
Moreover, when recall and precision are higher than 0.01, it is apparent that mostly precision is higher than recall, which indicates that the model struggles to find the ground truth labels rather than the positives prediction accuracy itself.

---

## Further Research

> [!NOTE]
> These improvements could significantly boost model performance. Prioritize expanding the dataset first—more data has the highest ROI for low-resource language tasks.

- Add more images to increase dataset diversity (target 200+ for robust performance)
- Implement additional evaluation metrics (CIDEr, METEOR)
- Support multi-reference captions per image
- Test on other low-resource languages
- Compare fine-tuned vs. base model performance

## Submission Readiness Checklist

- Be ready to explain how the dataset was created, including where the images came from, how the Lithuanian captions were written or checked, and why the set is original.
- Be ready to state the dataset size and structure clearly: 79 training images, 13 test images, and 13 semantic categories.
- Be ready to explain the fine-tuning method: BLIP was pretrained first, then fine-tuned on the Lithuanian captions, with the vision encoder frozen and the text side updated.
- Be ready to show the saved checkpoints and training logs, especially the `output/results.txt` file and the saved model folders.
- Be ready to demonstrate the model on new instructor-provided inputs and describe how the dataset influenced the output.
- LoRA or DoRA is not required for this assignment; the current fine-tuning setup is already valid as a standard VLM fine-tuning approach.