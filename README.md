# Deep Learning Assignment 2: Lithuanian Image Captioning with Vision-Language Models

## Problem Definition

**Motivation:** Vision-language models (VLMs) like BLIP are trained on large English-centric datasets. They work well for English but often perform poorly on non-dominant languages like Lithuanian due to limited representation in pre-training data.

**The Problem:** How can we adapt a pre-trained VLM to generate **accurate image captions in Lithuanian**?

**Why This Matters:** Lithuanian is a low-resource language. By creating a curated dataset and fine-tuning a VLM on Lithuanian captions paired with images, we can teach the model to describe visual content in Lithuanian, enabling accessibility for Lithuanian speakers and advancing multilingual AI research.

## Goals & Objectives

### Core Task

1. **Preparing for training**
   - Collect and organize images across diverse semantic categories
   - Provide dual-language captions (English and Lithuanian) for each image
   - Ensure the dataset is manually curated and representative of real and sureal world visual content

2. **Fine-Tuning a Pre-trained VLM**
   - Adapt `Salesforce/blip-image-captioning-base` using our Lithuanian dataset
   - Use strategic fine-tuning: freeze vision encoder to preserve visual understanding, train the text decoder
   - Save the best-performing checkpoint based on training metrics

3. **Validating the model**
   - Evaluate the model on held-out test images
   - Generate Lithuanian captions for unseen images

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

### Dataset Categories & Statistics

| Category | Count | Examples |
|----------|-------|----------|
| `people/` | 14 | hairdresser, knight, family, cyclist |
| `animals/` | 7 | horse, dog, cat, sheep, snail |
| `nature/` | 5 | flowers, fireplace, trees, water, pine_cone |
| `objects/` | 16 | car, glasses, money, sculpture, garbage |
| `landscapes/` | 5 | ark, ice, nature, space |
| `miscellaneous/` | 8 | bubbles, star_wars, structure, game_cones |
| `furniture/` | 2 | bench, sofa |
| `tools/` | 6 | cuttlery, pencils, tools |
| `activities/` | 3 | sport, firebreathing |
| `body_part/` | 3 | eye, feet_nails |
| `clothing/` | 3 | hat, socks, dresses |
| `complex/` | 2 | people_car, woman_horse |
| `food/` | 5 | peas, egg |

### Example Dataset Entries

**Example 1: People - Simple Object**
```json
{
  "image": "people/cyclist.jpg",
  "caption_en": "a girl riding a bicycle with a virtual reality glasses on",
  "caption_lt": "mergaitė važiuoja dviračiu, užsidėjusi virtualios realybės akinius"
}
```

**Example 2: Nature - Scene Description**
```json
{
  "image": "nature/flowers.jpg",
  "caption_en": "wildflowers in a field on the higher ground at sunset",
  "caption_lt": "laukinės gėlės lauke aukštumoje saulėlydžio metu"
}
```

**Example 3: Complex Scenes - Multi-object Relationships**
```json
{
  "image": "complex/people_car.jpg",
  "caption_en": "two women lying hair down on top of a car hood",
  "caption_lt": "dvi moterys guli, pasileidusios plaukus, ant automobilio kapoto"
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
- **Organized 92 images** (79 train + 13 test) into semantic category subfolders
- **Created structured JSON captions** with category-relative image paths for both train and test
- **Validated coverage**: 100% of train/test images have corresponding caption entries
- **Ensured consistency**: No duplicate image keys, no missing file references
- **Verified split integrity**: No overlapping image paths between `captions_train.json` and `captions_test.json`

### Dataset Structure

```
data/
├── images/
│   ├── train/
│   │   ├── animals/        (7 images)
│   │   ├── people/         (14 images)
│   │   ├── nature/         (5 images)
│   │   ├── objects/        (16 images)
│   │   ├── [9 other categories]
│   │   └── ...
│   └── test/
│       ├── food/           (egg.jpg)
│       ├── objects/        (train.jpg)
│       └── people/         (girls_umbrella.jpg)
|       └── [10 other categories]
├── captions_train.json     (79 entries with category-relative image paths)
└── captions_test.json      (13 entries for evaluation)
```

### Training Infrastructure
- **Model initialized**: BLIP (`Salesforce/blip-image-captioning-base`)
- **Fine-tuning strategy**: Vision encoder frozen, text decoder trained
- **Training loop implemented**: Batch processing, loss tracking, epoch-based training
- **Validation strategy**: A held-out subset of the training captions is reserved as validation data, and we track validation F1-score and cosine similarity by comparing generated captions against the held-out references
- **Best model tracking**: Automatic checkpoint saving on minimum loss
- **Evaluation pipeline ready**: Test set prepared for caption generation and comparison

---

## The workflow

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
- `VAL_SPLIT`: fraction of training captions reserved for validating the parameters
- `TRAIN_SUBSET_RATIO` : fraction of data kept for training the model

### Why These Constants Were Chosen

- `LEARNING_RATE = 3e-5`: conservative fine-tuning rate to avoid damaging pre-trained BLIP weights on a small dataset.
- `BATCH_SIZE = 4`: fits typical student GPU memory while keeping gradient estimates reasonably stable.
- `EPOCHS = 10`: a rouch estimate before the overfitting starts and signal catches the noise.
- `NUM_BEAMS = 5`: balances quality and speed for short caption generation.
- `DO_SAMPLE = True` with `TEMPERATURE = 0.7`: keeps outputs temperature-sensitive while limiting overly random captions.
- `VAL_SPLIT = 0`: none of the intermediate parameter adjustment was set due to small sample size.

Validation is reported as F1-style token overlap between generated and reference Lithuanian captions as token-level F1-score, and a cosine similarity using CLIP embeddings. These two metrics provide complementary views of caption quality on a small dataset.

Practical tuning guideline:
- Lower `TEMPERATURE` (e.g., `0.4-0.6`) for safer, more deterministic captions.
- Higher `TEMPERATURE` (e.g., `0.8-1.0`) for more diverse but less stable phrasing.

> [!NOTE]
> The `TEMPERATURE` parameter was selected as `0.7` as an optimal trade-off between safety and diversity.

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

## Example Pipeline Results

![Training loss](output/training_loss.png)

We can observe the decline in loss which indicates a proper learning (learning parameter is set correctly) and at the end starting stabilizing, indicating a more probable overfitting for even higher epoch numbers.

![Sport run](data/images/test/activities/sport_run.jpg)

TEST activities/sport_run.jpg
Pred_LT: the men ' s 100m hurdles at the iaafc
Real_LT: keli sportininkai begioja arba šoka per kartį prie vandens
CLIP_sim_pred: 0.2111
CLIP_sim_gt:   0.2486
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Sheep](data/images/test/animals/sheep.jpg)

TEST animals/sheep.jpg
Pred_LT: eziukas ir baltas sukrautos ant grindu
Real_LT: trys baltos mažos avys stovi arba sėdi ant žalios trumpos žolės
CLIP_sim_pred: 0.2001
CLIP_sim_gt:   0.2019
Precision: 0.1667
Recall:    0.0909
F1:        0.1176

![Hands feet](data/images/test/body_part/hands_feet.jpg)

TEST body_part/hands_feet.jpg
Pred_LT: hands holding a baby ' s foot
Real_LT: dvi poros tamsiaodžių rankų, laikančių baltaodes mažas pėdas
CLIP_sim_pred: 0.3152
CLIP_sim_gt:   0.1945
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Sweater](data/images/test/clothing/sweater.jpg)

TEST clothing/sweater.jpg
Pred_LT: ralpho sweaters, blau
Real_LT: mėlynas megztinis baltame fone
CLIP_sim_pred: 0.3346
CLIP_sim_gt:   0.2056
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Mosaic](data/images/test/complex/mosaic.jpg)

TEST complex/mosaic.jpg
Pred_LT: vyras, leidziantis dumus is savo burnos
Real_LT: mozaika žmonių skirtingais kampais ir formomis, sudėtais ratu
CLIP_sim_pred: 0.2130
CLIP_sim_gt:   0.2024
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Lemons](data/images/test/food/lemons.jpg)

TEST food/lemons.jpg
Pred_LT: sliced lemons on a white background
Real_LT: dvi perpjautos ir viena pilna geltona citrina baltame fone
CLIP_sim_pred: 0.3240
CLIP_sim_gt:   0.1987
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Bed](data/images/test/furniture/bed.jpg)

TEST furniture/bed.jpg
Pred_LT: bed is made of wood
Real_LT: baltos patalynės lova kambaryje su lempa
CLIP_sim_pred: 0.3162
CLIP_sim_gt:   0.2021
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Nature](data/images/test/landscapes/nature.jpg)

TEST landscapes/nature.jpg
Pred_LT: medinis, laikantys ezero su keliu ir zaluma
Real_LT: natūralus kraštovaizdis, kurį sudaro tiltas ir tolumoje esanti žaluma
CLIP_sim_pred: 0.2042
CLIP_sim_gt:   0.1705
Precision: 0.1429
Recall:    0.1111
F1:        0.1250

![Rocks](data/images/test/miscellaneous/rocks.jpg)

TEST miscellaneous/rocks.jpg
Pred_LT: vyras, sedintys ant zemes
Real_LT: tamsūs akmenys, pažerti ant grindų su maža mėlyna sraige viduryje
CLIP_sim_pred: 0.2439
CLIP_sim_gt:   0.2493
Precision: 0.2500
Recall:    0.1000
F1:        0.1429

![Forest](data/images/test/nature/forest.jpg)

TEST nature/forest.jpg
Pred_LT: sunlight shining through the trees in the forest
Real_LT: medžiai miške su blankia saulės šviesa
CLIP_sim_pred: 0.2900
CLIP_sim_gt:   0.2153
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Tram](data/images/test/objects/tram.jpg)

TEST objects/tram.jpg
Pred_LT: blue and white train
Real_LT: mėlynos ir baltos spalvų tramvajus ant bėgių
CLIP_sim_pred: 0.2871
CLIP_sim_gt:   0.3009
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Woman](data/images/test/people/woman.jpg)

TEST people/woman.jpg
Pred_LT: a woman in a black dress is smiling at the camera
Real_LT: tamsiaplaukė mergina, besišypsanti su viena akimi užmerkta
CLIP_sim_pred: 0.2770
CLIP_sim_gt:   0.2194
Precision: 0.0000
Recall:    0.0000
F1:        0.0000

![Drill](data/images/test/tools/drill.jpg)

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
Interestingly enough, sometimes the predicted cosine similarity is higher than the ground truth's. The reason behind that is the dominance of english words in the prediction: if there are still english words dominant in the predicted caption, CLIP model fancies the predicted over the lithuanian.
Moreover, when recall and precision are higher than 0.01, it is apparent that mostly precision is higher than recall, which indicates that the model struggles to find the ground truth labels rather than the positives prediction accuracy itself.

---

## Further Research

-- Add more images to increase dataset diversity
-- Implement additional evaluation metrics (CIDEr, METEOR)
- Support multi-reference captions per image
- Test on other low-resource languages
- Compare fine-tuned vs. base model performance