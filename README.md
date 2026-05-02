# Deep_learning_assignment2

This repository demonstrates a simple image captioning fine-tuning pipeline based on a vision-language model (VLM). The project uses `Salesforce/blip-image-captioning-base` by default and provides utilities for dataset preparation, training, evaluation, and saving the best fine-tuned model.

Task Description
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

**Contents**

- `main.py`: entrypoint — loads data, model, runs training and evaluation.
- `_modules/config.py`: runtime configuration (paths, flags, hyperparams).
- `_modules/VLM.py`: model & processor loader (supports loading local best model directory).
- `_modules/model.py`: training and evaluation routines (saves final + optionally best model).
- `_modules/dataset.py`: dataset loader and collate function.
- `_modules/write.py`: simple logging helpers (`write_to_file`, `clear_file`).
- `data/`: images and caption files.
	- `images/train/` and `images/test/` — image files.
	- `captions_train.json` and `captions_test.json` — caption lists (one object per image).
- `output/`: training outputs, saves, and `results.txt`.

**Quick Start**

1. Install requirements (recommended inside a virtualenv):

```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token (if required for downloading models):

```bash
export HF_TOKEN=your_token_here    # Windows PowerShell: $Env:HF_TOKEN = "your_token_here"
```

3. Run training and evaluation:

```bash
python main.py
```

By default `main.py` loads training captions from the path in `_modules/config.py` (`CAPTIONS_TRAIN`) and test captions from `CAPTIONS_TEST`.

**Data format**

Caption files are JSON arrays of objects with the following fields:

```json
{
	"image": "filename.jpg",
	"caption_en": "an english caption",
	"caption_lt": "a lithuanian caption"
}
```

Image paths are resolved as `{IMAGE_DIR}/train/{image}` for training and `{IMAGE_DIR}/test/{image}` for evaluation. Update `IMAGE_DIR` in `_modules/config.py` if needed.

**Configuration**

Edit `_modules/config.py` to change runtime behaviour. Key variables:

- `VLM_NAME`: pretrained model name (default: `Salesforce/blip-image-captioning-base`).
- `DEVICE`: `cuda` if available, otherwise `cpu`.
- `IMAGE_DIR`: base images folder (default `data/images/`).
- `CAPTIONS_TRAIN`, `CAPTIONS_TEST`: paths to caption JSONs.
- `EPOCHS`: number of training epochs.
- `MODEL_SAVE_DIR`: directory where the final model is saved.
- `BEST_MODEL_SAVE_DIR`: directory where the best checkpoint (lowest avg loss) is saved.
- `SAVE_BEST`: when `True`, training saves the best-performing model to `BEST_MODEL_SAVE_DIR`.
- `USE_BEST_MODEL`: when `True` and a model exists at `BEST_MODEL_SAVE_DIR`, `main.py` will load it instead of the original pretrained name.

**How the pipeline works**

1. `main.py` loads training captions (`CAPTIONS_TRAIN`) and creates the model + processor via `_modules/VLM.py`.
2. `train()` in `_modules/model.py` runs the training loop, logging samples and average loss to `output/results.txt`.
	 - If `SAVE_BEST` is enabled, the best model (lowest epoch average loss) is saved to `BEST_MODEL_SAVE_DIR`.
3. After training, the final model is saved to `MODEL_SAVE_DIR`.
4. `main.py` loads `CAPTIONS_TEST` and runs `evaluate()` to generate predictions and compare to ground-truth captions.

**Using the best fine-tuned model**

1. Enable `USE_BEST_MODEL = True` in `_modules/config.py`.
2. Ensure `BEST_MODEL_SAVE_DIR` contains a saved model (created by training with `SAVE_BEST = True`).
3. Run `python main.py` — the code will prefer the local best model directory when it exists.

**Validation utilities**

It is recommended to validate that caption JSONs reference actual image files before training. A simple check verifies every `image` referenced in `captions_*.json` exists under the expected `images/train` or `images/test` folder.

**Git and model files**

This repository explicitly allows committing the best-model directory at `output/models/best/` (the `.gitignore` contains an exception), so you can include the best checkpoint in commits if desired. Note the following:

<warning>Committing model checkpoints</warning>

- Model checkpoints can be very large — committing them will bloat the repository and may be inappropriate for public repos.
- Prefer storing checkpoints in a model registry, cloud storage, or a separate LFS-enabled repository for large artifacts.

<caution>Privacy & tokens</caution>

- Do not commit secrets such as `HF_TOKEN` or `.env` files. Keep them in your CI secrets or local environment.

**Warnings & Cautions**

- <warning>GPU recommended:</warning> Fine-tuning the VLM without a GPU will be slow and may not fit in memory.
- <warning>Disk space:</warning> Check available disk space before saving model checkpoints — they can require multiple GBs.
- <caution>Reproducibility:</caution> For reproducible runs, set seeds and record versions of `transformers`, `torch`, and other dependencies.

**Troubleshooting**

- If `BlipProcessor.from_pretrained` or `BlipForConditionalGeneration.from_pretrained` fails, ensure internet access or provide a local model directory.
- If images cannot be opened, verify the `IMAGE_DIR` path and that the filenames in captions JSON match exactly (case-sensitive on some OSes).

**Extending / Next steps**

- Add support for multiple captions per image (augment dataset) or for different languages.
- Add evaluation metrics (BLEU, CIDEr, METEOR) and a validation loop to compute them after each epoch.

**License**

See the `LICENSE` file in the repository root.

---

If you'd like, I can also add a small `scripts/validate_captions.py` helper and a `requirements.txt` with pinned versions. Tell me which next step you prefer.