#%% Run the __main__ script

import os
import json
import torch as th
from dotenv import load_dotenv
import _modules.config as cfg
from _modules.VLM import load_VLM
from _modules.write import clear_file, write_to_file
from _modules.model import train, evaluate
from _modules.plots import plot_training_curves

load_dotenv()
token = os.getenv("HF_TOKEN")
results_file = cfg.OUTPUT_DIR + "results.txt"
clear_file(results_file)

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

write_to_file(results_file, f"TRAIN set JSON: {cfg.CAPTIONS_TRAIN}")
write_to_file(results_file, f"Total TRAIN samples available: {num_all_train_samples}")
write_to_file(results_file, f"TRAIN_SUBSET_RATIO: {train_subset_ratio}")
write_to_file(results_file, f"Loaded {len(train_data)} TRAIN samples")

load_dir = cfg.BEST_MODEL_SAVE_DIR if getattr(cfg, 'USE_BEST_MODEL', False) and os.path.exists(cfg.BEST_MODEL_SAVE_DIR) else None
if load_dir:
    write_to_file(results_file, f"Loading best model from {cfg.BEST_MODEL_SAVE_DIR}")

model, processor, device = load_VLM(cfg.VLM_NAME, cfg.DEVICE, load_dir=load_dir)

for param in model.vision_model.parameters():
        param.requires_grad = False

optimizer = th.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.LEARNING_RATE
)

model, loss_history, epoch_times = train(model, processor, optimizer, train_data, device, results_file)
write_to_file(results_file, f"Total training time: {sum(epoch_times):.2f}s")

plot_training_curves(loss_history, epoch_times)

with open(cfg.CAPTIONS_TEST, "r", encoding="utf-8") as f:
    test_data = json.load(f)

write_to_file(results_file, f"TEST set JSON: {cfg.CAPTIONS_TEST}")
write_to_file(results_file, f"Loaded {len(test_data)} TEST samples")

#model = evaluate(model, processor, test_data, device, results_file)
# %%
