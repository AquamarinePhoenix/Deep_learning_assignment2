#%% Initialization

import os
import json
import torch as th
from dotenv import load_dotenv
import _modules.config as cfg
from _modules.VLM import load_VLM
from _modules.write import clear_file, write_to_file
from _modules.model import train, evaluate, validate
from _modules.plots import plot_loss_history

load_dotenv()
token = os.getenv("HF_TOKEN")
results_file = cfg.OUTPUT_DIR + "results.txt"
clear_file(results_file)

with open(cfg.CAPTIONS_TRAIN, "r", encoding="utf-8") as f:
    train_data = json.load(f)

write_to_file(results_file, f"Loaded {len(train_data)} training samples")

train_data, val_data = validate(train_data, results_file)

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

model, loss_history, val_history, val_bleu_history, epoch_times = train(model, processor, optimizer, train_data, device, results_file, val_data=val_data)

plot_loss_history(loss_history, val_history, val_bleu_history, epoch_times, save_path=cfg.OUTPUT_DIR + "training_metrics.png")
write_to_file(results_file, f"Total training time: {sum(epoch_times):.2f}s")

with open(cfg.CAPTIONS_TEST, "r", encoding="utf-8") as f:
    test_data = json.load(f)

write_to_file(results_file, f"Loaded {len(test_data)} test samples")

model = evaluate(model, processor, test_data, device, results_file)