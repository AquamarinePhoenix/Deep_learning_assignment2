#%% Initialization

import os
import json
import torch as th
from dotenv import load_dotenv
import _modules.config as cfg
from _modules.VLM import load_VLM
from _modules.write import clear_file, write_to_file
from _modules.model import train, evaluate

load_dotenv()
token = os.getenv("HF_TOKEN")
results_file = cfg.OUTPUT_DIR + "results.txt"
clear_file(results_file)

with open(cfg.CAPTIONS_TRAIN, "r", encoding="utf-8") as f:
    train_data = json.load(f)

write_to_file(results_file, f"Loaded {len(train_data)} training samples")

model, processor, device = load_VLM(cfg.VLM_NAME, cfg.DEVICE)
optimizer = th.optim.AdamW(model.parameters(), lr=3e-5)

model = train(model, processor, optimizer, train_data, device, results_file)

with open(cfg.CAPTIONS_TEST, "r", encoding="utf-8") as f:
    test_data = json.load(f)

write_to_file(results_file, f"Loaded {len(test_data)} test samples")

model = evaluate(model, processor, test_data, device, results_file)