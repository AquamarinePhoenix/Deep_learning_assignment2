#%% Initialization

import os
import _modules.config as cfg
from PIL import Image
from _modules.VLM import load_VLM
from _modules.write import clear_file, write_to_file
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
clear_file(cfg.OUTPUT_DIR + "results.txt")

model, processor, device = load_VLM(cfg.VLM_NAME, cfg.DEVICE)

image = Image.open(cfg.IMAGE_DIR + "test.jpg").convert("RGB")

inputs = processor(image, return_tensors="pt").to(device)
out = model.generate(**inputs, max_new_tokens=50)
caption = processor.decode(out[0], skip_special_tokens=True)

write_to_file(cfg.OUTPUT_DIR + "results.txt", "Caption: " + caption)
# %%
