import torch as th

VLM_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
IMAGE_DIR = "data/images/"
CAPTIONS_FILE = "data/captions.json"
OUTPUT_DIR = "output/"