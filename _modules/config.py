import torch as th

VLM_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
IMAGE_DIR = "data/images/"
CAPTIONS_TRAIN = "data/captions_train.json"
CAPTIONS_TEST = "data/captions_test.json"
OUTPUT_DIR = "output/"
EPOCHS = 1
MODEL_SAVE_DIR = "output/models/"