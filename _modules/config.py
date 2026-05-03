import torch as th

EPOCHS = 2
TEMPERATURE = 0.7
LEARNING_RATE = 3e-5
BATCH_SIZE = 4
NUM_BEAMS = 5
DO_SAMPLE = True
VAL_SPLIT = 0
TRAIN_SUBSET_RATIO = 0.05

VLM_NAME = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
IMAGE_DIR = "data/images/"
CAPTIONS_TRAIN = "data/captions_train.json"
CAPTIONS_TEST = "data/captions_test.json"
OUTPUT_DIR = "output/"
MODEL_SAVE_DIR = "output/models/"
SAVE_BEST = True
BEST_MODEL_SAVE_DIR = "output/models/best/"
USE_BEST_MODEL = False