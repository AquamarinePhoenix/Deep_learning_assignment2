import torch as th

EPOCHS = 2
TEMPERATURE = 0.7
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
NUM_BEAMS = 5
VAL_SPLIT = 0.1
TRAIN_SUBSET_RATIO = 1.0
OPENIMAGES_TRAIN_SIZE = 200
COCO_TRAIN_RATIO = 0.5
COCO_TRAIN_LABEL_RATIOS = {
	"horse": 0.45,
	"dog": 0.45,
	"background": 0.10,
}

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
DO_SAMPLE = True
USE_OPENIMAGES_EXPERIMENT = True
USE_EXPERIMENT = True

OPENIMAGES_DATA_DIR = "data/coco_simple/"
OPENIMAGES_RESULTS_FILE = "output/coco_results.txt"
OPENIMAGES_IMAGE_DIR = "data/coco_simple/images/"
OPENIMAGES_TRAIN_JSON = "data/coco_simple/captions_train.json"
OPENIMAGES_TEST_JSON = "data/coco_simple/captions_test.json"
OPENIMAGES_TARGET_CLASSES = ["horse", "dog", "background"]
OPENIMAGES_TRAIN_SPLIT = 0.8
OPENIMAGES_SPLIT_SEED = 42
OPENIMAGES_METRIC_N = 4
OPENIMAGES_MAX_SAMPLES_PER_CLASS = 100
OPENIMAGES_SOURCE_SAMPLE_POOL = 5000
OPENIMAGES_SOURCE_DATASET = "coco-2017"