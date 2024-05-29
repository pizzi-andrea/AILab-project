# define global constant for model

from pathlib import Path

DATASET_PATH = Path('../GTSRB')
LABELS_PATH_TEST  = DATASET_PATH.joinpath('Test.csv')
LABELS_PATH_TRAIN = DATASET_PATH.joinpath('Train.csv')
IMGS_PATH_TEST = DATASET_PATH
IMGS_PATH_TRAIN = DATASET_PATH
NUM_CLASSES = 43
