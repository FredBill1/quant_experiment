from enum import StrEnum, auto
from pathlib import Path

CWD = Path(__file__).parent.parent.resolve()
DATASETS_DIR = CWD / "datasets"


class DatasetSplit(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


IMAGEWOOF_DIR = DATASETS_DIR / "imagewoof2-160"
IMAGEWOOF_TRAIN_DIR = IMAGEWOOF_DIR / "train"
IMAGEWOOF_TEST_DIR = IMAGEWOOF_DIR / "val"

IMAGEWOOF_NUM_CLASSES = 10
IMAGE_SIZE = 160

SEED = 514
