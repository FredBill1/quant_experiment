from enum import StrEnum, auto
from pathlib import Path

CWD = Path(__file__).parent.parent.resolve()
DATASETS_DIR = CWD / "datasets"


class DatasetSplit(StrEnum):
    TRAIN = auto()
    VAL = auto()


IMAGEWOOF_DIR = DATASETS_DIR / "imagewoof2-160"
IMAGEWOOF_TRAIN_DIR = IMAGEWOOF_DIR / "train"
IMAGEWOOF_VAL_DIR = IMAGEWOOF_DIR / "val"
IMAGEWOOF_DIRS = {
    DatasetSplit.TRAIN: IMAGEWOOF_TRAIN_DIR,
    DatasetSplit.VAL: IMAGEWOOF_VAL_DIR,
}

IMAGE_SIZE = 160
