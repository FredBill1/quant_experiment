from enum import StrEnum, auto
from pathlib import Path

CWD = Path(__file__).parent.parent.resolve()
DATASETS_DIR = CWD / "datasets"


class DatasetSplit(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    TRAIN_AND_VAL = auto()


IMAGEWOOF_DIR = DATASETS_DIR / "imagewoof2-160"
IMAGEWOOF_TRAIN_DIR = IMAGEWOOF_DIR / "train"
IMAGEWOOF_TEST_DIR = IMAGEWOOF_DIR / "val"

IMAGEWOOF_NUM_CLASSES = 10
IMAGE_SIZE = 160

SEED = 514

LOSS_EPS = 1e-5
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 15

DATALOADER_ARGS = dict(
    batch_size=64,
    num_workers=5,
    pin_memory=True,
    persistent_workers=True,
)

# MODEL_NAME = "mobilenet_v3_large"
MODEL_NAME = "resnet18"

MODEL_PATH = CWD / f"runs/{MODEL_NAME}/model.pth"
