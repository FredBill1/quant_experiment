import numpy as np
import numpy.typing as npt
import torch
from torchvision import transforms as T

from ..config import IMAGE_SIZE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

TRAIN_TRANSFORMS = T.Compose(
    [
        T.RandomResizedCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

VAL_TRANSFORMS = T.Compose(
    [
        T.Resize(round(256 / 224 * IMAGE_SIZE)),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def tensor_to_numpy_image(tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
    tensor = tensor.permute(1, 2, 0)  # C, H, W -> H, W, C
    tensor = tensor * IMAGENET_STD + IMAGENET_MEAN
    return (torch.clamp(tensor * 255, 0, 255)).byte().numpy()
