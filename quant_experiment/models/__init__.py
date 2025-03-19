import torch

from ..config import MODEL_NAME
from .mobilenet_v3_large import create_model as create_mobilenet_v3_large
from .resnet18 import create_model as create_resnet18

MAPPING = {
    "resnet18": create_resnet18,
    "mobilenet_v3_large": create_mobilenet_v3_large,
}


def create_model(
    name: str = MODEL_NAME,
    *,
    from_pretrained: bool = False,
    frozen: bool = False,
    quantable: bool = False,
    quantize: bool = False,
) -> torch.nn.Module:
    return MAPPING[name](from_pretrained=from_pretrained, frozen=frozen, quantable=quantable, quantize=quantize)
