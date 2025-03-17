import torch
import torchvision.models as models

from ..config import IMAGEWOOF_NUM_CLASSES


def create_model(*, from_pretrained: bool = False, frozen: bool = False) -> torch.nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if from_pretrained else None)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, IMAGEWOOF_NUM_CLASSES)
    return model
