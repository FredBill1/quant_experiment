import torch
import torchvision.models as models
import torchvision.models.quantization as qmodels

from ..config import IMAGEWOOF_NUM_CLASSES


def create_model(
    *,
    from_pretrained: bool = False,
    frozen: bool = False,
    quantable: bool = False,
    quantize: bool = False,
) -> models.ResNet | qmodels.QuantizableResNet:
    if quantize and not quantable:
        raise ValueError("Model must be quantable to be quantized")
    weights = None if not from_pretrained else qmodels.ResNet18_QuantizedWeights.DEFAULT if quantize else models.ResNet18_Weights.DEFAULT
    model = qmodels.resnet18(weights=weights, quantize=quantize) if quantable else models.resnet18(weights=weights)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, IMAGEWOOF_NUM_CLASSES)
    return model
