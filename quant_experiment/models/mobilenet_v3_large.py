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
) -> models.MobileNetV3 | qmodels.QuantizableMobileNetV3:
    if quantize and not quantable:
        raise ValueError("Model must be quantable to be quantized")
    weights = (
        None
        if not from_pretrained
        else qmodels.MobileNet_V3_Large_QuantizedWeights.DEFAULT if quantize else models.MobileNet_V3_Large_Weights.DEFAULT
    )
    model = qmodels.mobilenet_v3_large(weights=weights, quantize=quantize) if quantable else models.mobilenet_v3_large(weights=weights)
    if frozen:
        for param in model.parameters():
            param.requires_grad = False
    classifier = model.classifier
    classifier[-1] = torch.nn.Linear(classifier[-1].in_features, IMAGEWOOF_NUM_CLASSES)
    classifier.requires_grad = True
    return model
