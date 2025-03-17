import torch
from optimum.quanto import freeze, qint8, quantize

from .models.resnet18 import create_model


def main():
    model = create_model(from_pretrained=False, frozen=False)
    model.load_state_dict(torch.load("runs/Mar16_23-43-58_FredBill/model.pth"))
    quantize(model, weights=qint8, activations=qint8)
    freeze(model)
    torch.save(model.state_dict(), "runs/Mar16_23-43-58_FredBill/quant_model.pth")


if __name__ == "__main__":
    main()
