import json

import torch
from optimum.quanto import Calibration, freeze, qint4, qint8, quantization_map, quantize, requantize

from .config import DATALOADER_ARGS
from .data.imagewoof import DatasetSplit, get_imagewoof_dataset
from .models.resnet18 import create_model
from .training import get_device, train_one_epoch, val_one_epoch


def main():
    device = get_device()

    model = create_model(from_pretrained=False, frozen=False)
    model.load_state_dict(torch.load("runs/Mar16_23-43-58_FredBill/model.pth"))
    model.to(device)

    test_data = get_imagewoof_dataset(DatasetSplit.TEST)[0]
    test_loader = torch.utils.data.DataLoader(test_data, **DATALOADER_ARGS)

    # print("Original model:")
    # test_loss, test_acc = val_one_epoch(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    # print(f"{test_loss=} {test_acc=}")

    quantize(model, weights=qint4, activations=qint8)
    print("Calibrating...")
    model.to(device)
    with Calibration():
        # TODO: use training data instead of test data
        val_one_epoch(model, test_loader, torch.nn.CrossEntropyLoss(), device)

    print("Calibrated:")
    test_loss, test_acc = val_one_epoch(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"{test_loss=} {test_acc=}")

    print("Freezed model:")
    freeze(model)
    test_loss, test_acc = val_one_epoch(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"{test_loss=} {test_acc=}")

    print("Serializing...")
    torch.save(model.state_dict(), "runs/Mar16_23-43-58_FredBill/model_quant.pth")
    with open("runs/Mar16_23-43-58_FredBill/quantization_map.json", "w") as f:
        json.dump(quantization_map(model), f)

    print("Deserializing...")
    state_dict = torch.load("runs/Mar16_23-43-58_FredBill/model_quant.pth")
    with open("runs/Mar16_23-43-58_FredBill/quantization_map.json") as f:
        q_map = json.load(f)
    model_reloaded = create_model(from_pretrained=False, frozen=False)
    requantize(model_reloaded, state_dict, q_map, device)

    print("Reloaded model:")
    test_loss, test_acc = val_one_epoch(model_reloaded, test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"{test_loss=} {test_acc=}")  # TODO: why did performance increase?


if __name__ == "__main__":
    main()
