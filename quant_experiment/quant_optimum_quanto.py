import json

import torch
from optimum.quanto import Calibration, freeze, qint4, qint8, quantization_map, quantize, requantize

from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .models.resnet18 import create_model
from .utils.training import evaluate, get_device, train_one_epoch

FINETUNE_EPOCH = 5
FINETUNE_LR = 1e-5


def main():
    device = get_device()

    model = create_model(from_pretrained=False, frozen=False)
    model.load_state_dict(torch.load("runs/Mar16_23-43-58_FredBill/model.pth"))
    model.to(device)

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=2)
    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=6)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=6)
    criterion = torch.nn.CrossEntropyLoss()

    print("Original model:")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")

    quantize(model, weights=qint4, activations=qint8)
    print("Calibrating...")
    model.to(device)
    with Calibration():
        evaluate(model, train_loader, criterion, device)

    print("Calibrated:")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")

    print("Finetuning (Quantization-Aware Training)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    for epoch in range(1, FINETUNE_EPOCH + 1):
        print(f"Epoch {epoch}")
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        evaluate(model, val_loader, criterion, device)

    print("Frozen model:")
    freeze(model)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
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
    test_loss, test_acc = evaluate(model_reloaded, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")  # TODO: why did performance increase?


if __name__ == "__main__":
    main()
