from typing import TYPE_CHECKING

import torch

from .config import DATALOADER_ARGS
from .data.imagewoof import DatasetSplit, get_imagewoof_dataset
from .models.resnet18 import create_model
from .utils.training import train_one_epoch, val_one_epoch

if TYPE_CHECKING:
    import torch.ao.quantization

DEVICE = "cpu"  # PyTorch latest version 2.6 does not support quantization on CUDA yet


def main():
    test_data = get_imagewoof_dataset(DatasetSplit.TEST)[0]
    test_loader = torch.utils.data.DataLoader(test_data, **DATALOADER_ARGS)
    criterion = torch.nn.CrossEntropyLoss()

    def dynamic() -> None:
        print("Dynamic quantization")
        model = create_model(from_pretrained=False, frozen=False, quantable=True, quantize=False)
        model.load_state_dict(torch.load("runs/Mar16_23-43-58_FredBill/model.pth"))

        model_int8 = torch.ao.quantization.quantize_dynamic(
            model,
            qconfig_spec=None,
            dtype=torch.qint8,
        )
        model_int8.to(DEVICE)
        test_loss, test_acc = val_one_epoch(model_int8, test_loader, criterion, DEVICE)
        print(f"{test_loss=} {test_acc=}")

    dynamic()


if __name__ == "__main__":
    main()
