from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from .config import DATALOADER_ARGS, IMAGE_SIZE, DatasetSplit
from .data.imagewoof import get_imagewoof_dataset
from .models.resnet18 import create_model
from .training import get_device, train_one_epoch, val_one_epoch

FROZEN_EPOCHS = 10
FROZEN_LR = 1e-3
UNFROZEN_EPOCHS = 10
UNFROZEN_LR = 1e-5


def main() -> None:
    device = get_device()
    model = create_model(from_pretrained=True, frozen=True)
    model.to(device)
    summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN_LR)

    train_loader = DataLoader(get_imagewoof_dataset(DatasetSplit.TRAIN)[0], shuffle=True, **DATALOADER_ARGS)
    val_loader = DataLoader(get_imagewoof_dataset(DatasetSplit.VAL)[0], **DATALOADER_ARGS)
    with SummaryWriter() as writer:
        for epoch in range(1, FROZEN_EPOCHS + 1):
            print(f"Epoch {epoch}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)

        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=UNFROZEN_LR)

        for epoch in range(FROZEN_EPOCHS + 1, FROZEN_EPOCHS + UNFROZEN_EPOCHS + 1):
            print(f"Epoch {epoch}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)

    torch.save(model.state_dict(), Path(writer.log_dir) / "model.pth")


if __name__ == "__main__":
    main()
