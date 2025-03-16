from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from .config import IMAGE_SIZE, DatasetSplit
from .data.imagewoof import get_imagewoof_dataset
from .models.resnet18 import create_model

DATALOADER_ARGS = dict(
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

FROZEN_EPOCHS = 10
FROZEN_LR = 1e-3
UNFROZEN_EPOCHS = 10
UNFROZEN_LR = 1e-5


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(train_loader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)
        loss, accuracy = total_loss / total_samples, total_correct / total_samples
        pbar.set_postfix(loss=loss, accuracy=accuracy)
    return loss, accuracy


def val_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            loss, accuracy = total_loss / total_samples, total_correct / total_samples
            pbar.set_postfix(loss=loss, accuracy=accuracy)
    return loss, accuracy


def main() -> None:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = create_model(frozen=True)
    model = model.to(device)
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
