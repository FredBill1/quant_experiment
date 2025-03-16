from typing import Optional

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .common import IMAGEWOOF_NUM_CLASSES, DatasetSplit
from .dataset import get_imagewoof_dataset


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
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
    if writer is not None:
        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/accuracy", accuracy, epoch)
    return loss, accuracy


def val_one_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
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
    if writer is not None:
        writer.add_scalar("val/loss", loss, epoch)
        writer.add_scalar("val/accuracy", accuracy, epoch)
    return loss, accuracy


def create_model() -> torch.nn.Module:
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, IMAGEWOOF_NUM_CLASSES)
    return model


def train() -> None:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = create_model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(
        get_imagewoof_dataset(DatasetSplit.TRAIN),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        get_imagewoof_dataset(DatasetSplit.VAL),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    with SummaryWriter() as writer:
        for epoch in range(1, 6):
            print(f"Epoch {epoch}")
            train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
            val_one_epoch(model, val_loader, criterion, device, epoch, writer)


if __name__ == "__main__":
    train()
