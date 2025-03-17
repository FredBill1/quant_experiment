import torch
from optimum.quanto import QTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional


def get_device() -> str:
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(train_loader)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast(torch.device(device).type, enabled=scaler is not None):
            outputs = model(inputs)
            if isinstance(outputs, QTensor):
                outputs = outputs.dequantize()
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)
        loss, accuracy = total_loss / total_samples, total_correct / total_samples
        pbar.set_description(f"{loss=:.4f} {accuracy=:.4f}")
    return loss, accuracy


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(val_loader)
    with torch.inference_mode():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, QTensor):
                outputs = outputs.dequantize()
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            loss, accuracy = total_loss / total_samples, total_correct / total_samples
            pbar.set_description(f"{loss=:.4f} {accuracy=:.4f}")
    return loss, accuracy
