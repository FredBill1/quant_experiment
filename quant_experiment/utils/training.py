from itertools import islice
from typing import Optional

import torch
from optimum.quanto import QTensor
from torch.utils.data import DataLoader
from tqdm.rich import tqdm


def get_device() -> str:
    return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def get_desc(prefix: str, loss: float, acc: float) -> str:
    return f"{prefix}[yellow]{loss=:.4f}[/yellow] [green]{acc=:.4f}[/green]"


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_step: Optional[int] = None,
    desc_prefix: str = "Train: ",
) -> tuple[float, float]:
<<<<<<< Updated upstream
    model.to(device)
=======
    model.to("mps")
>>>>>>> Stashed changes
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total = len(train_loader) if max_step is None else min(max_step, len(train_loader))
    train_loader = islice(train_loader, max_step) if max_step is not None and max_step < len(train_loader) else train_loader
    pbar = tqdm(train_loader, total=total, desc=get_desc(desc_prefix, 0.0, 0.0) + ":")
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

        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)
        loss, accuracy = total_loss / total_samples, total_correct / total_samples
        pbar.set_description(get_desc(desc_prefix, loss, accuracy))
    return loss, accuracy


def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    *,
    max_step: Optional[int] = None,
    desc_prefix: str = "Val:   ",
) -> tuple[float, float]:
    model.to(device)
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    total = len(val_loader) if max_step is None else min(max_step, len(val_loader))
    val_loader = islice(val_loader, max_step) if max_step is not None and max_step < len(val_loader) else val_loader
    pbar = tqdm(val_loader, total=total, desc=get_desc(desc_prefix, 0.0, 0.0) + ":")
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, QTensor):
                outputs = outputs.dequantize()
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            loss, accuracy = total_loss / total_samples, total_correct / total_samples
            pbar.set_description(get_desc(desc_prefix, loss, accuracy))
    return loss, accuracy
