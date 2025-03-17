from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from .config import IMAGE_SIZE, DatasetSplit
from .data.imagewoof import get_imagewoof_dataloader
from .models import create_model
from .utils.EarlyStopping import EarlyStopping
from .utils.training import evaluate, get_device, train_one_epoch

FROZEN_MIN_EPOCHS = 10
FROZEN_MAX_EPOCHS = 200
UNFROZEN_MIN_EPOCHS = 10
UNFROZEN_MAX_EPOCHS = 200

USE_AMP = True


def main() -> None:
    device = get_device()
    model = create_model(from_pretrained=True, frozen=True)
    model.to(device)
    summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    criterion = torch.nn.CrossEntropyLoss()

    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=6)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=6)
    with SummaryWriter() as writer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=10)
        scaler = torch.amp.GradScaler(torch.device(device).type) if USE_AMP else None

        for frozen_epoch in range(1, FROZEN_MAX_EPOCHS + 1):
            epoch = frozen_epoch
            print(f"Epoch {epoch}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            if early_stopping(val_loss, model) and epoch >= FROZEN_MIN_EPOCHS:
                print("Early stopping")
                break
            scheduler.step(val_loss)
            print(f"Learning rate: {scheduler.get_last_lr()}")

        model.load_state_dict(early_stopping.best_state_dict)
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        early_stopping.reset_counter()
        scaler = torch.amp.GradScaler(torch.device(device).type) if USE_AMP else None

        for unfrozen_epoch in range(1, UNFROZEN_MAX_EPOCHS + 1):
            epoch = frozen_epoch + unfrozen_epoch
            print(f"Epoch {epoch}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            if early_stopping(val_loss, model) and epoch >= UNFROZEN_MIN_EPOCHS:
                print("Early stopping")
                break
            scheduler.step(val_loss)
            print(f"Learning rate: {scheduler.get_last_lr()}")

    print(f"Saving model to {writer.log_dir}")
    torch.save(early_stopping.best_state_dict, Path(writer.log_dir) / "model.pth")


if __name__ == "__main__":
    main()
