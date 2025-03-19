from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from .config import IMAGE_SIZE, MODEL_NAME, DatasetSplit
from .data.imagewoof import get_imagewoof_dataloader
from .models import create_model
from .utils.EarlyStopping import EarlyStopping
from .utils.training import evaluate, get_device, train_one_epoch

LR = 1e-3
MIN_EPOCHS = 10
MAX_EPOCHS = 500


def main() -> None:
    device = get_device()
    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False)
    model.to(device)
    summary(model, input_size=(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(torch.device(device).type)
    early_stopping = EarlyStopping(patience=10, min_delta=0.0)

    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=6)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=6)
    with SummaryWriter() as writer:
        for epoch in range(1, MAX_EPOCHS + 1):
            print(f"Epoch {epoch}")
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            if early_stopping(val_loss) and epoch >= MIN_EPOCHS:
                print("Early stopping")
                break
            scheduler.step(val_loss)
            print(f"Learning rate: {scheduler.get_last_lr()}")

    torch.save(model.state_dict(), Path(writer.log_dir) / "model.pth")


if __name__ == "__main__":
    main()
