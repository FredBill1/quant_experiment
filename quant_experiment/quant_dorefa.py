import torch
from torch.utils.tensorboard import SummaryWriter

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.dorefa.quantize import prepare as dorefa_prepare
from .models import create_model
from .utils.EarlyStopping import EarlyStopping
from .utils.training import evaluate, get_device, train_one_epoch

QAT_MIN_EPOCHS = 10
QAT_MAX_EPOCHS = 200


def main() -> None:
    device = get_device()

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=2)
    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=6)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=6)
    criterion = torch.nn.CrossEntropyLoss()

    def qat() -> None:
        model = create_model(MODEL_NAME, from_pretrained=False, frozen=False)
        model.load_state_dict(torch.load(MODEL_PATH))
        dorefa_prepare(model, inplace=True)
        model.to(device)

        print("Before QAT")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"{test_loss=} {test_acc=}")

        print("QAT")
        with SummaryWriter(log_dir=MODEL_PATH.parent / "dorefa") as writer:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            scaler = torch.amp.GradScaler(torch.device(device).type)
            early_stopping = EarlyStopping(patience=10)
            for epoch in range(1, QAT_MAX_EPOCHS + 1):
                print(f"Epoch {epoch}")
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/accuracy", train_acc, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch)
                if early_stopping(val_loss, model) and epoch >= QAT_MIN_EPOCHS:
                    print("Early stopping")
                    break
                scheduler.step(val_loss)
                print(f"Learning rate: {scheduler.get_last_lr()}")

        print("After QAT")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"{test_loss=} {test_acc=}")

        torch.save(model.state_dict(), MODEL_PATH.with_stem(MODEL_PATH.stem + "_dorefa"))

    def test_qat() -> None:
        model = create_model(MODEL_NAME, from_pretrained=False, frozen=False)
        dorefa_prepare(model, inplace=True)
        model.load_state_dict(torch.load(MODEL_PATH.with_stem(MODEL_PATH.stem + "_dorefa")))
        model.to(device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"{test_loss=} {test_acc=}")

    qat()
    test_qat()


if __name__ == "__main__":
    main()
