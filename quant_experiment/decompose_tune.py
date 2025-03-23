import math

import optuna
import torch
import torch.nn as nn
from optuna.trial import Trial

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import decompose_model
from .models import create_model
from .utils.EarlyStopping import EarlyStopping
from .utils.training import evaluate, get_device, train_one_epoch

DECOMPOSE_FINE_TUNE_MAX_EPOCHS = 80
DO_EARLY_STOP = False


def target(accuracy: float, size: float) -> float:
    # From "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
    # Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
    error = 1 - accuracy
    return error * math.log(size)


def main():
    device = get_device()

    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=6)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=6)
    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)
    criterion = nn.CrossEntropyLoss()

    def objective(trail: Trial) -> float:
        model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)
        decompose_model(model, trail, do_calculation=True)
        num_params = sum(p.numel() for p in model.parameters())
        trail.set_user_attr("num_params", num_params)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        trail.set_user_attr("loss_after_decompose", val_loss)
        trail.set_user_attr("acc_after_decompose", val_acc)

        objective_value = target(val_acc, num_params)
        trail.report(objective_value, 0)
        print(f"Objective value: {objective_value}")
        if trail.should_prune():
            raise optuna.TrialPruned()

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        scheduler = EarlyStopping.create_lr_scheduler(optimizer)
        early_stopping = EarlyStopping()
        scaler = torch.amp.GradScaler()
        for epoch in range(1, DECOMPOSE_FINE_TUNE_MAX_EPOCHS + 1):
            print(f"Epoch {epoch}")
            train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if early_stopping(val_loss, model) and DO_EARLY_STOP:
                print("Early stopping")
                break
            objective_value = target(val_acc, num_params)
            trail.report(objective_value, epoch)
            print(f"Objective value: {objective_value}")
            if trail.should_prune():
                raise optuna.TrialPruned()
            scheduler.step(val_loss)
            print(f"Learning rate: {scheduler.get_last_lr()}")
        model.load_state_dict(early_stopping.best_state_dict)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        trail.set_user_attr("loss_after_fine_tune", val_loss)
        trail.set_user_attr("acc_after_fine_tune", val_acc)

        return target(val_acc, num_params)

    study_name = "decompose_tune"
    db_file = MODEL_PATH.parent / f"{study_name}.db"
    study = optuna.create_study(
        storage=f"sqlite:///{db_file.as_posix()}",
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=DECOMPOSE_FINE_TUNE_MAX_EPOCHS,
        ),
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=100,
        n_jobs=1,
        gc_after_trial=True,
    )


if __name__ == "__main__":
    main()
