import optuna
import torch
import torch.nn as nn
from optuna.trial import Trial

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import decompose_model
from .models import create_model
from .utils.training import evaluate, get_device


def main():
    device = get_device()

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)
    criterion = nn.CrossEntropyLoss()

    def objective(trail: Trial) -> float:
        model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)
        decompose_model(model, trail, do_calculation=True)
        num_params = sum(p.numel() for p in model.parameters())
        trail.set_user_attr("num_params", num_params)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        trail.set_user_attr("loss_after_decompose", test_loss)
        trail.set_user_attr("acc_after_decompose", test_acc)

        return test_acc

    study_name = "decompose_tune"
    db_file = MODEL_PATH.parent / f"{study_name}.db"
    study = optuna.create_study(
        storage=f"sqlite:///{db_file.as_posix()}",
        direction="maximize",
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
