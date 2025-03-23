import torch
import torch.nn as nn
from optuna.trial import FixedTrial

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import Conv2dDecomposeMethod, decompose_model, is_decomposeable_conv2d, is_decomposeable_linear
from .models import create_model
from .utils.training import evaluate, get_device


def main() -> None:
    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True)
    model.load_state_dict(torch.load(MODEL_PATH))

    hparams = {}
    for fullname, m in model.named_modules():
        if is_decomposeable_conv2d(m):
            hparams[f"decompose_skip {fullname}"] = False
            hparams[f"decompose_rank_factor {fullname}"] = 1.0
            hparams[f"decompose_method {fullname}"] = Conv2dDecomposeMethod.TUCKER
        if is_decomposeable_linear(m):
            hparams[f"decompose_skip {fullname}"] = False
            hparams[f"decompose_rank_factor {fullname}"] = 1.0

    trail = FixedTrial(hparams)

    device = get_device()
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))
    decompose_model(model, trail, do_calculation=True)
    print(sum(p.numel() for p in model.parameters()))

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")


if __name__ == "__main__":
    main()
