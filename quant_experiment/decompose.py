import torch
import torch.nn as nn

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import Conv2dDecomposeMethod, decompose_model, decompose_model_hparam_space
from .models import create_model
from .utils.training import evaluate, get_device


def main() -> None:
    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True)
    model.load_state_dict(torch.load(MODEL_PATH))

    hparam_space = decompose_model_hparam_space(model)
    print(hparam_space.keys())

    hparams = {}
    for fullname, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1):
            hparams[f"decompose_skip {fullname}"] = False
            hparams[f"decompose_rank_factor {fullname}"] = 1.0
            hparams[f"decompose_method {fullname}"] = Conv2dDecomposeMethod.TUCKER
        if isinstance(m, nn.Linear) or (
            isinstance(m, nn.Conv2d)
            and m.kernel_size == (1, 1)
            and m.stride == (1, 1)
            and m.padding == (0, 0)
            and m.dilation == (1, 1)
            and m.groups == 1
        ):
            hparams[f"decompose_skip {fullname}"] = False
            hparams[f"decompose_rank_factor {fullname}"] = 1.0

    hparams = {name: domain.sample() for name, domain in hparam_space.items()}

    device = get_device()
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))
    decompose_model(model, hparams, do_calculation=True)
    print(sum(p.numel() for p in model.parameters()))

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")


if __name__ == "__main__":
    main()
