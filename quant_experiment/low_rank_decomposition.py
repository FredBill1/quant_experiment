import torch

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.low_rank_decompose import DecomposeMethod, get_module_parameter_num, low_rank_decompose
from .models import create_model
from .utils.training import evaluate, get_device


def main() -> None:
    device = get_device()

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()

    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False)
    model.load_state_dict(torch.load(MODEL_PATH))

    param_num_before = get_module_parameter_num(model)
    model.to(device)
    low_rank_decompose(
        model,
        align_channels=8,
        in_place=True,
        tucker_cp_minimal_ratio=0.9,
        reserved_singular_value_ratio=0.9,
        decompose_method=DecomposeMethod.TUCKER,
    )
    model.cpu()
    param_num_after = get_module_parameter_num(model)

    print(f"{param_num_before=} {param_num_after=}, ratio: {param_num_after * 100 / param_num_before:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH.with_stem(MODEL_PATH.stem + "_low_rank_decompose"))
    model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")


if __name__ == "__main__":
    main()
