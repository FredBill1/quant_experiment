import torch
from mnncompress.pytorch import low_rank_decompose

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .models import create_model
from .utils.training import evaluate, get_device


def main() -> None:
    device = get_device()

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()

    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False)
    model.load_state_dict(torch.load(MODEL_PATH))

    low_rank_decompose(
        model,
        str(MODEL_PATH.with_name(MODEL_PATH.stem + "_low_rank_decompose.proto")),
        align_channels=8,
        tucker_minimal_ratio=0.9,
        reserved_singular_value_ratio=0.9,
        in_place=True,
    )
    torch.save(model.state_dict(), MODEL_PATH.with_stem(MODEL_PATH.stem + "_low_rank_decompose"))

    model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")


if __name__ == "__main__":
    main()
