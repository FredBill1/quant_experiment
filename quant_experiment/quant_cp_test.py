import numpy as np
import torch
import torch.ao.quantization
import torch.nn.intrinsic.qat
from optuna.trial import FixedTrial
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver, PerChannelMinMaxObserver
from tqdm import tqdm

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .methods.low_rank_decompose.decompose_model import decompose_model
from .models import create_model
from .utils.training import evaluate, get_device


def main():
    device = get_device()

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)
    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=0, persistent_workers=False)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=0, persistent_workers=False)
    criterion = torch.nn.CrossEntropyLoss()

    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True)
    decompose_model(
        model,
        FixedTrial(
            {
                "decompose_rank_factor": np.float32(0.90001),
                "decompose_method": "cp",
            }
        ),
        do_calculation=False,
        layerwise=False,
        skip_linear=True,
    )
    ckpt_path = MODEL_PATH.parent / "search/cp_0.9.pth"
    model.load_state_dict(torch.load(ckpt_path))

    activation_observers = {
        "moving_average_min_max": MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        "histogram": HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    }

    weight_observers = {
        "min_max": MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine),
        "per_channel_min_max": PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine),
    }

    weight_observer = weight_observers["per_channel_min_max"]
    activation_observer = activation_observers["histogram"]

    qconfig = QConfig(
        activation=activation_observer,
        weight=weight_observer,
    )

    model.to(device)
    model.eval()
    model.qconfig = qconfig
    torch.ao.quantization.prepare(model, inplace=True)
    evaluate(model, train_loader, criterion, device)
    model.to("cpu")
    torch.ao.quantization.convert(model, inplace=True)
    test_loss, test_acc = evaluate(model, test_loader, criterion, "cpu")
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")


if __name__ == "__main__":
    main()
