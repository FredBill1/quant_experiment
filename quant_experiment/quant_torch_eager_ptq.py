import torch
import torch.ao.quantization
import torch.nn.intrinsic.qat
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver, MovingAverageMinMaxObserver, PerChannelMinMaxObserver

from .config import MODEL_NAME, MODEL_PATH
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .models import create_model
from .utils.training import evaluate, get_device


def main():
    device = get_device()

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)
    train_loader = get_imagewoof_dataloader(DatasetSplit.TRAIN, num_workers=0, persistent_workers=False)
    val_loader = get_imagewoof_dataloader(DatasetSplit.VAL, num_workers=0, persistent_workers=False)
    criterion = torch.nn.CrossEntropyLoss()

    def test_qconfig(qconfig: QConfig) -> tuple[float, float]:
        model = create_model(MODEL_NAME, quantable=True)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.to(device)
        model.eval()
        model.qconfig = qconfig
        model.fuse_model(is_qat=False)
        torch.ao.quantization.prepare(model, inplace=True)
        evaluate(model, val_loader, criterion, device, max_step=1)  # calibration
        model.to("cpu")
        torch.ao.quantization.convert(model, inplace=True)
        return evaluate(model, test_loader, criterion, "cpu", max_step=1)

    activation_observers = {
        "moving_average_min_max": MovingAverageMinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        "histogram": HistogramObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    }

    weight_observers = {
        "min_max": MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
        "per_channel_min_max": PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric),
    }

    for activation_observer in activation_observers:
        for weight_observer in weight_observers:
            print(f"Activation observer: {activation_observer}, Weight observer: {weight_observer}")
            test_qconfig(QConfig(activation=activation_observers[activation_observer], weight=weight_observers[weight_observer]))


if __name__ == "__main__":
    main()
