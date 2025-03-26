from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from optuna.trial import Trial
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.qconfig import QConfig

if TYPE_CHECKING:
    import torch.ao.quantization


def ptq_prepare(model: nn.Module, trail: Trial) -> None:
    model.eval()

    # 设置量化后端
    backend = trail.suggest_categorical("backend", ["fbgemm", "qnnpack"])
    torch.backends.quantized.engine = backend

    # 选择激活值和权重的观察器
    act_observers = {"minmax": MinMaxObserver, "moving_avg_minmax": MovingAverageMinMaxObserver, "histogram": HistogramObserver}
    weight_observers = {
        "minmax": MinMaxObserver,
        "per_channel_minmax": PerChannelMinMaxObserver,
        "moving_avg_per_channel_minmax": MovingAveragePerChannelMinMaxObserver,
    }

    act_observer = act_observers[trail.suggest_categorical("act_observer", list(act_observers.keys()))]
    weight_observer = weight_observers[trail.suggest_categorical("weight_observer", list(weight_observers.keys()))]

    # 创建量化配置
    qconfig = QConfig(
        activation=act_observer.with_args(reduce_range=trail.suggest_categorical("reduce_range", [True, False])),
        weight=weight_observer.with_args(
            dtype=torch.qint8,
            qscheme=getattr(torch, trail.suggest_categorical("weight_qscheme", ["per_tensor_symmetric", "per_channel_symmetric"])),
        ),
    )

    # 设置默认的量化配置
    model.qconfig = qconfig

    model.fuse_model(is_qat=False)
    torch.ao.quantization.prepare(model, inplace=True)


def ptq_convert(model: nn.Module) -> None:
    torch.ao.quantization.convert(model.to("cpu"), inplace=True)


if __name__ == "__main__":
    from optuna.trial import FixedTrial
    from torchvision.models.quantization import resnet18

    configs = {
        "backend": "fbgemm",  # x86平台首选fbgemm，ARM平台使用qnnpack
        "act_observer": "histogram",  # 直方图观察器通常能提供更好的量化精度
        "weight_observer": "per_channel_minmax",  # 每通道量化通常优于每张量量化
        "reduce_range": True,  # 减少激活值范围可能提高精度
        "weight_qscheme": "per_channel_symmetric",  # 每通道对称量化通常效果较好
    }

    model = resnet18(pretrained=True)
    ptq_prepare(model, FixedTrial(configs))
    # dummy input
    model(torch.randn(1, 3, 224, 224))
    ptq_convert(model)
