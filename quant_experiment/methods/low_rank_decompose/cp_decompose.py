import torch
import torch.nn as nn
from tensorly.decomposition import parafac

from ...config import SEED


def cp_decompose(m: nn.Conv2d, rank: int, do_calculation: bool) -> nn.Module:
    has_bias = m.bias is not None

    pointwise_s_to_r_layer = nn.Conv2d(
        in_channels=m.in_channels,
        out_channels=rank,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )

    depthwise_vertical_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=(m.kernel_size[0], 1),
        stride=(m.stride[0], 1),
        padding=(m.padding[0], 0),
        dilation=(m.dilation[0], 1),
        groups=rank,
        bias=False,
    )

    depthwise_horizontal_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=(1, m.kernel_size[1]),
        stride=(1, m.stride[1]),
        padding=(0, m.padding[1]),
        dilation=(1, m.dilation[1]),
        groups=rank,
        bias=False,
    )

    pointwise_r_to_t_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=m.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=has_bias,
    )

    if do_calculation:
        if has_bias:
            pointwise_r_to_t_layer.bias.data = m.bias.data

        weight = m.weight.data.detach()
        last, first, vertical, horizontal = parafac(weight, rank=rank, init="random", random_state=SEED)[1]
        pointwise_s_to_r_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        depthwise_vertical_layer.weight.data = torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
        depthwise_horizontal_layer.weight.data = torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
        pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

        assert pointwise_s_to_r_layer.weight.shape == (rank, m.in_channels, 1, 1)
        assert depthwise_vertical_layer.weight.shape == (rank, 1, m.kernel_size[0], 1)
        assert depthwise_horizontal_layer.weight.shape == (rank, 1, 1, m.kernel_size[1])
        assert pointwise_r_to_t_layer.weight.shape == (m.out_channels, rank, 1, 1)

    return nn.Sequential(pointwise_s_to_r_layer, depthwise_vertical_layer, depthwise_horizontal_layer, pointwise_r_to_t_layer)
