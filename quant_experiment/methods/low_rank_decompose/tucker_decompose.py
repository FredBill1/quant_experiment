import torch
import torch.nn as nn
from tensorly.decomposition import partial_tucker


def tucker_decompose(m: nn.Conv2d, ranks: tuple[int, int], do_calculation: bool) -> nn.Module:
    has_bias = m.bias is not None

    first_layer = nn.Conv2d(
        in_channels=m.in_channels,
        out_channels=ranks[1],
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )

    core_layer = nn.Conv2d(
        in_channels=ranks[1],
        out_channels=ranks[0],
        kernel_size=m.kernel_size,
        stride=m.stride,
        padding=m.padding,
        dilation=m.dilation,
        bias=False,
    )

    last_layer = nn.Conv2d(
        in_channels=ranks[0],
        out_channels=m.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=has_bias,
    )

    if do_calculation:
        if has_bias:
            last_layer.bias.data = m.bias.data

        weight = m.weight.data.detach()
        ret = partial_tucker(weight, modes=[0, 1], rank=ranks, init="svd")
        # there is a change in return type of partial_tucker in tensorly
        try:
            (core, (last, first)), _rec_errors = ret
        except ValueError:
            core, (last, first) = ret

        first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
        last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
        core_layer.weight.data = core

        assert first_layer.weight.shape == (ranks[1], m.in_channels, 1, 1)
        assert core_layer.weight.shape == (ranks[0], ranks[1], m.kernel_size[0], m.kernel_size[1])
        assert last_layer.weight.shape == (m.out_channels, ranks[0], 1, 1)

    return nn.Sequential(first_layer, core_layer, last_layer)
