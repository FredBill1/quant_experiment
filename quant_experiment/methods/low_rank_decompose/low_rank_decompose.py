"""
Downloaded and modified from https://github.com/alibaba/MNN/blob/b3c288d212a74822f463e0733da1b495c6e1c256/tools/mnncompress/mnncompress/pytorch/decomposition.py

Adapted the CP decomposition from https://github.com/ruihangdu/Decompose-CNN/blob/864648fa24e6cd70c65e6ae5bb4541de362453fe/scripts/torch_cp_decomp.py
"""

from copy import deepcopy
from enum import StrEnum, auto
from pprint import pp

import numpy as np
import scipy
import tensorly as tl
import torch
import torch.nn as nn
from tensorly.decomposition import parafac, partial_tucker

from .VBMF import EVBMF


def get_module_parameter_num(module: nn.Module) -> int:
    nm = dict(module.named_modules())
    count_types = (nn.Conv2d, nn.Linear)

    num_params = 0
    for n, m in nm.items():
        if isinstance(m, count_types):
            if m.weight is not None:
                num_params += m.weight.numel()
            if m.bias is not None:
                num_params += m.bias.numel()

    return num_params


def get_align_channels(value, max_value, align_channels, minimal_ratio=0.0):
    res = value // align_channels * align_channels
    if res == 0:
        if align_channels <= max_value:
            res = align_channels
        else:
            res = max_value

    if (res / float(max_value)) < minimal_ratio:
        res = int(max_value * minimal_ratio)
        res = res // align_channels * align_channels
        if res == 0:
            if align_channels <= max_value:
                res = align_channels
            else:
                res = max_value

    return res


class DecomposeMethod(StrEnum):
    TUCKER = auto()
    CP = auto()


def low_rank_decompose(
    model: nn.Module,
    skip_layers: set[str] = set(),
    align_channels: int = 8,
    in_place: bool = False,
    tucker_cp_minimal_ratio: float = 0.25,
    reserved_singular_value_ratio: float = 0.5,
    decompose_method: DecomposeMethod = DecomposeMethod.TUCKER,
) -> nn.Module:
    """
    Parameters:
        model: nn.Module instance, trained float model
        skip_layers: set[str], names of layers to skip decomposition, must be nn.Conv2d or nn.Linear type, e.g. {"features.conv1"}
        align_channels: int, multiplier for channel alignment after decomposition
        in_place: whether to use the original model's memory space; if False, a deep copy of the original model will be made
        tucker_cp_minimal_ratio: float 0~1, minimum ratio of channels to retain in tucker/cp decomposition of convolutional layers
        reserved_singular_value_ratio: ratio of sum of preserved singular values to the total sum in SVD decomposition

    Returns:
        The decomposed model, an nn.Module instance
    """
    tl.set_backend("pytorch")

    origin_params_num = get_module_parameter_num(model)
    decompose_model = model
    if not in_place:
        decompose_model = deepcopy(model)

    def _decompose_module(module, name=""):
        for n, m in module.named_children():
            m_name = name + "." + n
            if name == "":
                m_name = n
            if not isinstance(m, (nn.Conv2d, nn.Linear)):
                _decompose_module(m, m_name)
            else:
                if m_name in skip_layers:
                    print("skip decomposition:", m_name)
                    continue

                if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1):
                    weight = m.weight.data.detach()

                    if m.in_channels <= align_channels or m.out_channels <= align_channels:
                        print("skip tucker for:", m_name, "weight shape:", weight.shape)
                        continue

                    u0 = tl.base.unfold(weight, 0)
                    u1 = tl.base.unfold(weight, 1)
                    res0 = EVBMF(u0.cpu().numpy())
                    res1 = EVBMF(u1.cpu().numpy())
                    rank0 = get_align_channels(res0[1].shape[0], m.out_channels, align_channels, tucker_cp_minimal_ratio)
                    rank1 = get_align_channels(res1[1].shape[1], m.in_channels, align_channels, tucker_cp_minimal_ratio)
                    ranks = [rank0, rank1]

                    match decompose_method:
                        case DecomposeMethod.TUCKER:
                            (core, [last, first]), _rec_errors = partial_tucker(weight, modes=[0, 1], rank=ranks, init="svd")
                            core_out, core_in = core.shape[:2]
                            print(f"tucker for {m_name}: {[m.in_channels, m.out_channels]} <===> {[core.shape[0], core.shape[1]]} ranks: {ranks}")
                        case DecomposeMethod.CP:
                            rank = max(ranks)
                            last, first, vertical, horizontal = parafac(weight, rank=rank, init="random")[1]
                            core = torch.stack([vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1) for i in range(rank)]).unsqueeze_(1)
                            core_out = core_in = rank
                            print(f"cp for {m_name}: {[m.in_channels, m.out_channels]} <===> {[core.shape[0], core.shape[1]]} rank: {rank}")

                    has_bias = m.bias is not None

                    first_layer = nn.Conv2d(in_channels=first.shape[0], out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, bias=False)

                    core_layer = nn.Conv2d(
                        in_channels=core_in,
                        out_channels=core_out,
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                        dilation=m.dilation,
                        groups=1 if decompose_method == DecomposeMethod.TUCKER else rank,
                        bias=False,
                    )

                    last_layer = nn.Conv2d(in_channels=last.shape[1], out_channels=last.shape[0], kernel_size=1, stride=1, padding=0, bias=has_bias)

                    if has_bias:
                        last_layer.bias.data = m.bias.data

                    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
                    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
                    core_layer.weight.data = core

                    # first_bn = nn.BatchNorm2d(first_layer.out_channels)
                    # core_bn = nn.BatchNorm2d(core_layer.out_channels)
                    # last_bn = nn.BatchNorm2d(last_layer.out_channels)

                    decomposed_layers = [first_layer, core_layer, last_layer]
                    setattr(module, n, nn.Sequential(*decomposed_layers))
                    continue

                if isinstance(m, nn.Linear) or (
                    isinstance(m, nn.Conv2d)
                    and m.kernel_size == (1, 1)
                    and m.stride == (1, 1)
                    and m.padding == (0, 0)
                    and m.dilation == (1, 1)
                    and m.groups == 1
                ):
                    weight = m.weight.data.detach().cpu().numpy()
                    squeeze_shape = weight.squeeze().shape
                    if len(squeeze_shape) != 2:
                        print("skip svd for", m_name, "weight shape:", weight.shape)
                        continue

                    if squeeze_shape[0] <= align_channels or squeeze_shape[1] <= align_channels:
                        print("skip svd for", m_name, "weight shape:", weight.shape)
                        continue

                    u, s, v = scipy.linalg.svd(weight.squeeze())
                    singular_value_sum = np.sum(s)
                    n_dim = 1
                    temp_sum = 0.0
                    for i in range(0, s.size):
                        temp_sum += s[i]
                        n_dim = i + 1
                        if temp_sum / singular_value_sum >= reserved_singular_value_ratio:
                            break
                    n_dim = get_align_channels(n_dim, s.size, align_channels)

                    has_bias = m.bias is not None

                    if isinstance(m, nn.Conv2d):
                        print("svd for", m_name, ":", [m.in_channels, m.out_channels], "<===>", [m.in_channels, n_dim, m.out_channels])
                        fc1_weight = (np.matmul(np.diag(s[0:n_dim]), v[0:n_dim, :])).reshape((n_dim, -1, 1, 1))
                        fc2_weight = u[:, 0:n_dim].reshape((-1, n_dim, 1, 1))
                        fc1 = nn.Conv2d(m.in_channels, n_dim, 1, bias=False)
                        fc2 = nn.Conv2d(n_dim, m.out_channels, 1, bias=has_bias)
                    else:
                        print("svd for", m_name, ":", [m.in_features, m.out_features], "<===>", [m.in_features, n_dim, m.out_features])
                        fc1_weight = np.matmul(np.diag(s[0:n_dim]), v[0:n_dim, :])
                        fc2_weight = u[:, 0:n_dim]
                        fc1 = nn.Linear(m.in_features, n_dim, bias=False)
                        fc2 = nn.Linear(n_dim, m.out_features, bias=has_bias)

                    fc1.weight.data = torch.Tensor(fc1_weight)
                    fc2.weight.data = torch.Tensor(fc2_weight)

                    if has_bias:
                        fc2.bias.data = m.bias.data

                    decomposed_layers = [fc1, fc2]
                    setattr(module, n, nn.Sequential(*decomposed_layers))
                    continue

    _decompose_module(decompose_model)

    decompose_model_params_num = get_module_parameter_num(decompose_model)

    detail = {
        "algorithm": "low_rank_decompose",
        "compression_rate": origin_params_num / decompose_model_params_num,
        "ori_model_size": origin_params_num * 4.0 / 1024.0 / 1024.0,
        "config": {
            "skip_layers": skip_layers,
            "align_channels": align_channels,
            "tucker_minimal_ratio": tucker_cp_minimal_ratio,
            "reserved_singular_value_ratio": reserved_singular_value_ratio,
        },
    }

    pp(detail)

    return decompose_model
