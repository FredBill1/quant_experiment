from typing import Any

import tensorly as tl
import torch
import torch.nn as nn
from ray import tune
from tensorly.decomposition import parafac, partial_tucker
from tensorly.tenalg.svd import truncated_svd
from tqdm import tqdm

from .config import MODEL_NAME, MODEL_PATH, SEED
from .data.imagewoof import DatasetSplit, get_imagewoof_dataloader
from .models import create_model
from .utils.training import evaluate, get_device

CP_MAX_RANK = 512


def tucker_decompose(m: nn.Conv2d, ranks: list[int], calculate: bool) -> nn.Module:
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

    if calculate:
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


def cp_decompose(m: nn.Conv2d, rank: int, calculate: bool) -> nn.Module:
    has_bias = m.bias is not None

    pointwise_s_to_r_layer = torch.nn.Conv2d(
        in_channels=m.in_channels,
        out_channels=rank,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )

    depthwise_vertical_layer = torch.nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=(m.kernel_size[0], 1),
        stride=(m.stride[0], 1),
        padding=(m.padding[0], 0),
        dilation=(m.dilation[0], 1),
        groups=rank,
        bias=False,
    )

    depthwise_horizontal_layer = torch.nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=(1, m.kernel_size[1]),
        stride=(1, m.stride[1]),
        padding=(0, m.padding[1]),
        dilation=(1, m.dilation[1]),
        groups=rank,
        bias=False,
    )

    pointwise_r_to_t_layer = torch.nn.Conv2d(
        in_channels=rank,
        out_channels=m.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=has_bias,
    )

    if calculate:
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


def svd_decompose(m: nn.Conv2d | nn.Linear, rank: int, calculate: bool) -> nn.Module:
    has_bias = m.bias is not None

    if isinstance(m, nn.Linear):
        fc1 = nn.Linear(m.in_features, rank, bias=False)
        fc2 = nn.Linear(rank, m.out_features, bias=has_bias)
    else:
        fc1 = nn.Conv2d(m.in_channels, rank, 1, bias=False)
        fc2 = nn.Conv2d(rank, m.out_channels, 1, bias=has_bias)

    if calculate:
        weight = m.weight.data.detach()
        u, s, v = truncated_svd(weight.squeeze(), n_eigenvecs=rank)

        if isinstance(m, nn.Linear):
            fc1.weight.data = (v.t() @ torch.diag(torch.sqrt(s))).t()
            fc2.weight.data = u @ torch.diag(torch.sqrt(s))
        else:
            fc1.weight.data = (v.t() @ torch.diag(torch.sqrt(s))).t().unsqueeze(-1).unsqueeze(-1)
            fc2.weight.data = (u @ torch.diag(torch.sqrt(s))).unsqueeze(-1).unsqueeze(-1)

    return nn.Sequential(fc1, fc2)


def decompose_model(model: nn.Module, hparams: dict[str, Any]) -> None:
    tl.set_backend("pytorch")

    named_modules = dict(model.named_modules())
    for fullname, m in tqdm(named_modules.items()):
        m_new = None
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1):
            if hparams[f"decompose_skip {fullname}"]:
                continue
            factor = hparams[f"decompose_rank_factor {fullname}"]
            method = hparams[f"decompose_method {fullname}"]
            if method == "cp":
                # The maximum rank such that the number of parameters does not increase
                max_rank = (
                    m.out_channels
                    * m.in_channels
                    * m.kernel_size[0]
                    * m.kernel_size[1]
                    // (m.out_channels + m.in_channels + m.kernel_size[0] + m.kernel_size[1])
                )
                max_rank = min(max_rank, CP_MAX_RANK)
                rank = max(1, round(max_rank * factor))
                tqdm.write(f"cp {fullname=} {rank=}")
                m_new = cp_decompose(m, rank, calculate=True)
            else:
                # The maximum factor such that the number of parameters does not increase
                a = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                b = m.in_channels**2 + m.out_channels**2
                c = -a
                delta = b**2 - 4 * a * c
                max_factor = (-b + delta**0.5) / (2 * a)

                ranks = [m.out_channels, m.in_channels]
                new_ranks = [max(1, round(x * factor * max_factor)) for x in ranks]
                tqdm.write(f"tucker {fullname=} {max_factor=} {factor*max_factor=} {ranks} -> {new_ranks}")
                m_new = tucker_decompose(m, new_ranks, calculate=True)

        elif isinstance(m, nn.Linear) or (
            isinstance(m, nn.Conv2d)
            and m.kernel_size == (1, 1)
            and m.stride == (1, 1)
            and m.padding == (0, 0)
            and m.dilation == (1, 1)
            and m.groups == 1
        ):
            if hparams[f"decompose_skip {fullname}"]:
                continue
            factor = hparams[f"decompose_rank_factor {fullname}"]

            ranks = [m.out_features, m.in_features] if isinstance(m, nn.Linear) else [m.out_channels, m.in_channels]
            rank = max(1, round(min(ranks) * factor))
            m_new = svd_decompose(m, rank, calculate=True)

        if m_new is not None:
            parts = fullname.rsplit(".", 1)
            parent = named_modules[parts[0]] if len(parts) == 2 else model
            setattr(parent, parts[-1], m_new)


def main() -> None:
    model = create_model(MODEL_NAME, from_pretrained=False, frozen=False, quantable=True)
    model.load_state_dict(torch.load(MODEL_PATH))

    param_space = {}
    for fullname, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1):
            param_space[f"decompose_skip {fullname}"] = tune.choice([True, False])
            param_space[f"decompose_rank_factor {fullname}"] = tune.uniform(0.1, 1.0)
            param_space[f"decompose_method {fullname}"] = tune.choice(["tucker", "cp"])
        if isinstance(m, nn.Linear) or (
            isinstance(m, nn.Conv2d)
            and m.kernel_size == (1, 1)
            and m.stride == (1, 1)
            and m.padding == (0, 0)
            and m.dilation == (1, 1)
            and m.groups == 1
        ):
            param_space[f"decompose_skip {fullname}"] = tune.choice([True, False])
            param_space[f"decompose_rank_factor {fullname}"] = tune.uniform(0.1, 1.0)

    hparams = {}
    for fullname, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1):
            hparams[f"decompose_skip {fullname}"] = False
            hparams[f"decompose_rank_factor {fullname}"] = 1.0
            hparams[f"decompose_method {fullname}"] = "tucker"
        if isinstance(m, nn.Linear) or (
            isinstance(m, nn.Conv2d)
            and m.kernel_size == (1, 1)
            and m.stride == (1, 1)
            and m.padding == (0, 0)
            and m.dilation == (1, 1)
            and m.groups == 1
        ):
            hparams[f"decompose_skip {fullname}"] = True
            hparams[f"decompose_rank_factor {fullname}"] = 0.9

    device = get_device()
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))
    decompose_model(model, hparams)
    print(sum(p.numel() for p in model.parameters()))

    test_loader = get_imagewoof_dataloader(DatasetSplit.TEST, num_workers=0, persistent_workers=False)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{test_loss=} {test_acc=}")


if __name__ == "__main__":
    main()
