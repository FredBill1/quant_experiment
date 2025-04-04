from enum import StrEnum, auto

import tensorly as tl
import torch.nn as nn
from optuna.trial import Trial
from tqdm import tqdm

from .cp_decompose import cp_decompose
from .svd_decompose import svd_decompose
from .tucker_decompose import tucker_decompose


class Conv2dDecomposeMethod(StrEnum):
    TUCKER = auto()
    CP = auto()


# Limit the maximum rank for CP decomposition to avoid memory error
CP_DECOMPOSE_MAX_RANK = 512

MIN_DECOMPOSE_FACTOR = 0.1


def is_decomposeable_conv2d(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and m.groups == 1 and m.kernel_size != (1, 1)


def is_decomposeable_linear(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear) or (
        isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.stride == (1, 1) and m.padding == (0, 0) and m.dilation == (1, 1) and m.groups == 1
    )


def decompose_model(
    model: nn.Module,
    trail: Trial,
    do_calculation: bool,
    *,
    layerwise: bool = True,
    skip_linear: bool = False,
    verbose: bool = True,
) -> None:
    tl.set_backend("pytorch")
    if not layerwise:
        factor = trail.suggest_float("decompose_rank_factor", MIN_DECOMPOSE_FACTOR, 1.0)
        method = trail.suggest_categorical("decompose_method", [Conv2dDecomposeMethod.TUCKER, Conv2dDecomposeMethod.CP])

    named_modules = dict(model.named_modules())
    progress = tqdm(named_modules.items(), desc="Decomposing") if verbose else named_modules.items()
    for fullname, m in progress:
        m_new = None
        if is_decomposeable_conv2d(m):
            if layerwise:
                if trail.suggest_categorical(f"decompose_skip {fullname}", [True, False]):
                    continue
                factor = trail.suggest_float(f"decompose_rank_factor {fullname}", MIN_DECOMPOSE_FACTOR, 1.0)
                method = trail.suggest_categorical(f"decompose_method {fullname}", [Conv2dDecomposeMethod.TUCKER, Conv2dDecomposeMethod.CP])

            if method == Conv2dDecomposeMethod.CP:
                # The maximum rank such that the number of parameters does not increase
                max_rank = (
                    m.out_channels
                    * m.in_channels
                    * m.kernel_size[0]
                    * m.kernel_size[1]
                    // (m.out_channels + m.in_channels + m.kernel_size[0] + m.kernel_size[1])
                )
                rank = max(1, min(CP_DECOMPOSE_MAX_RANK, round(max_rank * factor)))
                if verbose:
                    tqdm.write(f"cp {fullname=} {rank=:.4f}")
                m_new = cp_decompose(m, rank, do_calculation)
            elif method == Conv2dDecomposeMethod.TUCKER:
                # The maximum factor such that the number of parameters does not increase
                a = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                b = m.in_channels**2 + m.out_channels**2
                c = -a
                delta = b**2 - 4 * a * c
                max_factor = (-b + delta**0.5) / (2 * a)

                shape = [m.out_channels, m.in_channels]
                new_ranks = [max(1, round(x * factor * max_factor)) for x in shape]
                if verbose:
                    tqdm.write(f"tucker {fullname=} {max_factor=:.4f} factor={factor*max_factor:.4f} {shape} -> {new_ranks}")
                m_new = tucker_decompose(m, tuple(new_ranks), do_calculation)

        elif is_decomposeable_linear(m):
            if skip_linear:
                continue
            if layerwise:
                if trail.suggest_categorical(f"decompose_skip {fullname}", [True, False]):
                    continue
                factor = trail.suggest_float(f"decompose_rank_factor {fullname}", 0.1, 1.0)

            shape = [m.out_features, m.in_features] if isinstance(m, nn.Linear) else [m.out_channels, m.in_channels]
            max_rank = shape[0] * shape[1] // (shape[0] + shape[1])  # The maximum rank such that the number of parameters does not increase
            max_rank = min(max_rank, min(shape))  # The maximum rank is the minimum of the two dimensions
            rank = max(1, round(max_rank * factor))
            if verbose:
                tqdm.write(f"svd {fullname=} {shape=} {rank=:.4f}")
            m_new = svd_decompose(m, rank, do_calculation)

        if m_new is not None:
            parts = fullname.rsplit(".", 1)
            parent = named_modules[parts[0]] if len(parts) == 2 else model
            setattr(parent, parts[-1], m_new)
