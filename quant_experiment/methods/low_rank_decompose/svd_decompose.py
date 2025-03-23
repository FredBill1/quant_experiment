import torch
import torch.nn as nn
from tensorly.tenalg.svd import truncated_svd


def svd_decompose(m: nn.Conv2d | nn.Linear, rank: int, do_calculation: bool) -> nn.Module:
    has_bias = m.bias is not None

    if isinstance(m, nn.Linear):
        fc1 = nn.Linear(m.in_features, rank, bias=False)
        fc2 = nn.Linear(rank, m.out_features, bias=has_bias)
    else:
        fc1 = nn.Conv2d(m.in_channels, rank, 1, bias=False)
        fc2 = nn.Conv2d(rank, m.out_channels, 1, bias=has_bias)

    if do_calculation:
        weight = m.weight.data.detach()
        u, s, v = truncated_svd(weight.squeeze(), n_eigenvecs=rank)

        if isinstance(m, nn.Linear):
            fc1.weight.data = (v.t() @ torch.diag(torch.sqrt(s))).t()
            fc2.weight.data = u @ torch.diag(torch.sqrt(s))
        else:
            fc1.weight.data = (v.t() @ torch.diag(torch.sqrt(s))).t().unsqueeze(-1).unsqueeze(-1)
            fc2.weight.data = (u @ torch.diag(torch.sqrt(s))).unsqueeze(-1).unsqueeze(-1)

    return nn.Sequential(fc1, fc2)
