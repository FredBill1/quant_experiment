from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from optimum.quanto import quantization_map, requantize
from torch.serialization import FILE_LIKE


def quanto_save(model: nn.Module, path: FILE_LIKE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "quantization_map": quantization_map(model)}
    torch.save(ckpt, path)


def quanto_load(model: nn.Module, path: FILE_LIKE, device: Optional[torch.device] = None) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    requantize(model, ckpt["state_dict"], ckpt["quantization_map"], device)
    return model
