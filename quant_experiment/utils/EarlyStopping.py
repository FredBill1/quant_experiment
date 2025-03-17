from copy import deepcopy
from typing import Optional

import torch


class EarlyStopping:
    def __init__(self, *, patience: int = 0, min_delta: float = 0.0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._best_loss = float("inf")
        self._best_state_dict: Optional[dict] = None

    def __call__(self, loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        if self._best_loss - loss > self._min_delta:
            self._best_loss = loss
            self._counter = 0
            if model is not None:
                self._best_state_dict = deepcopy(model.state_dict())
        else:
            self._counter += 1
        return self._counter >= self._patience

    @property
    def best_state_dict(self) -> Optional[dict]:
        return self._best_state_dict
