from copy import deepcopy
from typing import Literal, Optional

import torch


class EarlyStopping:
    def __init__(
        self,
        *,
        patience: int = 0,
        previous_best_loss: float = float("inf"),
        previous_best_state_dict: Optional[dict] = None,
    ) -> None:
        self._patience = patience
        self._counter = 0
        self._best_loss = previous_best_loss
        self._best_state_dict = previous_best_state_dict

    def __call__(self, loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        if loss < self._best_loss:
            self._best_loss = loss
            self._counter = 0
            if model is not None:
                self._best_state_dict = deepcopy(model.state_dict())
        else:
            self._counter += 1
        return self._counter >= self._patience

    def reset_counter(self) -> None:
        self._counter = 0

    @property
    def best_state_dict(self) -> dict:
        if self._best_state_dict is None:
            raise ValueError("model hasn't been passed to the __call__ method")
        return self._best_state_dict
