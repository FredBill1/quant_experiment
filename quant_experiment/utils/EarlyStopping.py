from copy import deepcopy
from typing import Optional

import torch

from ..config import EARLY_STOPPING_PATIENCE, LOSS_EPS, REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE


class EarlyStopping:
    def __init__(
        self,
        *,
        patience: int = EARLY_STOPPING_PATIENCE,
        previous_best_loss: float = float("inf"),
        previous_best_state_dict: Optional[dict] = None,
    ) -> None:
        self._patience = patience
        self._counter = 0
        self._best_loss = previous_best_loss
        self._best_state_dict = previous_best_state_dict

    def __call__(self, loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        if loss + LOSS_EPS < self._best_loss:
            print(f"EarlyStopping: Loss improved from {self._best_loss:.4f} to {loss:.4f}")
            self._best_loss = loss
            self._counter = 0
            if model is not None:
                self._best_state_dict = deepcopy(model.state_dict())
        else:
            self._counter += 1
            print(f"EarlyStopping: Loss did not improve from {self._best_loss:.4f}, counter={self._counter}")
        return self._counter >= self._patience

    def reset_counter(self) -> None:
        self._counter = 0

    @property
    def best_loss(self) -> float:
        return self._best_loss

    @property
    def best_state_dict(self) -> dict:
        if self._best_state_dict is None:
            raise ValueError("model hasn't been passed to the __call__ method")
        return self._best_state_dict

    @property
    def counter(self) -> int:
        return self._counter

    @staticmethod
    def create_lr_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            threshold=LOSS_EPS,
            threshold_mode="abs",
        )
