"""Early stopping with a per-model criterion (D8): patience counted in evaluations."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

from torch import nn


@dataclass
class EarlyStopping:
    patience: int = 3
    mode: Literal["min", "max"] = "min"
    min_delta: float = 0.0
    best_value: float | None = None
    best_step: int = -1
    bad_evaluations: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)
    best_state: dict[str, Any] | None = None

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(self, value: float, step: int, model: nn.Module | None = None) -> bool:
        """Record an evaluation; returns True when it improved (and snapshots ``model``)."""
        self.history.append((step, value))
        if self.is_better(value):
            self.best_value, self.best_step, self.bad_evaluations = value, step, 0
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
            return True
        self.bad_evaluations += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_evaluations >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
