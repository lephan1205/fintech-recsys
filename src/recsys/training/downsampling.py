"""Negative down-sampling with logit correction (He et al. 2014, "practical lessons").

Keep every clicked impression; keep a non-clicked impression with probability ``r``
(default 0.25).  The click tower then learns odds inflated by ``1/r``; the exact fix
at inference is ``z_true = z_train + log r`` (:attr:`HSTUPLERanker.logit_correction`).
This is applied to the *training data* (a keep mask over the batch), never as a loss
weight, so the towers stay proper scoring rules on the data they see.  The keep mask
has the batch's ``(B, K)`` shape and multiplies every ``mean_all`` term, which is
equivalent to dropping the rows while keeping tensor shapes static.
"""

from __future__ import annotations

import math

import torch


class NegativeDownsampler:
    def __init__(self, rate: float = 0.25, seed: int = 0) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        self.rate = rate
        self.generator = torch.Generator().manual_seed(seed)

    @property
    def logit_correction(self) -> float:
        """``log r`` to add to the trained click logit at inference."""
        return math.log(self.rate)

    def train_mask(self, y_click: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        """``(B, K)`` bool keep mask: all clicked rows, non-clicked rows w.p. ``rate``."""
        if self.rate >= 1.0:
            return candidate_mask.clone()
        u = torch.rand(y_click.shape, generator=self.generator)
        keep = (y_click > 0.5) | (u < self.rate)
        return keep & candidate_mask
