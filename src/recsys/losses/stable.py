"""Numerically stable log-space helpers for the unified funnel loss.

``log1mexp(x) = log(1 - exp(x))`` for ``x <= 0`` is computed as ``log(-expm1(x))`` when
``x < -ln 2`` and ``log1p(-exp(x))`` otherwise (Mächler 2012).  This is what makes the
entire-space terms exact and gradient-safe at ``|z| = 30``: the product of sigmoids
underflows in probability space, and ``BCE_with_logits`` cannot be used because the
product of sigmoids is not a sigmoid of a sum.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LN2 = math.log(2.0)
#: ``log p`` is clamped below this so ``log(1 - p)`` stays finite when p -> 1.
LOG_P_MAX = -1e-6


def logsigmoid(z: torch.Tensor) -> torch.Tensor:
    """``log sigmoid(z)``, stable for any ``z``."""
    out: torch.Tensor = F.logsigmoid(z)
    return out


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """``log(1 - exp(x))`` for ``x <= 0`` (clamped to ``LOG_P_MAX``), stable everywhere."""
    x = x.clamp(max=LOG_P_MAX)
    small = x < -_LN2
    # evaluate both branches on safe inputs, then select
    a = torch.log(-torch.expm1(torch.where(small, x, torch.full_like(x, -_LN2 - 1.0))))
    b = torch.log1p(-torch.exp(torch.where(small, torch.full_like(x, -_LN2), x)))
    out: torch.Tensor = torch.where(small, a, b)
    return out


def bce_from_log_prob(log_p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-element ``-(y log p + (1 - y) log(1 - p))`` from ``log p`` (no reduction)."""
    out: torch.Tensor = -(y * log_p + (1.0 - y) * log1mexp(log_p))
    return out


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``sum(values * mask) / sum(mask)``; exactly 0 (with gradient) for an empty mask."""
    m = mask.to(values.dtype)
    denom = m.sum()
    if float(denom) == 0.0:
        return values.sum() * 0.0
    out: torch.Tensor = (values * m).sum() / denom
    return out
