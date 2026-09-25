"""Zero-Inflated LogNormal loss (Wang et al. 2019) and expected-value helpers.

Approved credit limits / funded loan sizes are (a) exactly zero for most impressions
and (b) heavy-tailed when positive.  ZILN models the amount as a mixture: with
probability ``1 - pi`` it is zero, otherwise it is ``LogNormal(mu, sigma)``.  A
three-headed output ``[p_logit, mu, sigma_raw]`` is trained with

    L = BCE(p, 1[y > 0]) + 1[y > 0] * ( log sigma + log y + (log y - mu)^2 / (2 sigma^2) + const )

and the expected amount is ``E[y] = pi * exp(mu + sigma^2 / 2)``.

Placement: the amount tower is trained in Stage 3 (4th PLE tower, resolved approved rows
only) but consumed only in Stage 4 — ``E[amount]`` feeds the net-user-benefit formulas;
the funnel probabilities never depend on it.

Numerical guards: ``sigma = softplus(raw)`` clamped to ``[1e-3, 10]``, ``log y`` is only
evaluated on positives, and the EV exponent is clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

SIGMA_MIN = 1e-3
SIGMA_MAX = 10.0
EV_EXP_MAX = 30.0


class ZILNHead(nn.Module):
    """``(..., input_dim)`` -> ``(..., 3)`` = ``[p_logit, mu, sigma_raw]``.

    ``mu_bias_init`` warm-starts the location head at (roughly) the mean log-amount of
    positives; without it the bias has to travel ~8 nats from zero and ``sigma`` inflates
    to absorb the early misfit.
    """

    def __init__(self, input_dim: int, mu_bias_init: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 3)
        with torch.no_grad():
            self.linear.bias[1] = mu_bias_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(x)
        return out


@dataclass
class ZILNLoss:
    total: torch.Tensor
    bce: torch.Tensor
    nll: torch.Tensor


def ziln_parameters(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``(..., 3)`` into ``(p_logit, mu, sigma)`` with ``sigma`` clamped."""
    p_logit, mu, sigma_raw = logits.unbind(dim=-1)
    sigma = F.softplus(sigma_raw).clamp(SIGMA_MIN, SIGMA_MAX)
    return p_logit, mu, sigma


def ziln_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> ZILNLoss:
    """``logits (..., 3)``, ``y (...)`` non-negative amounts -> mean loss over ``mask``."""
    p_logit, mu, sigma = ziln_parameters(logits)
    positive = y > 0
    bce = F.binary_cross_entropy_with_logits(p_logit, positive.to(logits.dtype), reduction="none")
    safe_log_y = torch.log(torch.where(positive, y, torch.ones_like(y)))
    nll = (
        torch.log(sigma)
        + 0.5 * math.log(2.0 * math.pi)
        + safe_log_y
        + (safe_log_y - mu) ** 2 / (2.0 * sigma**2)
    )
    nll = nll * positive.to(logits.dtype)
    m = torch.ones_like(y) if mask is None else mask.to(logits.dtype)
    denom = m.sum().clamp(min=1.0)
    bce_mean = (bce * m).sum() / denom
    nll_mean = (nll * m).sum() / denom
    return ZILNLoss(total=bce_mean + nll_mean, bce=bce_mean, nll=nll_mean)


def ziln_positive_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits[..., 0])


def ziln_expected_value(logits: torch.Tensor) -> torch.Tensor:
    """``E[y] = pi * exp(mu + sigma^2 / 2)`` with the exponent clamped for safety."""
    p_logit, mu, sigma = ziln_parameters(logits)
    return torch.sigmoid(p_logit) * torch.exp((mu + 0.5 * sigma**2).clamp(max=EV_EXP_MAX))
