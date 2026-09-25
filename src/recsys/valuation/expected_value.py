"""Expected value under calibrated funnel probabilities (D4.1, NumPy).

Serving never uses application statuses: ``P_funded = p̂1 · p̂2 · p̂3`` with *predicted,
calibrated* probabilities, ``EV = P_funded · partner_payout``, and ``E[amount]`` from
the ZILN tower.  A pending status only affects training labels (D1).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from recsys.losses.ziln_loss import ziln_expected_value

FloatArray = npt.NDArray[np.float64]


def p_funded(
    p_click: npt.ArrayLike, p_apply: npt.ArrayLike, p_approve: npt.ArrayLike
) -> FloatArray:
    """``p(click) · p(apply | click) · p(approve | apply)``, elementwise."""
    out: FloatArray = (
        np.asarray(p_click, dtype=np.float64)
        * np.asarray(p_apply, dtype=np.float64)
        * np.asarray(p_approve, dtype=np.float64)
    )
    return out


def expected_value(p_funded_: npt.ArrayLike, payout: npt.ArrayLike) -> FloatArray:
    """``EV_i = P_funded,i · partner_payout_i`` (dollars)."""
    out: FloatArray = np.asarray(p_funded_, dtype=np.float64) * np.asarray(payout, dtype=np.float64)
    return out


def expected_revenue(
    p_click: npt.ArrayLike, p_apply: npt.ArrayLike, p_approve: npt.ArrayLike, payout: npt.ArrayLike
) -> FloatArray:
    """Convenience: ``expected_value(p_funded(...), payout)``."""
    return expected_value(p_funded(p_click, p_apply, p_approve), payout)


def expected_amount(amount_logits: torch.Tensor) -> FloatArray:
    """ZILN ``E[Y] = π · exp(μ + σ² / 2)`` from ``(..., 3)`` logits -> NumPy float64."""
    ev: FloatArray = ziln_expected_value(amount_logits.detach()).cpu().numpy().astype(np.float64)
    return ev
