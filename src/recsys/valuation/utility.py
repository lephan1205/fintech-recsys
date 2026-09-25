"""Multi-objective utility and suitability guardrails (D5), applied after calibration and
before PRM.

    U_i = P_funded,i · ( α · payout_i + (1 − α) · NB_i )        # dollars, α ∈ [0, 1]

Both terms are multiplied by the *same* funnel probability because the user only
realizes the benefit if approved and funded; both are in dollars, so ``α`` is
interpretable: ``α = 1`` ranks purely by expected partner revenue (``U = EV``), ``α = 0``
purely by expected user benefit, ``α = 0.5`` values one dollar of payout the same as one
dollar of user savings.  ``α`` is a **serving-time policy parameter** — not a loss
weight, not learned — chosen by the offline Pareto sweep (``scripts/pareto_sweep.py``).

Guardrails (hard rules that must not be learnable away; PRM only ever sees survivors):

* **Do-no-harm**: exclude any candidate with ``NB < −δ`` when at least one same-family
  candidate with ``NB ≥ 0`` exists in the slate; otherwise penalize by ``harm_penalty``.
* **Refinance sanity**: ``AUTO_REFINANCE`` / ``MORTGAGE`` only if ``apr < current_rate``.
* **Pending family policy** (D4): families with rule ``"penalize"`` cost
  ``pending_family_penalty`` dollars of NB when the user has an open application in that
  family (``"mask"`` families were removed by the eligibility engine already).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class UtilityConfig:
    alpha: float = 0.5  # platform-vs-user trade-off weight
    delta: float = 25.0  # do-no-harm threshold, dollars
    harm_penalty: float = 25.0  # utility penalty when exclusion is impossible, dollars
    pending_family_penalty: float = 15.0  # NB penalty for a second application in a family

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")


def utility(
    p_funded: npt.ArrayLike, payout: npt.ArrayLike, nb: npt.ArrayLike, alpha: float
) -> FloatArray:
    """``P_funded · (α · payout + (1 − α) · NB)`` elementwise, dollars."""
    pf = np.asarray(p_funded, dtype=np.float64)
    out: FloatArray = pf * (
        alpha * np.asarray(payout, dtype=np.float64)
        + (1.0 - alpha) * np.asarray(nb, dtype=np.float64)
    )
    return out


@dataclass
class GuardrailResult:
    utility: FloatArray  # (B, K) final utility fed to PRM (−inf on excluded slots)
    nb: FloatArray  # (B, K) NB after the pending-family penalty
    keep: BoolArray  # (B, K) survivors
    excluded_harm: BoolArray  # (B, K)
    penalized_harm: BoolArray  # (B, K)
    excluded_refinance: BoolArray  # (B, K)
    penalized_pending_family: BoolArray  # (B, K)


def apply_guardrails(
    p_funded: npt.ArrayLike,
    payout: npt.ArrayLike,
    nb: npt.ArrayLike,
    family_ids: npt.ArrayLike,
    candidate_mask: npt.ArrayLike,
    refinance_ok: npt.ArrayLike,
    pending_family_penalized: npt.ArrayLike,
    config: UtilityConfig | None = None,
) -> GuardrailResult:
    """All inputs ``(B, K)``.  ``pending_family_penalized`` marks candidates whose family is
    pending for the user *and* has the ``"penalize"`` rule."""
    cfg = config or UtilityConfig()
    pf = np.asarray(p_funded, dtype=np.float64)
    pay = np.asarray(payout, dtype=np.float64)
    nb_arr = np.asarray(nb, dtype=np.float64).copy()
    fam = np.asarray(family_ids, dtype=np.int64)
    mask = np.asarray(candidate_mask, dtype=bool)
    refi_ok = np.asarray(refinance_ok).astype(bool)
    pend = np.asarray(pending_family_penalized, dtype=bool) & mask

    nb_arr = nb_arr - cfg.pending_family_penalty * pend
    u = utility(pf, pay, nb_arr, cfg.alpha)

    harmful = mask & (nb_arr < -cfg.delta)
    safe = mask & (nb_arr >= 0.0)
    # same-family safe alternative present in the slate?
    same_fam = fam[:, :, None] == fam[:, None, :]  # (B, K, K)
    has_safe_alt = (same_fam & safe[:, None, :]).any(axis=2)  # (B, K)
    excluded_harm = harmful & has_safe_alt
    penalized_harm = harmful & ~has_safe_alt
    u = u - cfg.harm_penalty * penalized_harm
    excluded_refi = mask & ~refi_ok

    keep = mask & ~excluded_harm & ~excluded_refi
    u = np.where(keep, u, -np.inf)
    return GuardrailResult(
        utility=u,
        nb=nb_arr,
        keep=keep,
        excluded_harm=excluded_harm,
        penalized_harm=penalized_harm,
        excluded_refinance=excluded_refi,
        penalized_pending_family=pend,
    )
