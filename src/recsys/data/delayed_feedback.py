"""Delayed feedback: observed status, delay CDF and approval weights (pure NumPy).

Design (D1): the slate stores the *oracle* approval and the partner's decision
delay; at any training cut-off ``snapshot_at_days`` the observed status is a pure
function of those.  Pending rows stay in the batch and are masked out of the two
approval terms only, via ``approve_weight``:

* ``"drop"``      weight 0 on PENDING rows, 1 everywhere else (default).
* ``"ipw"``       same masks, but resolved applications get the Horvitz-Thompson
                  weight ``1 / max(F_{y,fam}(elapsed), w_floor)`` where ``F`` is the
                  delay CDF given the outcome (uses the generator's ``DelayConfig``).
* ``"negative"``  pending trained as ``y_approve = 0`` with weight 1 — wrong by
                  construction; kept only as the baseline.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from recsys.data.schema import ApplicationStatus, DelayConfig

PendingPolicy = Literal["drop", "ipw", "negative"]

_erf = np.vectorize(math.erf, otypes=[np.float64])


def observed_status(
    y_apply: npt.ArrayLike,
    served_at_days: npt.ArrayLike,
    decision_delay_days: npt.ArrayLike,
    snapshot_at_days: float,
    y_approve: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """Vectorized twin of :func:`recsys.data.schema.observed_status`.

    All array arguments broadcast together; returns ``ApplicationStatus`` values.
    """
    ya = np.asarray(y_apply, dtype=np.int64)
    yp = np.asarray(y_approve, dtype=np.int64)
    served = np.asarray(served_at_days, dtype=np.float64)
    delay = np.asarray(decision_delay_days, dtype=np.float64)
    pending = served + delay > snapshot_at_days
    resolved = np.where(yp == 1, ApplicationStatus.APPROVED, ApplicationStatus.DECLINED)
    applied = np.where(pending, ApplicationStatus.PENDING, resolved)
    out: npt.NDArray[np.int64] = np.where(ya == 1, applied, ApplicationStatus.NOT_APPLIED)
    return out.astype(np.int64)


def _normal_cdf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))
    return out


def delay_cdf(
    elapsed_days: npt.ArrayLike,
    family: npt.ArrayLike,
    outcome: npt.ArrayLike,
    config: DelayConfig,
) -> npt.NDArray[np.float64]:
    """``F_{outcome,family}(elapsed) = P(delay <= elapsed)`` under the mixture law.

    ``outcome`` is 0 for approved and 1 for declined.  Mass ``p_instant`` sits at
    ``instant_delay_days``; the remainder is log-normal.
    """
    t = np.asarray(elapsed_days, dtype=np.float64)
    fam = np.asarray(family, dtype=np.int64)
    out = np.asarray(outcome, dtype=np.int64)
    p_inst = np.asarray(config.p_instant, dtype=np.float64)[fam, out]
    mu = np.asarray(config.log_mu, dtype=np.float64)[fam, out]
    sigma = np.asarray(config.log_sigma, dtype=np.float64)[fam, out]
    positive = t > 0.0
    z = (np.log(np.where(positive, t, 1.0)) - mu) / sigma
    tail = np.where(positive, _normal_cdf(z), 0.0)
    instant = np.where(t >= config.instant_delay_days, 1.0, 0.0)
    cdf: npt.NDArray[np.float64] = p_inst * instant + (1.0 - p_inst) * tail
    return np.clip(cdf, 0.0, 1.0)


def sample_decision_delay(
    rng: np.random.Generator,
    family: npt.ArrayLike,
    outcome: npt.ArrayLike,
    config: DelayConfig,
) -> npt.NDArray[np.float64]:
    """Draw one delay per element of ``family`` / ``outcome`` (broadcast)."""
    fam = np.asarray(family, dtype=np.int64)
    out = np.asarray(outcome, dtype=np.int64)
    fam, out = np.broadcast_arrays(fam, out)
    p_inst = np.asarray(config.p_instant, dtype=np.float64)[fam, out]
    mu = np.asarray(config.log_mu, dtype=np.float64)[fam, out]
    sigma = np.asarray(config.log_sigma, dtype=np.float64)[fam, out]
    is_instant = rng.random(fam.shape) < p_inst
    lognormal = np.exp(mu + sigma * rng.normal(size=fam.shape))
    delay: npt.NDArray[np.float64] = np.where(is_instant, config.instant_delay_days, lognormal)
    return np.round(delay, 4)


def approve_observed(status: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """True where the approval label is usable: NOT_APPLIED, APPROVED or DECLINED."""
    s = np.asarray(status, dtype=np.int64)
    out: npt.NDArray[np.bool_] = s != int(ApplicationStatus.PENDING)
    return out


def approve_weights(
    status: npt.ArrayLike,
    elapsed_days: npt.ArrayLike,
    family: npt.ArrayLike,
    y_approve_observed: npt.ArrayLike,
    config: DelayConfig,
    policy: PendingPolicy = "drop",
    w_floor: float = 0.05,
) -> npt.NDArray[np.float64]:
    """Per-row weight for the approval terms (``L_approve`` and ``L_ctcavr``).

    ``y_approve_observed`` is the *observed* label (0 on pending rows); it selects
    the outcome-conditional delay CDF for ``"ipw"``.
    """
    if policy not in ("drop", "ipw", "negative"):
        raise ValueError(f"unknown pending_policy {policy!r}")
    if not 0.0 < w_floor <= 1.0:
        raise ValueError("w_floor must be in (0, 1]")
    s = np.asarray(status, dtype=np.int64)
    w = np.ones(s.shape, dtype=np.float64)
    pending = s == int(ApplicationStatus.PENDING)
    if policy == "negative":
        return w
    w[pending] = 0.0
    if policy == "ipw":
        resolved = (s == int(ApplicationStatus.APPROVED)) | (s == int(ApplicationStatus.DECLINED))
        yp = np.asarray(y_approve_observed, dtype=np.int64)
        outcome = np.where(yp == 1, 0, 1)
        cdf = delay_cdf(elapsed_days, family, outcome, config)
        ipw = 1.0 / np.maximum(cdf, w_floor)
        w = np.where(resolved, ipw, w)
    return w
