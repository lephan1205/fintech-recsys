"""Slate-level valuation metrics (D5): ExpectedRevenue@K, ExpectedUserBenefit@K, HarmRate@K.

All take an ``order (B, K)`` of slot indices (display order, as :func:`rerank` returns)
and evaluate the first ``k`` served slots.  Slices by credit tier / family are computed
with :func:`sliced_mean`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _take(values: npt.ArrayLike, order: npt.ArrayLike, k: int) -> FloatArray:
    v = np.asarray(values, dtype=np.float64)
    o = np.asarray(order, dtype=np.int64)[:, :k]
    out: FloatArray = np.take_along_axis(v, o, axis=1)
    return out


def _served(order: npt.ArrayLike, mask: npt.ArrayLike | None, k: int) -> npt.NDArray[np.bool_]:
    o = np.asarray(order, dtype=np.int64)[:, :k]
    if mask is None:
        return np.ones_like(o, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    out: npt.NDArray[np.bool_] = np.take_along_axis(m, o, axis=1)
    return out


def expected_revenue_at_k(
    p_funded: npt.ArrayLike,
    payout: npt.ArrayLike,
    order: npt.ArrayLike,
    k: int,
    mask: npt.ArrayLike | None = None,
) -> FloatArray:
    """Per-slate ``Σ_{served} P_funded · payout`` over the first ``k`` slots -> ``(B,)``."""
    v = _take(np.asarray(p_funded) * np.asarray(payout), order, k) * _served(order, mask, k)
    out: FloatArray = v.sum(axis=1)
    return out


def expected_user_benefit_at_k(
    p_funded: npt.ArrayLike,
    nb: npt.ArrayLike,
    order: npt.ArrayLike,
    k: int,
    mask: npt.ArrayLike | None = None,
) -> FloatArray:
    """Per-slate ``Σ_{served} P_funded · NB`` over the first ``k`` slots -> ``(B,)``."""
    v = _take(np.asarray(p_funded) * np.asarray(nb), order, k) * _served(order, mask, k)
    out: FloatArray = v.sum(axis=1)
    return out


def harm_rate_at_k(
    nb: npt.ArrayLike, order: npt.ArrayLike, k: int, mask: npt.ArrayLike | None = None
) -> FloatArray:
    """Per-slate share of served items with ``NB < 0`` -> ``(B,)`` (nan if nothing served)."""
    served = _served(order, mask, k)
    harm = (_take(nb, order, k) < 0.0) & served
    n = served.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out: FloatArray = np.where(n > 0, harm.sum(axis=1) / np.maximum(n, 1), np.nan)
    return out


def sliced_mean(
    values: npt.ArrayLike, groups: npt.ArrayLike, names: Mapping[int, str]
) -> dict[str, float]:
    """``nanmean`` of ``values`` per group id, keyed by ``names[group]``."""
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups)
    out: dict[str, float] = {}
    for gid, name in names.items():
        sel = g == gid
        out[name] = (
            float(np.nanmean(v[sel])) if sel.any() and np.isfinite(v[sel]).any() else float("nan")
        )
    return out
