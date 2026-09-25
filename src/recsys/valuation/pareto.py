"""Offline Pareto sweep over the platform-vs-user weight ``α`` (D5).

For each ``α`` the slate is re-valued with :func:`utility`, guardrails are applied, the
top-``k`` by utility is served (PRM is not involved: the sweep isolates the valuation
policy), and ``ExpectedRevenue@k``, ``ExpectedUserBenefit@k`` and ``HarmRate@k`` are
reported, overall and by credit tier.  Revenue is weakly increasing and user benefit
weakly decreasing in ``α`` (tested).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from recsys.data.schema import TIER_ORDER
from recsys.metrics.slate_metrics import (
    expected_revenue_at_k,
    expected_user_benefit_at_k,
    harm_rate_at_k,
    sliced_mean,
)
from recsys.valuation.utility import UtilityConfig, apply_guardrails

FloatArray = npt.NDArray[np.float64]
DEFAULT_ALPHAS: tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(11))


@dataclass
class ParetoRow:
    alpha: float
    revenue_at_k: float
    user_benefit_at_k: float
    harm_rate_at_k: float
    by_tier_revenue: dict[str, float]
    by_tier_user_benefit: dict[str, float]
    by_tier_harm_rate: dict[str, float]


def top_k_by_utility(utility: npt.ArrayLike, k: int) -> npt.NDArray[np.int64]:
    """``(B, K)`` utilities (−inf excluded) -> ``(B, k)`` slot indices, best first."""
    u = np.asarray(utility, dtype=np.float64)
    order = np.argsort(-u, axis=1, kind="stable")
    out: npt.NDArray[np.int64] = order[:, :k].astype(np.int64)
    return out


def pareto_sweep(
    p_funded: npt.ArrayLike,
    payout: npt.ArrayLike,
    nb: npt.ArrayLike,
    family_ids: npt.ArrayLike,
    candidate_mask: npt.ArrayLike,
    refinance_ok: npt.ArrayLike,
    pending_family_penalized: npt.ArrayLike,
    user_tiers: npt.ArrayLike,
    k: int = 10,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    config: UtilityConfig | None = None,
) -> list[ParetoRow]:
    base = config or UtilityConfig()
    tiers = np.asarray(user_tiers, dtype=np.int64)
    names = {i: t.value for i, t in enumerate(TIER_ORDER)}
    rows: list[ParetoRow] = []
    for alpha in alphas:
        g = apply_guardrails(
            p_funded, payout, nb, family_ids, candidate_mask, refinance_ok,
            pending_family_penalized, replace(base, alpha=float(alpha)),
        )  # fmt: skip
        order = top_k_by_utility(g.utility, k)
        rev = expected_revenue_at_k(p_funded, payout, order, k, g.keep)
        ben = expected_user_benefit_at_k(p_funded, g.nb, order, k, g.keep)
        harm = harm_rate_at_k(g.nb, order, k, g.keep)
        rows.append(
            ParetoRow(
                alpha=float(alpha),
                revenue_at_k=float(rev.mean()),
                user_benefit_at_k=float(ben.mean()),
                harm_rate_at_k=float(np.nanmean(harm)),
                by_tier_revenue=sliced_mean(rev, tiers, names),
                by_tier_user_benefit=sliced_mean(ben, tiers, names),
                by_tier_harm_rate=sliced_mean(harm, tiers, names),
            )
        )
    return rows


def pareto_table_markdown(rows: Sequence[ParetoRow], k: int = 10) -> str:
    lines = [
        f"| α | ExpectedRevenue@{k} ($) | ExpectedUserBenefit@{k} ($) | HarmRate@{k} |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.alpha:.1f} | {r.revenue_at_k:.2f} | {r.user_benefit_at_k:.2f} "
            f"| {r.harm_rate_at_k:.3f} |"
        )
    tiers = list(rows[0].by_tier_revenue) if rows else []
    if tiers:
        lines.append("")
        lines.append(f"| α | tier | Revenue@{k} | UserBenefit@{k} | HarmRate@{k} |")
        lines.append("|---|---|---:|---:|---:|")
        for r in rows:
            for t in tiers:
                lines.append(
                    f"| {r.alpha:.1f} | {t} | {r.by_tier_revenue[t]:.2f} | "
                    f"{r.by_tier_user_benefit[t]:.2f} | {r.by_tier_harm_rate[t]:.3f} |"
                )
    return "\n".join(lines) + "\n"
