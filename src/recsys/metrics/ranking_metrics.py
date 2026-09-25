"""Ranking metrics (NumPy): AUC, PR-AUC, GAUC, NCE, NDCG@k, Recall@k (D7), hit rate, Gini.

Every probability-quality metric accepts optional per-row ``weights`` so approval
metrics can be computed on resolved rows with the same ``approve_weight`` used in
training (D4).  ``recall_at_k`` is the retrieval metric of D7: candidate ids against a
padded positives array, ineligible positives excluded from the denominator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _rankdata_average(x: FloatArray) -> FloatArray:
    """1-based ranks with ties averaged (no scipy dependency)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _prep(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, weights: npt.ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != y.shape or s.shape != y.shape:
        raise ValueError("y_true, y_score and weights must have the same length")
    return y, s, w


def auc(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> float:
    """(Weighted) ROC AUC via the Mann-Whitney statistic; ``nan`` if one class is absent.

    Ties count half.  With unit weights this equals the classic rank formula.
    """
    y, s, w = _prep(y_true, y_score, weights)
    pos, neg = y > 0.5, y <= 0.5
    w_pos, w_neg = float(w[pos].sum()), float(w[neg].sum())
    if w_pos <= 0 or w_neg <= 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
    neg_w = np.where(y_sorted <= 0.5, w_sorted, 0.0)
    pos_w = np.where(y_sorted > 0.5, w_sorted, 0.0)
    # group by unique score
    _, first = np.unique(s_sorted, return_index=True)
    bounds = np.append(first, len(s_sorted))
    total = 0.0
    cum_neg = 0.0
    for a, b_ in zip(bounds[:-1], bounds[1:], strict=True):
        neg_here = float(neg_w[a:b_].sum())
        pos_here = float(pos_w[a:b_].sum())
        total += pos_here * (cum_neg + 0.5 * neg_here)
        cum_neg += neg_here
    return float(total / (w_pos * w_neg))


def pr_auc(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> float:
    """(Weighted) average precision; ``nan`` if there is no positive."""
    y, s, w = _prep(y_true, y_score, weights)
    if float(w[y > 0.5].sum()) <= 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y_s, w_s = y[order], w[order]
    tp = np.cumsum(w_s * (y_s > 0.5))
    fp = np.cumsum(w_s * (y_s <= 0.5))
    precision = tp / np.maximum(tp + fp, 1e-12)
    gain = w_s * (y_s > 0.5)
    return float((precision * gain).sum() / gain.sum())


def normalized_cross_entropy(
    y_true: npt.ArrayLike,
    probs: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    eps: float = 1e-7,
) -> float:
    """Weighted log-loss divided by the log-loss of the (weighted) prior; < 1 beats the prior."""
    y, p, w = _prep(y_true, probs, weights)
    if w.sum() <= 0:
        return float("nan")
    p = np.clip(p, eps, 1.0 - eps)
    ll = -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum()
    prior = float(np.clip((w * y).sum() / w.sum(), eps, 1.0 - eps))
    ll_prior = -(prior * np.log(prior) + (1 - prior) * np.log(1 - prior))
    return float(ll / ll_prior) if ll_prior > 0 else float("nan")


def gauc(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    group_ids: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> float:
    """Group AUC: impression-weighted mean of per-user AUC over users with both classes."""
    y, s, w = _prep(y_true, y_score, weights)
    g = np.asarray(group_ids).reshape(-1)
    total_w, acc = 0.0, 0.0
    for gid in np.unique(g):
        idx = g == gid
        a = auc(y[idx], s[idx], w[idx])
        if np.isnan(a):
            continue
        wt = float(w[idx].sum())
        total_w += wt
        acc += wt * a
    return float(acc / total_w) if total_w > 0 else float("nan")


def _dcg(rel: FloatArray) -> FloatArray:
    discounts = 1.0 / np.log2(np.arange(2, rel.shape[-1] + 2))
    out: FloatArray = (rel * discounts).sum(axis=-1)
    return out


def ndcg_at_k(scores: npt.ArrayLike, relevance: npt.ArrayLike, k: int) -> float:
    """Mean NDCG@k over rows ``(N, M)``; rows with zero ideal DCG are skipped."""
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(relevance, dtype=np.float64)
    if s.ndim == 1:
        s, r = s[None, :], r[None, :]
    k = min(k, s.shape[1])
    order = np.argsort(-s, axis=1, kind="stable")[:, :k]
    ranked_rel = np.take_along_axis(r, order, axis=1)
    ideal = -np.sort(-r, axis=1)[:, :k]
    idcg = _dcg(ideal)
    valid = idcg > 0
    if not valid.any():
        return float("nan")
    return float((_dcg(ranked_rel)[valid] / idcg[valid]).mean())


@dataclass
class RecallResult:
    per_user: FloatArray  # (B,) recall, nan for users with no eligible positive
    mean: float
    num_users: int  # users counted (>= 1 eligible positive)
    num_excluded_positives: int  # positives dropped because ineligible


def recall_at_k(
    candidate_ids: npt.ArrayLike,
    positives: npt.ArrayLike,
    positives_valid: npt.ArrayLike,
    k: int,
) -> RecallResult:
    """D7 retrieval recall.

    ``candidate_ids (B, K)`` beam output (``-1`` for dead beams, shorter beams handled),
    ``positives (B, P)`` padded positive item ids with ``positives_valid (B, P)`` marking
    real *and eligible* positives; ``positives_invalid`` (real but ineligible) are
    counted in ``num_excluded_positives`` when passed as ``positives_valid == False`` with
    a non-zero id.  Per user: ``|top-k ∩ P_u ∩ E_u| / |P_u ∩ E_u|``.
    """
    cand = np.asarray(candidate_ids, dtype=np.int64)
    pos = np.asarray(positives, dtype=np.int64)
    valid = np.asarray(positives_valid, dtype=bool)
    if cand.ndim != 2 or pos.shape != valid.shape or pos.shape[0] != cand.shape[0]:
        raise ValueError("candidate_ids (B, K), positives / positives_valid (B, P) expected")
    top = cand[:, :k]  # (B, k')
    hit = (pos[:, :, None] == top[:, None, :]).any(axis=2) & valid & (pos > 0)  # (B, P)
    n_valid = (valid & (pos > 0)).sum(axis=1)
    per_user = np.full(cand.shape[0], np.nan, dtype=np.float64)
    counted = n_valid > 0
    per_user[counted] = hit.sum(axis=1)[counted] / n_valid[counted]
    excluded = int(((pos > 0) & ~valid).sum())
    mean = float(np.nanmean(per_user)) if counted.any() else float("nan")
    return RecallResult(per_user, mean, int(counted.sum()), excluded)


def recall_at_k_sets(ranked_ids: npt.ArrayLike, true_items: Sequence[set[int]], k: int) -> float:
    """Mean ``|top-k ∩ true| / |true|`` over rows with at least one true item."""
    ranked = np.asarray(ranked_ids)
    vals = []
    for row, truth in zip(ranked, true_items, strict=True):
        if not truth:
            continue
        top = {int(i) for i in row[:k]}
        vals.append(len(top & truth) / len(truth))
    return float(np.mean(vals)) if vals else float("nan")


def hit_rate_at_k(ranked_ids: npt.ArrayLike, true_item: npt.ArrayLike, k: int) -> float:
    """Fraction of rows whose single true item appears in the top-k."""
    ranked = np.asarray(ranked_ids)
    t = np.asarray(true_item).reshape(-1)
    hits = (ranked[:, :k] == t[:, None]).any(axis=1)
    return float(hits.mean())


def gini(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    order = np.lexsort((np.arange(len(p)), -p))  # sort by prediction desc, stable
    y_sorted = y[order]
    cum = np.cumsum(y_sorted) / y_sorted.sum()
    return float(cum.sum() / len(y) - (len(y) + 1) / (2.0 * len(y)))


def normalized_gini(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Gini of the prediction ordering divided by the Gini of the perfect ordering."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y.sum() == 0:
        return float("nan")
    return gini(y, y_pred) / gini(y, y)
