"""Calibration metrics (NumPy): reliability bins, ECE, MCE, Brier — all weight-aware.

Weights let the approval tower be scored on resolved applications with the same
``approve_weight`` used in training and calibration (D4).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BinStrategy = Literal["uniform", "quantile"]


def _prep(
    probs: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if p.shape != y.shape or w.shape != p.shape:
        raise ValueError("probs, labels and weights must have the same length")
    return p, y, w


def reliability_bins(
    probs: npt.ArrayLike,
    labels: npt.ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    weights: npt.ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-bin ``(weighted mean confidence, weighted accuracy, total weight)``."""
    p, y, w = _prep(probs, labels, weights)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    count = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        wb = float(w[m].sum())
        count[b] = wb
        if wb > 0:
            conf[b] = float((w[m] * p[m]).sum() / wb)
            acc[b] = float((w[m] * y[m]).sum() / wb)
    return conf, acc, count


def expected_calibration_error(
    probs: npt.ArrayLike,
    labels: npt.ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    weights: npt.ArrayLike | None = None,
) -> float:
    """``sum_b (w_b / W) * |acc_b - conf_b|``."""
    conf, acc, count = reliability_bins(probs, labels, n_bins, strategy, weights)
    n = count.sum()
    if n == 0:
        return float("nan")
    return float((count / n * np.abs(acc - conf)).sum())


def max_calibration_error(
    probs: npt.ArrayLike,
    labels: npt.ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    weights: npt.ArrayLike | None = None,
) -> float:
    conf, acc, count = reliability_bins(probs, labels, n_bins, strategy, weights)
    if not (count > 0).any():
        return float("nan")
    return float(np.abs(acc - conf)[count > 0].max())


def brier_score(
    probs: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> float:
    p, y, w = _prep(probs, labels, weights)
    if w.sum() <= 0:
        return float("nan")
    return float((w * (p - y) ** 2).sum() / w.sum())
