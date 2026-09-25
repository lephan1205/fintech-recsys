"""Weighted probability calibration for compliance-grade scores (NumPy only).

Ranking models optimise ordering, not probability.  The valuation stage and any
approval-odds surface shown to a consumer need *calibrated* ``p(click)``,
``p(apply | click)`` and ``p(approve | apply)``: a model that says "80 % approval
odds" when the realised rate is 40 % is a fair-lending problem, not just a metric
problem.  Serving order is fixed: raw logits -> **calibrate** -> valuation -> PRM.
PRM outputs an ordering, so nothing is re-calibrated after it.

Two calibrators share one ``fit / predict`` interface and both accept per-row
**sample weights** (D4.3: ``p3`` is fitted on resolved applications with the same
``approve_weight`` used in training):

* :class:`IsotonicCalibrator` (default) - weighted pool-adjacent-violators, a monotone
  step function fitted on *probabilities*; with enough data its non-parametric fit
  beats any two-parameter sigmoid.
* :class:`PlattCalibrator` (fallback) - ``sigmoid(a * z + b)`` on *logits* by weighted
  Newton on the exact Hessian; used when the weighted positive count is below
  ``min_positives_isotonic`` (isotonic overfits into a jagged staircase on scarce
  positives).  Temperature scaling is omitted: for one binary logit it is Platt with
  ``b = 0``, so it can only match or lose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
CalibrationMethod = Literal["isotonic", "platt", "auto"]
InputKind = Literal["logit", "prob"]


def _sigmoid(z: FloatArray) -> FloatArray:
    """Overflow-safe logistic function."""
    out: FloatArray = np.where(
        z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z)))
    )
    return out


def _logit(p: FloatArray, eps: float = 1e-7) -> FloatArray:
    q = np.clip(p, eps, 1.0 - eps)
    out: FloatArray = np.log(q) - np.log1p(-q)
    return out


def _as_float(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _weights(w: npt.ArrayLike | None, n: int) -> FloatArray:
    out = np.ones(n) if w is None else _as_float(w)
    if out.shape[0] != n:
        raise ValueError("weights must have one entry per row")
    if (out < 0).any():
        raise ValueError("weights must be non-negative")
    return out


def weighted_nll(
    probs: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> float:
    """Weighted binary log-loss (proper scoring rule) with clipping."""
    p = np.clip(_as_float(probs), 1e-7, 1 - 1e-7)
    y = _as_float(labels)
    w = _weights(weights, len(y))
    if w.sum() <= 0:
        return float("nan")
    return float(-(w * (y * np.log(p) + (1 - y) * np.log1p(-p))).sum() / w.sum())


class Calibrator(Protocol):
    input_kind: InputKind

    def fit(
        self, scores: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike | None = None
    ) -> None: ...

    def predict(self, scores: npt.ArrayLike) -> FloatArray: ...

    def to_dict(self) -> dict[str, Any]: ...


class PlattCalibrator:
    """Fit ``sigmoid(a * z + b)`` on logits by weighted, damped Newton's method."""

    input_kind: InputKind = "logit"

    def __init__(self, max_iter: int = 100, l2: float = 1e-4) -> None:
        self.max_iter = max_iter
        self.l2 = l2
        self.a = 1.0
        self.b = 0.0
        self.iterations = 0

    def _objective(self, x: FloatArray, y: FloatArray, w: FloatArray, theta: FloatArray) -> float:
        z = x @ theta
        loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
        return float((w * loss).sum() / w.sum() + 0.5 * self.l2 * float(theta @ theta))

    def fit(
        self, scores: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike | None = None
    ) -> None:
        s, y = _as_float(scores), _as_float(labels)
        w = _weights(weights, len(y))
        theta = np.array([1.0 / max(float(np.std(s)), 1e-6), 0.0])
        x = np.stack([s, np.ones_like(s)], axis=1)  # (N, 2)
        total = w.sum()
        self.iterations = 0
        for _ in range(self.max_iter):
            self.iterations += 1
            p = _sigmoid(x @ theta)
            grad = x.T @ (w * (p - y)) / total + self.l2 * theta
            r = w * p * (1.0 - p)
            hess = (x * r[:, None]).T @ x / total + self.l2 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            f0, t = self._objective(x, y, w, theta), 1.0
            while t > 1e-6 and self._objective(x, y, w, theta - t * step) > f0:
                t *= 0.5
            theta = theta - t * step
            if np.abs(t * step).max() < 1e-8:
                break
        self.a, self.b = float(theta[0]), float(theta[1])

    def predict(self, scores: npt.ArrayLike) -> FloatArray:
        return _sigmoid(self.a * _as_float(scores) + self.b)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "platt", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlattCalibrator:
        cal = cls()
        cal.a, cal.b = float(d["a"]), float(d["b"])
        return cal


class IsotonicCalibrator:
    """Weighted pool-adjacent-violators on probabilities, linear interpolation between blocks.

    Minimizes ``sum_i w_i (f(x_i) - y_i)^2`` over non-decreasing ``f``; each block stores
    ``(sum w y, sum w, x_lo, x_hi)`` and adjacent violators are pooled by weighted mean.
    """

    input_kind: InputKind = "prob"

    def __init__(self) -> None:
        self.x_: FloatArray = np.zeros(0)
        self.y_: FloatArray = np.zeros(0)

    def fit(
        self, scores: npt.ArrayLike, labels: npt.ArrayLike, weights: npt.ArrayLike | None = None
    ) -> None:
        s, y = _as_float(scores), _as_float(labels)
        w = _weights(weights, len(y))
        keep = w > 0
        s, y, w = s[keep], y[keep], w[keep]
        if s.size == 0:
            raise ValueError("isotonic fit needs at least one positively weighted row")
        order = np.argsort(s, kind="stable")
        s, y, w = s[order], y[order], w[order]
        blocks: list[list[float]] = []  # [sum_wy, sum_w, x_lo, x_hi]
        for xi, yi, wi in zip(s, y, w, strict=True):
            blocks.append([wi * yi, wi, xi, xi])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                last = blocks.pop()
                blocks[-1][0] += last[0]
                blocks[-1][1] += last[1]
                blocks[-1][3] = last[3]
        xs: list[float] = []
        ys: list[float] = []
        for total, count, lo, hi in blocks:
            mean = total / count
            xs += [lo, hi]
            ys += [mean, mean]
        self.x_, self.y_ = np.asarray(xs), np.asarray(ys)

    def predict(self, scores: npt.ArrayLike) -> FloatArray:
        if self.x_.size == 0:
            raise RuntimeError("IsotonicCalibrator.fit must be called first")
        out: FloatArray = np.interp(_as_float(scores), self.x_, self.y_)
        return np.clip(out, 0.0, 1.0)

    @property
    def num_blocks(self) -> int:
        return int(self.x_.size // 2)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "isotonic", "x": self.x_.tolist(), "y": self.y_.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IsotonicCalibrator:
        cal = cls()
        cal.x_, cal.y_ = np.asarray(d["x"], dtype=np.float64), np.asarray(d["y"], dtype=np.float64)
        return cal


def calibrator_from_dict(d: dict[str, Any]) -> Calibrator:
    if d["kind"] == "platt":
        return PlattCalibrator.from_dict(d)
    if d["kind"] == "isotonic":
        return IsotonicCalibrator.from_dict(d)
    raise ValueError(f"unknown calibrator kind {d['kind']!r}")


@dataclass(frozen=True)
class CalibrationConfig:
    method: CalibrationMethod = "isotonic"
    min_positives_isotonic: float = 500.0  # weighted positive count
    auto_holdout_fraction: float = 0.2
    seed: int = 0


@dataclass
class FitResult:
    calibrator: Calibrator
    method_used: Literal["isotonic", "platt"]
    reason: str
    weighted_positives: float
    num_rows: int


def apply_calibrator(cal: Calibrator, logits: npt.ArrayLike) -> FloatArray:
    """Feed a calibrator its native input (logit or probability) from logits."""
    z = _as_float(logits)
    return cal.predict(z if cal.input_kind == "logit" else _sigmoid(z))


def fit_calibrator(
    method: CalibrationMethod,
    logits: npt.ArrayLike,
    labels: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    min_positives_isotonic: float = 500.0,
    auto_holdout_fraction: float = 0.2,
    seed: int = 0,
) -> FitResult:
    """Fit one tower's calibrator on ``logits`` with the D5 selection rules.

    ``"isotonic"`` falls back to Platt when the weighted positive count is below
    ``min_positives_isotonic``; ``"auto"`` picks the method with the lower held-out
    weighted NLL (seeded 80/20 split) and refits the winner on the full set.
    """
    z, y = _as_float(logits), _as_float(labels)
    w = _weights(weights, len(y))
    pos = float((w * (y > 0.5)).sum())
    n = int(len(y))

    def _fit(
        kind: Literal["isotonic", "platt"], zz: FloatArray, yy: FloatArray, ww: FloatArray
    ) -> Calibrator:
        cal: Calibrator = IsotonicCalibrator() if kind == "isotonic" else PlattCalibrator()
        cal.fit(zz if cal.input_kind == "logit" else _sigmoid(zz), yy, ww)
        return cal

    if method == "platt":
        return FitResult(_fit("platt", z, y, w), "platt", "requested", pos, n)
    if method == "isotonic":
        if pos < min_positives_isotonic:
            return FitResult(
                _fit("platt", z, y, w),
                "platt",
                f"fallback: weighted positives {pos:.1f} < {min_positives_isotonic}",
                pos,
                n,
            )
        return FitResult(_fit("isotonic", z, y, w), "isotonic", "requested", pos, n)
    if method != "auto":
        raise ValueError(f"unknown calibration method {method!r}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(1, int(round(auto_holdout_fraction * n)))
    hold, train = perm[:n_hold], perm[n_hold:]
    if train.size == 0 or hold.size == 0:
        return FitResult(_fit("platt", z, y, w), "platt", "auto: too few rows", pos, n)
    nll: dict[str, float] = {}
    for kind in ("isotonic", "platt"):
        cal = _fit(kind, z[train], y[train], w[train])
        nll[kind] = weighted_nll(apply_calibrator(cal, z[hold]), y[hold], w[hold])
    winner: Literal["isotonic", "platt"] = (
        "isotonic" if nll["isotonic"] <= nll["platt"] else "platt"
    )
    reason = f"auto: held-out NLL isotonic={nll['isotonic']:.4f} platt={nll['platt']:.4f}"
    return FitResult(_fit(winner, z, y, w), winner, reason, pos, n)


@dataclass
class CalibratorSet:
    """One calibrator per funnel tower, applied on raw logits in one call."""

    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    reports: dict[str, str] = field(default_factory=dict)

    def calibrate(
        self, z1: npt.ArrayLike, z2: npt.ArrayLike, z3: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Raw logits -> calibrated ``(p1, p2, p3)``; missing calibrators fall back to sigmoid."""
        outs = []
        for name, z in (("click", z1), ("apply", z2), ("approve", z3)):
            zz = _as_float(z)
            cal = self.calibrators.get(name)
            outs.append(_sigmoid(zz) if cal is None else apply_calibrator(cal, zz))
        return outs[0], outs[1], outs[2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibrators": {k: v.to_dict() for k, v in self.calibrators.items()},
            "reports": dict(self.reports),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibratorSet:
        return cls(
            calibrators={k: calibrator_from_dict(v) for k, v in d["calibrators"].items()},
            reports=dict(d.get("reports", {})),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibratorSet:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_funnel_calibrators(
    z1: npt.ArrayLike,
    z2: npt.ArrayLike,
    z3: npt.ArrayLike,
    y_click: npt.ArrayLike,
    y_apply: npt.ArrayLike,
    y_approve: npt.ArrayLike,
    approve_observed: npt.ArrayLike,
    approve_weight: npt.ArrayLike,
    candidate_mask: npt.ArrayLike,
    config: CalibrationConfig | None = None,
) -> CalibratorSet:
    """Fit ``p1`` on all rows, ``p2`` on clicked rows, ``p3`` on **resolved applications**
    with ``approve_weight`` as sample weights (never on unobserved rows).  Logits are
    the down-sampling-corrected serving logits."""
    cfg = config or CalibrationConfig()
    m = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    yc = _as_float(y_click)[m]
    ya = _as_float(y_apply)[m]
    yp = _as_float(y_approve)[m]
    obs = np.asarray(approve_observed, dtype=bool).reshape(-1)[m]
    w = _as_float(approve_weight)[m]
    zz1, zz2, zz3 = _as_float(z1)[m], _as_float(z2)[m], _as_float(z3)[m]
    clicked = yc > 0.5
    applied_resolved = (ya > 0.5) & obs
    out = CalibratorSet()
    rows: dict[str, tuple[FloatArray, FloatArray, FloatArray | None]] = {
        "click": (zz1, yc, None),
        "apply": (zz2[clicked], ya[clicked], None),
        "approve": (zz3[applied_resolved], yp[applied_resolved], w[applied_resolved]),
    }
    for name, (z, y, wt) in rows.items():
        if y.size < 2 or y.min() == y.max():
            out.reports[name] = "skipped: too few rows or constant labels"
            continue
        res = fit_calibrator(
            cfg.method, z, y, wt, cfg.min_positives_isotonic, cfg.auto_holdout_fraction, cfg.seed
        )
        out.calibrators[name] = res.calibrator
        out.reports[name] = (
            f"{res.method_used} ({res.reason}; n={res.num_rows}, w+={res.weighted_positives:.1f})"
        )
    return out
