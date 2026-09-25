"""Delayed feedback (D1): observed status, delay law, approval weights, pending-policy bias."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from recsys import seed_everything
from recsys.data.delayed_feedback import (
    PendingPolicy,
    approve_observed,
    approve_weights,
    delay_cdf,
    observed_status,
    sample_decision_delay,
)
from recsys.data.schema import ApplicationStatus, DelayConfig
from recsys.data.schema import observed_status as observed_status_scalar
from recsys.data.synthetic_generator import GeneratorConfig, SyntheticFintechDataGenerator

MORTGAGE_YOUNG = GeneratorConfig(
    num_users=400,
    num_products=150,
    seed=11,
    slates_per_user=3,
    slate_size=20,
    ineligible_per_slate=2,
    snapshot_at_days=40.0,
    recency_window_days=10.0,
    served_window_days=30.0,
    family_mix=(0.0, 0.0, 0.0, 0.0, 1.0),
    min_history=4,
    max_history=12,
    mean_gap_days=1.0,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)


def _random_rows(n: int = 2000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    y_apply = (rng.random(n) < 0.3).astype(np.int64)
    y_approve = y_apply * (rng.random(n) < 0.5).astype(np.int64)
    served = rng.uniform(0.0, 100.0, size=n)
    delay = np.where(y_apply == 1, rng.lognormal(2.0, 1.0, size=n), 0.0)
    family = rng.integers(0, 5, size=n)
    return {
        "y_apply": y_apply,
        "y_approve": y_approve,
        "served": served,
        "delay": delay,
        "family": family,
    }


# ------------------------------------------------------------------ observed status


def test_observed_status_invariants_vs_oracle() -> None:
    r = _random_rows()
    snap = 100.0
    st = observed_status(r["y_apply"], r["served"], r["delay"], snap, r["y_approve"])
    scalar = np.array(
        [
            int(observed_status_scalar(int(a), int(p), float(s), float(d), snap))
            for a, p, s, d in zip(
                r["y_apply"], r["y_approve"], r["served"], r["delay"], strict=True
            )
        ]
    )
    assert np.array_equal(st, scalar)
    not_applied = r["y_apply"] == 0
    assert np.all(st[not_applied] == int(ApplicationStatus.NOT_APPLIED))
    pending = (r["y_apply"] == 1) & (r["served"] + r["delay"] > snap)
    assert np.all(st[pending] == int(ApplicationStatus.PENDING))
    resolved = (r["y_apply"] == 1) & ~pending
    expect = np.where(r["y_approve"] == 1, ApplicationStatus.APPROVED, ApplicationStatus.DECLINED)
    assert np.array_equal(st[resolved], expect[resolved])
    assert pending.sum() > 0 and resolved.sum() > 0
    assert np.array_equal(approve_observed(st), st != int(ApplicationStatus.PENDING))


def test_snapshot_at_infinity_resolves_everything_to_the_oracle() -> None:
    r = _random_rows()
    st = observed_status(r["y_apply"], r["served"], r["delay"], np.inf, r["y_approve"])
    assert not np.any(st == int(ApplicationStatus.PENDING))
    applied = r["y_apply"] == 1
    assert np.array_equal(
        st[applied] == int(ApplicationStatus.APPROVED), r["y_approve"][applied] == 1
    )
    # and a very young snapshot leaves every application pending
    st0 = observed_status(r["y_apply"], r["served"], r["delay"], -1.0, r["y_approve"])
    assert np.all(st0[applied] == int(ApplicationStatus.PENDING))


# ------------------------------------------------------------------ delay law


def test_delay_cdf_shape_and_limits() -> None:
    cfg = DelayConfig()
    t = np.array([0.0, 0.005, 0.01, 0.5, 2.0, 10.0, 60.0, 400.0])
    for fam in range(5):
        for outcome in (0, 1):
            f = delay_cdf(t, np.full_like(t, fam, dtype=np.int64), outcome, cfg)
            assert f[0] == 0.0
            assert np.all(np.diff(f) >= -1e-12)
            assert f[-1] > 0.999
            assert np.all((f >= 0.0) & (f <= 1.0))
    cards = delay_cdf(t, np.zeros_like(t, dtype=np.int64), 0, cfg)
    assert cards[1] == 0.0 and cards[2] >= 0.8  # instant mass appears at instant_delay_days
    mortgage = delay_cdf(np.array([10.0]), np.array([4]), np.array([0, 1]), cfg)
    assert mortgage[0] < mortgage[1]  # declines resolve faster than approvals


def test_sample_decision_delay_matches_the_law() -> None:
    rng = np.random.default_rng(0)
    cfg = DelayConfig()
    n = 20_000
    cards = sample_decision_delay(rng, np.zeros(n, dtype=np.int64), 0, cfg)
    assert np.all(cards > 0.0)
    assert abs((cards == cfg.instant_delay_days).mean() - 0.8) < 0.02
    assert np.percentile(cards[cards > cfg.instant_delay_days], 90) < 2.0
    mort_ok = sample_decision_delay(rng, np.full(n, 4), 0, cfg)
    mort_no = sample_decision_delay(rng, np.full(n, 4), 1, cfg)
    assert abs(np.median(mort_ok) - 35.0) < 2.0
    assert abs(np.median(mort_no) - 20.0) < 1.5


# ------------------------------------------------------------------ approve weights


@pytest.mark.parametrize("policy", ["drop", "ipw"])
def test_pending_rows_have_zero_weight_and_non_applied_rows_weight_one(policy: str) -> None:
    r = _random_rows()
    snap = 100.0
    st = observed_status(r["y_apply"], r["served"], r["delay"], snap, r["y_approve"])
    y_obs = np.where(st == int(ApplicationStatus.PENDING), 0, r["y_approve"])
    w = approve_weights(
        st,
        snap - r["served"],
        r["family"],
        y_obs,
        DelayConfig(),
        cast(PendingPolicy, policy),
        w_floor=0.05,
    )
    assert np.all(w[st == int(ApplicationStatus.PENDING)] == 0.0)
    assert np.all(w[st == int(ApplicationStatus.NOT_APPLIED)] == 1.0)
    resolved = (st == int(ApplicationStatus.APPROVED)) | (st == int(ApplicationStatus.DECLINED))
    if policy == "drop":
        assert np.all(w[resolved] == 1.0)
    else:
        assert np.all(w[resolved] >= 1.0) and np.all(w[resolved] <= 20.0 + 1e-9)
        assert np.any(w[resolved] > 1.0)


def test_negative_policy_weights_everything_and_bad_inputs_raise() -> None:
    r = _random_rows()
    st = observed_status(r["y_apply"], r["served"], r["delay"], 100.0, r["y_approve"])
    w = approve_weights(
        st, 100.0 - r["served"], r["family"], r["y_approve"], DelayConfig(), "negative"
    )
    assert np.all(w == 1.0)
    with pytest.raises(ValueError):
        approve_weights(st, 1.0, 0, 0, DelayConfig(), "sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        approve_weights(st, 1.0, 0, 0, DelayConfig(), "ipw", w_floor=0.0)


# ------------------------------------------------------------------ policy bias


def test_pending_policy_bias_on_mortgage_slice_at_young_cutoff() -> None:
    """At a fresh cut-off on a mortgage-heavy slice: 'negative' grossly under-estimates the
    approval rate, 'drop' is closer (declines resolve faster than approvals), 'ipw' closest."""
    cfg = MORTGAGE_YOUNG
    ds = SyntheticFintechDataGenerator(cfg).generate()
    fam = ds.family_by_item()
    y_apply = np.array([v for s in ds.slates for v in s.y_apply])
    y_approve = np.array([v for s in ds.slates for v in s.y_approve])
    served = np.array([s.served_at_days for s in ds.slates for _ in s.y_apply])
    delay = np.array([v for s in ds.slates for v in s.decision_delay_days])
    family = np.array([fam[i] for s in ds.slates for i in s.candidate_item_ids])
    st = observed_status(y_apply, served, delay, cfg.snapshot_at_days, y_approve)
    applied = y_apply == 1
    pending = st == int(ApplicationStatus.PENDING)
    resolved = applied & ~pending
    assert applied.sum() > 200 and 0.5 < pending[applied].mean() < 0.95
    y_obs = np.where(pending, 0, y_approve)

    oracle = y_approve[applied].mean()
    negative = y_obs[applied].mean()  # pending trained as declined
    drop = y_obs[resolved].mean()
    w = approve_weights(st, cfg.snapshot_at_days - served, family, y_obs, cfg.delay, "ipw")
    ipw = float((w[resolved] * y_obs[resolved]).sum() / w[resolved].sum())

    assert negative < drop < ipw < oracle + 0.02
    assert abs(negative - oracle) > abs(drop - oracle) > abs(ipw - oracle)
    assert abs(negative - oracle) > 0.3  # the silent failure this design guards against
    assert abs(ipw - oracle) < 0.2
