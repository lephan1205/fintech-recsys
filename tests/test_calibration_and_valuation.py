"""Stage 4: weighted calibration, calibration metrics, valuation (EV / NB / utility),
guardrails, Pareto sweep, slate metrics and PRM building blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recsys import seed_everything
from recsys.data.schema import FinancialProduct, ProductFamily, UserProfile
from recsys.losses.listwise_loss import (
    prm_listwise_loss,
    prm_target_distribution,
    same_family_penalty,
)
from recsys.metrics.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    max_calibration_error,
    reliability_bins,
)
from recsys.metrics.ranking_metrics import auc, gauc, normalized_cross_entropy, pr_auc
from recsys.metrics.slate_metrics import (
    expected_revenue_at_k,
    expected_user_benefit_at_k,
    harm_rate_at_k,
)
from recsys.models.prm.model import PRM, PRMConfig, build_prm_features, rerank
from recsys.serving.calibration import (
    CalibrationConfig,
    CalibratorSet,
    IsotonicCalibrator,
    PlattCalibrator,
    apply_calibrator,
    fit_calibrator,
    fit_funnel_calibrators,
    weighted_nll,
)
from recsys.valuation.expected_value import expected_revenue, p_funded
from recsys.valuation.pareto import pareto_sweep, pareto_table_markdown
from recsys.valuation.user_benefit import (
    ProductEconomics,
    UserBenefitConfig,
    UserFinancialState,
    monthly_payment,
    net_user_benefit,
    present_value_of_annuity,
    remaining_balance,
)
from recsys.valuation.utility import UtilityConfig, apply_guardrails, utility


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


def a(x: float) -> np.ndarray:
    """One-element float64 array for the amortization helpers."""
    return np.array([float(x)])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-z))
    return out


# ------------------------------------------------------------ calibration metrics


def test_ece_mce_brier_known_cases_and_weights() -> None:
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    conf, acc, count = reliability_bins(p, y, n_bins=10)
    assert count[1] == 2 and count[9] == 2 and abs(conf[1] - 0.1) < 1e-12
    assert expected_calibration_error(p, y) == pytest.approx(0.1)
    assert max_calibration_error(p, y) == pytest.approx(0.1)
    assert brier_score(p, y) == pytest.approx(0.01)
    over = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    y2 = np.array([1, 1, 0, 0, 1])
    assert expected_calibration_error(over, y2) == pytest.approx(0.3)
    # weights: duplicating a row == weight 2
    p3 = np.array([0.2, 0.8, 0.8])
    y3 = np.array([0, 1, 0])
    dup = expected_calibration_error(np.array([0.2, 0.8, 0.8, 0.8]), np.array([0, 1, 0, 0]))
    assert expected_calibration_error(p3, y3, weights=[1, 1, 2]) == pytest.approx(dup)
    assert brier_score(p3, y3, weights=[1, 1, 2]) == pytest.approx(
        brier_score([0.2, 0.8, 0.8, 0.8], [0, 1, 0, 0])
    )
    assert np.isnan(expected_calibration_error(p3, y3, weights=[0, 0, 0]))


def test_weighted_ranking_metrics_match_duplication() -> None:
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.7, 0.2, 0.1])
    w = np.array([1, 2, 1, 1, 3, 1])
    y_dup = np.repeat(y, w)
    s_dup = np.repeat(s, w)
    assert auc(y, s, w) == pytest.approx(auc(y_dup, s_dup))
    assert pr_auc(y, s, w) == pytest.approx(pr_auc(y_dup, s_dup))
    assert normalized_cross_entropy(y, s, w) == pytest.approx(
        normalized_cross_entropy(y_dup, s_dup)
    )
    assert np.isnan(auc([1, 1], [0.2, 0.3]))
    assert gauc([1, 0, 1, 0], [0.9, 0.1, 0.2, 0.8], [0, 0, 1, 1]) == pytest.approx(0.5)
    assert normalized_cross_entropy([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(1.0)


# --------------------------------------------------------------------- calibrators


def test_isotonic_weighted_pava_hand_computed_and_monotone() -> None:
    cal = IsotonicCalibrator()
    cal.fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    assert cal.predict([0.1, 0.2, 0.3, 0.4]).tolist() == [0.0, 0.5, 0.5, 1.0]
    w = IsotonicCalibrator()
    w.fit([0.1, 0.2, 0.3], [1, 0, 1], weights=[1, 3, 1])
    assert w.predict([0.1, 0.2, 0.3]).tolist() == pytest.approx([0.25, 0.25, 1.0])
    assert w.num_blocks == 2
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000) * 2
    y = (rng.random(2000) < _sigmoid(1.5 * z)).astype(float)
    iso = IsotonicCalibrator()
    iso.fit(_sigmoid(z), y)
    grid = iso.predict(np.linspace(0, 1, 101))
    assert np.all(np.diff(grid) >= -1e-12)
    assert expected_calibration_error(iso.predict(_sigmoid(z)), y) < 0.03
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0.1, 0.2], [0, 1], weights=[0, 0])
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0.1, 0.2], [0, 1], weights=[-1, 1])


def test_platt_recovers_known_parameters_with_weights() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=20_000) * 3
    y = (rng.random(20_000) < _sigmoid(0.5 * z - 1.0)).astype(float)
    cal = PlattCalibrator()
    cal.fit(z, y)
    assert abs(cal.a - 0.5) < 0.05 and abs(cal.b + 1.0) < 0.08 and cal.iterations < 20
    w = rng.integers(1, 4, size=20_000).astype(float)
    wcal = PlattCalibrator()
    wcal.fit(z, y, weights=w)
    dcal = PlattCalibrator()
    dcal.fit(np.repeat(z, w.astype(int)), np.repeat(y, w.astype(int)))
    assert abs(wcal.a - dcal.a) < 1e-4 and abs(wcal.b - dcal.b) < 1e-4


def test_fit_calibrator_fallback_and_auto_selection() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(size=400) * 2
    y = (rng.random(400) < _sigmoid(z)).astype(float)
    res = fit_calibrator("isotonic", z, y, min_positives_isotonic=500)
    assert res.method_used == "platt" and "fallback" in res.reason
    res2 = fit_calibrator("isotonic", z, y, min_positives_isotonic=10)
    assert res2.method_used == "isotonic" and isinstance(res2.calibrator, IsotonicCalibrator)
    # weighted positive count drives the fallback
    res3 = fit_calibrator("isotonic", z, y, weights=np.full(400, 5.0), min_positives_isotonic=500)
    assert res3.method_used == "isotonic"
    # "auto": picks the lower held-out weighted NLL (replicate the seeded split)
    for n, scale in ((150, 1.0), (4000, 1.0)):
        zz = rng.normal(size=n) * scale
        # non-logistic monotone link: isotonic should win at large n
        yy = (rng.random(n) < np.clip(0.5 + 0.5 * np.tanh(3 * zz), 0, 1)).astype(float)
        res = fit_calibrator("auto", zz, yy, seed=7)
        perm = np.random.default_rng(7).permutation(n)
        n_hold = int(round(0.2 * n))
        hold, train = perm[:n_hold], perm[n_hold:]
        nll = {}
        for kind, cal in (("isotonic", IsotonicCalibrator()), ("platt", PlattCalibrator())):
            cal.fit(zz[train] if kind == "platt" else _sigmoid(zz[train]), yy[train])
            nll[kind] = weighted_nll(apply_calibrator(cal, zz[hold]), yy[hold])
        expect = "isotonic" if nll["isotonic"] <= nll["platt"] else "platt"
        assert res.method_used == expect
    with pytest.raises(ValueError):
        fit_calibrator("temperature", z, y)  # type: ignore[arg-type]


def test_p3_is_never_fitted_on_unobserved_rows_and_uses_weights(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 3000
    z1, z2, z3 = (rng.normal(size=n) * 2 for _ in range(3))
    y_click = (rng.random(n) < _sigmoid(z1)).astype(float)
    y_apply = y_click * (rng.random(n) < _sigmoid(z2))
    y_approve_oracle = y_apply * (rng.random(n) < _sigmoid(z3))
    observed = rng.random(n) > 0.4
    y_approve = np.where(observed, y_approve_oracle, 0.0)
    weight = observed.astype(float) * rng.uniform(1.0, 3.0, size=n)
    mask = np.ones(n, dtype=bool)
    cfg = CalibrationConfig(method="isotonic", min_positives_isotonic=10)
    cs = fit_funnel_calibrators(
        z1, z2, z3, y_click, y_apply, y_approve, observed, weight, mask, cfg
    )
    assert set(cs.calibrators) == {"click", "apply", "approve"}
    # reference: fit p3 only on resolved applications with the same weights
    rows = (y_apply > 0.5) & observed
    ref = IsotonicCalibrator()
    ref.fit(_sigmoid(z3[rows]), y_approve[rows], weights=weight[rows])
    grid = rng.normal(size=50) * 2
    assert np.allclose(
        apply_calibrator(cs.calibrators["approve"], grid), ref.predict(_sigmoid(grid))
    )
    # a calibrator fitted on ALL applied rows (pending included) differs
    wrong = IsotonicCalibrator()
    wrong.fit(_sigmoid(z3[y_apply > 0.5]), y_approve[y_apply > 0.5])
    assert not np.allclose(wrong.predict(_sigmoid(grid)), ref.predict(_sigmoid(grid)))
    # calibrate() applies each calibrator on its native input and round-trips through JSON
    p1, p2, p3 = cs.calibrate(z1[:5], z2[:5], z3[:5])
    assert p1.shape == (5,) and np.all((p3 >= 0) & (p3 <= 1))
    cs.save(tmp_path / "cal.json")
    back = CalibratorSet.load(tmp_path / "cal.json")
    q1, q2, q3 = back.calibrate(z1[:5], z2[:5], z3[:5])
    assert np.allclose(p1, q1) and np.allclose(p2, q2) and np.allclose(p3, q3)
    assert json.loads((tmp_path / "cal.json").read_text())["reports"]["approve"].startswith(
        "isotonic"
    )
    # missing calibrators fall back to the sigmoid
    empty = CalibratorSet()
    r1, _, _ = empty.calibrate([0.0], [0.0], [0.0])
    assert r1[0] == 0.5


# ---------------------------------------------------------------------- valuation


def _fp(item_id: int, name: str, family: ProductFamily, **kw: object) -> FinancialProduct:
    base: dict[str, object] = dict(
        item_id=item_id, name=name, family=family, min_fico=600, max_dti=0.6,
        min_annual_income=0.0, licensed_states=frozenset({"CA"}), reward_rate=0.0,
        term_months=0, partner_payout=100.0, annual_fee=0.0,
    )  # fmt: skip
    base.update(kw)
    return FinancialProduct.model_validate(base)


def _products() -> list[FinancialProduct]:
    return [
        _fp(1, "cc", ProductFamily.CREDIT_CARD, apr=0.2, annual_fee=95.0,
            signup_bonus_value=200.0, reward_rate=0.02),
        _fp(2, "bt", ProductFamily.BALANCE_TRANSFER_CARD, apr=0.18, intro_apr_months=18,
            balance_transfer_fee_rate=0.03),
        _fp(3, "pl", ProductFamily.PERSONAL_LOAN, apr=0.10, origination_fee_rate=0.05,
            term_months=36),
        _fp(4, "auto", ProductFamily.AUTO_REFINANCE, apr=0.05, origination_fee_rate=0.01,
            term_months=60),
        _fp(5, "mort", ProductFamily.MORTGAGE, apr=0.055, origination_fee_rate=0.005,
            closing_costs=4000.0, term_months=360),
        _fp(6, "auto_bad", ProductFamily.AUTO_REFINANCE, apr=0.12, term_months=60),
    ]  # fmt: skip


def _user(**kw: object) -> UserProfile:
    base: dict[str, object] = dict(
        user_index=0, fico=700, dti=0.3, annual_income=80_000.0, state="CA",
        revolving_balance=5000.0, revolving_apr=0.24, annual_card_spend=20_000.0,
        other_debt_balance=10_000.0, other_debt_apr=0.22, other_debt_remaining_months=36,
        auto_loan_balance=20_000.0, auto_loan_rate=0.09, auto_remaining_months=48,
        mortgage_balance=300_000.0, mortgage_rate=0.07, mortgage_remaining_months=300,
    )  # fmt: skip
    base.update(kw)
    return UserProfile.model_validate(base)


def test_net_user_benefit_hand_computed_per_family() -> None:
    econ = ProductEconomics.from_products(_products())
    state = UserFinancialState.from_users([_user(), _user(user_index=1, recent_hard_pulls_30d=2)])
    ids = np.array([[1, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 0]])
    amt = np.array([[0.0, 8000.0, 15_000.0, 0.0, 0.0, 0.0], [0.0] * 6])
    r = net_user_benefit(ids, np.array([0, 1]), amt, econ, state, UserBenefitConfig())
    cc = 0.02 * 20_000 * 2 + 200 - 95 * 2
    bt = 5000 * 0.24 * 1.5 - 0.03 * 5000 - 0.0
    pl = (0.22 - 0.10) * 10_000 * 3 / 2 - 0.05 * 10_000
    old_auto = monthly_payment(np.array([20_000.0]), np.array([0.09]), np.array([48.0]))
    new_auto = monthly_payment(np.array([20_000.0]), np.array([0.05]), np.array([60.0]))
    owed_old = remaining_balance(a(20_000), a(0.09), a(48), a(48))  # paid off: 0
    owed_new = remaining_balance(a(20_000), a(0.05), a(60), a(48))  # 12 months still owed
    assert owed_old[0] == 0.0 and owed_new[0] > 0.0
    auto = float(((old_auto - new_auto) * 48 - (owed_new - owed_old) - 0.01 * 20_000)[0])
    old_m = monthly_payment(a(300_000), a(0.07), a(300))
    new_m = monthly_payment(a(300_000), a(0.055), a(360))
    owed_old_m = remaining_balance(a(300_000), a(0.07), a(300), a(84))
    owed_new_m = remaining_balance(a(300_000), a(0.055), a(360), a(84))
    disc = (1 + 0.055 / 12) ** (-84)
    mort = float(present_value_of_annuity(old_m - new_m, a(0.055), a(84))[0])
    mort = mort - float((owed_new_m - owed_old_m)[0]) * disc - 4000 - 0.005 * 300_000
    assert r.gross[0].tolist() == pytest.approx([cc, bt, pl, auto, mort, r.gross[0, 5]], rel=1e-9)
    assert r.gross[0, 5] < 0  # refinancing 9 % into 12 % costs money
    assert r.hard_pull_cost[0].tolist() == [15.0] * 6
    assert r.nb[0, 0] == pytest.approx(cc - 15.0)
    assert r.refinance_ok[0].tolist() == [1, 1, 1, 1, 1, 0]
    # application fatigue doubles the hard-pull cost; empty slots have zero gross and cost
    assert r.hard_pull_cost[1].tolist() == [30.0, 0, 0, 0, 0, 0] and r.gross[1, 1:].sum() == 0
    # amortization helpers: zero months / zero rate branches
    assert monthly_payment(np.array([1000.0]), np.array([0.0]), np.array([10.0]))[0] == 100.0
    assert monthly_payment(np.array([1000.0]), np.array([0.1]), np.array([0.0]))[0] == 0.0
    assert present_value_of_annuity(np.array([10.0]), np.array([0.0]), np.array([12.0]))[0] == 120.0
    assert remaining_balance(a(1000), a(0.0), a(10), a(4))[0] == pytest.approx(600.0)
    assert remaining_balance(a(1000), a(0.12), a(10), a(10))[0] == pytest.approx(0.0, abs=1e-6)
    # no auto loan -> refinance not ok, gross 0
    r2 = net_user_benefit(
        np.array([[4]]), np.array([0]), np.array([[0.0]]), econ,
        UserFinancialState.from_users([_user(auto_loan_balance=0.0, auto_remaining_months=0)]),
    )  # fmt: skip
    assert r2.refinance_ok[0, 0] == 0 and r2.gross[0, 0] == 0.0
    with pytest.raises(ValueError):
        net_user_benefit(np.array([1, 2]), np.array([0]), np.array([0.0, 0.0]), econ, state)


def test_expected_value_and_utility_endpoints() -> None:
    pf = p_funded([0.5, 0.2], [0.4, 0.5], [0.9, 1.0])
    assert pf.tolist() == pytest.approx([0.18, 0.1])
    assert expected_revenue([0.5], [0.4], [0.9], [100.0])[0] == pytest.approx(18.0)
    nb = np.array([50.0, -10.0])
    payout = np.array([100.0, 200.0])
    assert utility(pf, payout, nb, 1.0).tolist() == pytest.approx((pf * payout).tolist())
    assert utility(pf, payout, nb, 0.0).tolist() == pytest.approx((pf * nb).tolist())
    assert utility(pf, payout, nb, 0.5).tolist() == pytest.approx(
        (pf * (0.5 * payout + 0.5 * nb)).tolist()
    )
    with pytest.raises(ValueError):
        UtilityConfig(alpha=1.5)


def test_guardrails_do_no_harm_refinance_and_pending_family() -> None:
    pf = np.full((1, 5), 0.1)
    payout = np.full((1, 5), 100.0)
    nb = np.array([[-40.0, 30.0, -40.0, 10.0, 5.0]])
    fam = np.array([[0, 0, 2, 3, 0]])  # two cards (one harmful, one safe), a harmful loan alone
    mask = np.array([[True, True, True, True, False]])
    refi_ok = np.array([[1, 1, 1, 0, 1]])
    pend = np.array([[False, True, False, False, False]])
    cfg = UtilityConfig(alpha=0.5, delta=25.0, harm_penalty=25.0, pending_family_penalty=15.0)
    g = apply_guardrails(pf, payout, nb, fam, mask, refi_ok, pend, cfg)
    assert g.keep.tolist() == [[False, True, True, False, False]]  # harmful-alone is kept
    assert g.excluded_harm.tolist() == [[True, False, False, False, False]]  # safe sibling exists
    assert g.penalized_harm.tolist() == [[False, False, True, False, False]]  # no safe sibling
    assert g.excluded_refinance.tolist() == [[False, False, False, True, False]]
    assert g.nb[0, 1] == pytest.approx(15.0)  # pending-family penalty applied to NB
    assert g.utility[0, 1] == pytest.approx(0.1 * (0.5 * 100 + 0.5 * 15.0))
    assert np.isinf(g.utility[0, 0]) and np.isinf(g.utility[0, 4])
    # the penalized (kept) harmful loan is kept only when exclusion is impossible
    g2 = apply_guardrails(pf, payout, nb, fam, mask, np.ones((1, 5)), pend, cfg)
    assert g2.keep[0, 2] and g2.utility[0, 2] == pytest.approx(0.1 * (50 - 20) - 25.0)


def test_alpha_sweep_is_weakly_monotone_and_renders() -> None:
    rng = np.random.default_rng(4)
    b, k = 40, 12
    pf = rng.uniform(0.0, 0.3, size=(b, k))
    payout = rng.uniform(50, 800, size=(b, k))
    nb = rng.normal(100, 300, size=(b, k))
    fam = rng.integers(0, 5, size=(b, k))
    mask = np.ones((b, k), dtype=bool)
    refi = np.ones((b, k))
    pend = np.zeros((b, k), dtype=bool)
    tiers = rng.integers(0, 5, size=b)
    rows = pareto_sweep(pf, payout, nb, fam, mask, refi, pend, tiers, k=5)
    rev = [r.revenue_at_k for r in rows]
    ben = [r.user_benefit_at_k for r in rows]
    assert [r.alpha for r in rows] == pytest.approx([0.1 * i for i in range(11)])
    assert all(a <= b_ + 1e-9 for a, b_ in zip(rev, rev[1:], strict=False))
    assert all(a >= b_ - 1e-9 for a, b_ in zip(ben, ben[1:], strict=False))
    assert rev[-1] > rev[0] and ben[0] > ben[-1]
    assert all(0.0 <= r.harm_rate_at_k <= 1.0 for r in rows)
    md = pareto_table_markdown(rows, k=5)
    assert md.count("\n") > 12 and "| 0.5 |" in md and "SUPER_PRIME" in md


def test_slate_metrics_hand_computed() -> None:
    pf = np.array([[0.5, 0.1, 0.2]])
    payout = np.array([[100.0, 100.0, 50.0]])
    nb = np.array([[-5.0, 20.0, 30.0]])
    order = np.array([[2, 0, 1]])
    assert expected_revenue_at_k(pf, payout, order, 2)[0] == pytest.approx(10.0 + 50.0)
    assert expected_user_benefit_at_k(pf, nb, order, 2)[0] == pytest.approx(6.0 - 2.5)
    assert harm_rate_at_k(nb, order, 2)[0] == pytest.approx(0.5)
    mask = np.array([[False, True, True]])
    assert harm_rate_at_k(nb, order, 2, mask)[0] == pytest.approx(0.0)
    assert np.isnan(harm_rate_at_k(nb, order, 2, np.zeros((1, 3), bool))[0])


# ---------------------------------------------------------------------------- PRM


def test_prm_features_targets_and_rerank() -> None:
    b, k, d = 2, 4, 8
    h = torch.randn(b, k, d)
    p = torch.rand(b, k)
    ev = torch.rand(b, k) * 100
    nb = torch.randn(b, k) * 50
    u = torch.randn(b, k) * 40
    u[0, 3] = float("-inf")
    fam = torch.tensor([[0, 0, 2, -1], [1, 4, 4, 4]])
    feats = build_prm_features(h, p, p, p, ev, nb, u, fam)
    cfg = PRMConfig(cand_dim=d, user_dim=d, slate_size=4)
    assert feats.shape == (b, k, cfg.feature_dim) and torch.isfinite(feats).all()
    assert feats[0, 3, d + 6 : d + 11].sum() == 0  # family -1 -> all-zero one-hot
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    out = PRM(cfg)(feats, torch.randn(b, d), mask)
    assert out.scores.shape == (b, k) and out.scores[0, 3] == float("-inf")
    # utility target stays a distribution with negative utilities
    q, has = prm_target_distribution(torch.zeros(b, k), mask, "utility", u, 25.0)
    assert has.all() and torch.allclose(q.sum(-1), torch.ones(b)) and q[0, 3] == 0
    labels = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    q2, has2 = prm_target_distribution(labels, mask, "click")
    assert has2.tolist() == [True, False] and q2[0].tolist() == [0, 1, 0, 0]
    loss_none = prm_listwise_loss(out.scores, torch.zeros(b, k), mask, fam)
    assert float(loss_none.detach()) == 0.0
    loss = prm_listwise_loss(out.scores, labels, mask, fam, cannibalization_weight=0.5)
    plain = prm_listwise_loss(out.scores, labels, mask, fam).detach()
    assert float(loss.detach()) >= float(plain)
    with pytest.raises(ValueError):
        prm_target_distribution(labels, mask, "utility")
    # rerank: penalty diversifies families; masked slots come last; distinct families -> 0 penalty
    scores = torch.tensor([[3.0, 2.9, 1.0, 0.5]])
    fam1 = torch.tensor([[1, 1, 2, 3]])
    m = torch.ones(1, 4, dtype=torch.bool)
    assert rerank(scores, fam1, m, 0.0)[0].tolist() == [0, 1, 2, 3]
    assert rerank(scores, fam1, m, 2.5)[0].tolist() == [0, 2, 3, 1]
    m2 = torch.tensor([[True, False, True, True]])
    assert rerank(scores, fam1, m2, 0.0)[0].tolist()[-1] == 1
    assert float(same_family_penalty(scores, torch.tensor([[0, 1, 2, 3]]), m)[0]) == 0.0
