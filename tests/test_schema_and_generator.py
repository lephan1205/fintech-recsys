"""Schema contracts and synthetic-generator invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from recsys import seed_everything
from recsys.data.schema import (
    APPLY_ACTIONS,
    NUM_FAMILIES,
    POSITIVE_ACTIONS,
    PRODUCT_FEATURE_DIM,
    USER_FEATURE_DIM,
    ActionType,
    ApplicationStatus,
    CreditTier,
    DelayConfig,
    FinancialProduct,
    ImpressionSlate,
    InteractionEvent,
    ProductFamily,
    SyntheticDataset,
    UserProfile,
    observed_status,
)
from recsys.data.synthetic_generator import (
    FORMAT_VERSION,
    GeneratorConfig,
    SyntheticFintechDataGenerator,
    approve_probability,
    dataset_summary,
    load_dataset,
    save_dataset,
)

SMALL = GeneratorConfig(num_users=60, num_products=120, seed=7, slates_per_user=2)


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)


@pytest.fixture(scope="module")
def small_dataset() -> tuple[SyntheticDataset, SyntheticFintechDataGenerator]:
    gen = SyntheticFintechDataGenerator(SMALL)
    ds = gen.generate()
    return ds, gen


def _product(**overrides: object) -> FinancialProduct:
    base: dict[str, object] = dict(
        item_id=1,
        name="p",
        family=ProductFamily.CREDIT_CARD,
        min_fico=640,
        max_dti=0.45,
        min_annual_income=30_000.0,
        licensed_states=frozenset({"CA", "NY"}),
        apr=0.2,
        annual_fee=0.0,
        reward_rate=0.02,
        term_months=0,
        partner_payout=150.0,
    )
    base.update(overrides)
    return FinancialProduct.model_validate(base)


# ----------------------------------------------------------------------------- schema


def test_action_type_values() -> None:
    assert [a.value for a in ActionType] == [0, 1, 2, 3, 4, 5, 6]
    assert ActionType.PAD.value == 0 and ActionType.SCORE_CHANGE.value == 2
    assert ActionType.APPLY_PENDING.value == 6
    assert ActionType.APPLY_PENDING in POSITIVE_ACTIONS
    assert {
        ActionType.APPLY_APPROVED, ActionType.APPLY_DECLINED, ActionType.APPLY_PENDING
    } == APPLY_ACTIONS  # fmt: skip
    assert ActionType.APPLY_DECLINED not in POSITIVE_ACTIONS


def test_defaults_keep_reference_style_objects_valid() -> None:
    """Every §4.1 extension is defaulted, so v1-shaped products / users still validate."""
    p = _product()
    assert p.intro_apr_months == 0 and p.signup_bonus_value == 0.0 and p.closing_costs == 0.0
    assert p.to_feature_vector().shape == (PRODUCT_FEATURE_DIM,) == (18,)
    u = UserProfile(user_index=0, fico=700, dti=0.2, annual_income=5e4, state="CA")
    assert u.pending_product_ids == () and u.recent_hard_pulls_30d == 0
    vec = u.to_feature_vector()
    assert vec.shape == (USER_FEATURE_DIM,) == (26,)
    assert vec[16 + u.tier_index] == 1.0 and vec[21:].sum() == 0.0
    pend = u.model_copy(update={"pending_product_ids": (3,), "pending_family_ids": (4,)})
    assert pend.to_feature_vector()[21 + 4] == 1.0
    with pytest.raises(ValidationError):
        UserProfile(user_index=0, fico=700, dti=0.2, annual_income=5e4, state="CA",
                    pending_family_ids=(NUM_FAMILIES,))  # fmt: skip
    with pytest.raises(ValidationError):
        UserProfile(user_index=0, fico=700, dti=0.2, annual_income=5e4, state="CA",
                    pending_product_ids=(0,))  # fmt: skip
    bt = _product(intro_apr_months=18, balance_transfer_fee_rate=0.03, signup_bonus_value=200.0)
    v = bt.to_feature_vector()
    assert v[13] == 18 / 24 and v[14] == pytest.approx(0.3) and v[16] > 0.0


def test_observed_status_and_delay_config() -> None:
    kw = dict(served_at_days=100.0, decision_delay_days=10.0)
    assert observed_status(0, 0, snapshot_at_days=105.0, **kw) is ApplicationStatus.NOT_APPLIED
    assert observed_status(1, 1, snapshot_at_days=105.0, **kw) is ApplicationStatus.PENDING
    assert observed_status(1, 1, snapshot_at_days=110.0, **kw) is ApplicationStatus.APPROVED
    assert observed_status(1, 0, snapshot_at_days=110.0, **kw) is ApplicationStatus.DECLINED
    cfg = DelayConfig()
    assert len(cfg.p_instant) == NUM_FAMILIES and cfg.instant_delay_days > 0.0
    with pytest.raises(ValueError):
        DelayConfig(p_instant=((1.5, 0.0),) * NUM_FAMILIES)
    with pytest.raises(ValueError):
        DelayConfig(log_sigma=((0.0, 0.1),) * NUM_FAMILIES)
    with pytest.raises(ValueError):
        GeneratorConfig(family_mix=(1.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("fico", "tier"),
    [
        (579, CreditTier.DEEP_SUBPRIME),
        (580, CreditTier.SUBPRIME),
        (619, CreditTier.SUBPRIME),
        (620, CreditTier.NEAR_PRIME),
        (679, CreditTier.NEAR_PRIME),
        (680, CreditTier.PRIME),
        (739, CreditTier.PRIME),
        (740, CreditTier.SUPER_PRIME),
    ],
)
def test_credit_tier_from_fico_boundaries(fico: int, tier: CreditTier) -> None:
    assert CreditTier.from_fico(fico) is tier


def test_event_item_zero_only_for_score_change() -> None:
    InteractionEvent(item_id=0, action=ActionType.SCORE_CHANGE, timestamp=0.0)
    with pytest.raises(ValidationError):
        InteractionEvent(item_id=0, action=ActionType.VIEW, timestamp=0.0)
    with pytest.raises(ValidationError):
        InteractionEvent(item_id=5, action=ActionType.SCORE_CHANGE, timestamp=0.0)
    with pytest.raises(ValidationError):
        InteractionEvent(item_id=5, action=ActionType.PAD, timestamp=0.0)


def test_slate_funnel_monotonicity_validated() -> None:
    kwargs: dict[str, object] = dict(
        user_index=0,
        slate_id=0,
        candidate_item_ids=(1, 2),
        p_click=(0.1, 0.2),
        p_apply=(0.3, 0.3),
        p_approve=(0.5, 0.0),
        payouts=(10.0, 20.0),
        eligible=(True, False),
        served_at_days=10.0,
        decision_delay_days=(2.0, 0.0),
    )
    ImpressionSlate.model_validate(
        {
            **kwargs,
            "y_click": (1, 0),
            "y_apply": (1, 0),
            "y_approve": (1, 0),
            "amounts": (500.0, 0.0),
        }
    )
    with pytest.raises(ValidationError):  # decision delay without an application
        ImpressionSlate.model_validate(
            {
                **kwargs,
                "y_click": (1, 0),
                "y_apply": (0, 0),
                "y_approve": (0, 0),
                "amounts": (0.0, 0.0),
            }
        )
    with pytest.raises(ValidationError):  # application without a decision delay
        ImpressionSlate.model_validate(
            {
                **kwargs,
                "decision_delay_days": (0.0, 0.0),
                "y_click": (1, 0),
                "y_apply": (1, 0),
                "y_approve": (0, 0),
                "amounts": (0.0, 0.0),
            }
        )
    with pytest.raises(ValidationError):  # served_at_days / decision_delay_days are required
        ImpressionSlate.model_validate(
            {
                **{k: v for k, v in kwargs.items() if k != "served_at_days"},
                "y_click": (0, 0),
                "y_apply": (0, 0),
                "y_approve": (0, 0),
                "amounts": (0.0, 0.0),
            }
        )
    with pytest.raises(ValidationError):  # apply without click
        ImpressionSlate.model_validate(
            {
                **kwargs,
                "y_click": (0, 0),
                "y_apply": (1, 0),
                "y_approve": (0, 0),
                "amounts": (0.0, 0.0),
            }
        )
    with pytest.raises(ValidationError):  # amount without approval
        ImpressionSlate.model_validate(
            {
                **kwargs,
                "y_click": (1, 0),
                "y_apply": (1, 0),
                "y_approve": (0, 0),
                "amounts": (500.0, 0.0),
            }
        )
    with pytest.raises(ValidationError):  # ineligible with p_approve > 0
        ImpressionSlate.model_validate(
            {
                **kwargs,
                "p_approve": (0.5, 0.5),
                "y_click": (0, 0),
                "y_apply": (0, 0),
                "y_approve": (0, 0),
                "amounts": (0.0, 0.0),
            }
        )


def test_user_profile_rejects_unknown_state() -> None:
    with pytest.raises(ValidationError):
        UserProfile(user_index=0, fico=700, dti=0.2, annual_income=5e4, state="ZZ")


def test_approve_probability_hard_gates() -> None:
    p = _product()
    assert approve_probability(700, 0.3, 50_000.0, "CA", p) > 0.5
    assert approve_probability(639, 0.3, 50_000.0, "CA", p) == 0.0
    assert approve_probability(700, 0.46, 50_000.0, "CA", p) == 0.0
    assert approve_probability(700, 0.3, 29_999.0, "CA", p) == 0.0
    assert approve_probability(700, 0.3, 50_000.0, "TX", p) == 0.0


# -------------------------------------------------------------------------- generator


def test_generator_deterministic(tmp_path: Path) -> None:
    a = SyntheticFintechDataGenerator(SMALL).generate()
    b = SyntheticFintechDataGenerator(SMALL).generate()
    save_dataset(a, tmp_path / "a", SMALL)
    save_dataset(b, tmp_path / "b", SMALL)
    for name in ("products.jsonl", "users.jsonl", "interactions.jsonl", "slates.jsonl"):
        ha = hashlib.sha256((tmp_path / "a" / name).read_bytes()).hexdigest()
        hb = hashlib.sha256((tmp_path / "b" / name).read_bytes()).hexdigest()
        assert ha == hb, name


def test_fico_dti_income_correlations() -> None:
    cfg = GeneratorConfig(num_users=500, num_products=50, seed=3, slates_per_user=1)
    ds = SyntheticFintechDataGenerator(cfg).generate()
    fico = np.array([u.fico for u in ds.users], dtype=float)
    dti = np.array([u.dti for u in ds.users], dtype=float)
    inc = np.log(np.array([u.annual_income for u in ds.users], dtype=float))
    assert np.corrcoef(fico, dti)[0, 1] < -0.2
    assert np.corrcoef(fico, inc)[0, 1] > 0.2


def test_catalog_tiers_and_family_floors(
    small_dataset: tuple[SyntheticDataset, SyntheticFintechDataGenerator],
) -> None:
    ds, _ = small_dataset
    ids = [p.item_id for p in ds.products]
    assert ids == list(range(1, len(ids) + 1))
    assert {p.family for p in ds.products} == set(ProductFamily)
    for p in ds.products:
        if p.family is ProductFamily.MORTGAGE:
            assert p.min_fico >= 620
        assert p.required_tier is CreditTier.from_fico(p.min_fico)


def test_history_contains_score_change_and_declines(
    small_dataset: tuple[SyntheticDataset, SyntheticFintechDataGenerator],
) -> None:
    ds, gen = small_dataset
    by_id = {p.item_id: p for p in ds.products}
    n_score, n_declined, n_approved = 0, 0, 0
    for rec, user, fico_path in zip(ds.interactions, ds.users, gen.fico_at_event, strict=True):
        assert rec.target.item_id > 0
        assert all(e.action != ActionType.PAD for e in rec.history)
        for t, e in enumerate(rec.history):
            if e.action == ActionType.SCORE_CHANGE:
                assert e.item_id == 0
                n_score += 1
            elif e.action == ActionType.APPLY_DECLINED:
                n_declined += 1
            elif e.action == ActionType.APPLY_APPROVED:
                n_approved += 1
                prod = by_id[e.item_id]
                assert int(fico_path[t]) >= prod.min_fico
                assert user.dti <= prod.max_dti
                assert user.annual_income >= prod.min_annual_income
                assert user.state in prod.licensed_states
    assert n_score > 0 and n_declined > 0 and n_approved > 0


def test_pending_events_and_user_pending_context(
    small_dataset: tuple[SyntheticDataset, SyntheticFintechDataGenerator],
) -> None:
    """Histories end near the snapshot, contain APPLY_PENDING events, and the user's
    pending_product_ids / pending_family_ids agree with those events."""
    ds, _ = small_dataset
    snap = ds.snapshot_at_days
    assert snap == SMALL.snapshot_at_days
    family = ds.family_by_item()
    n_pending, n_users_pending = 0, 0
    for rec, user in zip(ds.interactions, ds.users, strict=True):
        # the future window may repeat the target, so dedupe (events are frozen / hashable)
        events = tuple({*rec.history, rec.target, *rec.future_window})
        assert all(e.timestamp >= 0.0 for e in events)
        assert rec.target.timestamp <= snap
        pending_seen = {e.item_id for e in events if e.action == ActionType.APPLY_PENDING}
        n_pending += len(pending_seen)
        # every pending event visible in the record is in the user's pending context
        assert pending_seen <= set(user.pending_product_ids)
        assert set(user.pending_family_ids) == {int(family[i]) for i in user.pending_product_ids}
        assert (
            user.recent_hard_pulls_30d
            >= sum(
                1 for e in events if e.action in APPLY_ACTIONS and snap - 30.0 < e.timestamp <= snap
            )
            - 0
        )  # Poisson component is non-negative
        if user.pending_product_ids:
            n_users_pending += 1
    assert n_pending > 0 and n_users_pending > 0
    # financial state is drawn for everyone and correlates with the credit profile
    assert all(u.revolving_apr > 0.0 and u.annual_card_spend > 0.0 for u in ds.users)
    assert any(u.mortgage_balance > 0.0 for u in ds.users)


def test_slate_delays_and_summary(
    small_dataset: tuple[SyntheticDataset, SyntheticFintechDataGenerator],
) -> None:
    ds, _ = small_dataset
    snap = ds.snapshot_at_days
    for s in ds.slates:
        assert snap - SMALL.served_window_days <= s.served_at_days <= snap
        for ya, d in zip(s.y_apply, s.decision_delay_days, strict=True):
            assert (d > 0.0) == (ya == 1)
    summary = dataset_summary(ds)
    assert summary["frac_apply_pending_events"] > 0.0
    assert 0.0 < summary["pending_rate_given_apply"] < 1.0
    assert summary["impressions"] == len(ds.slates) * SMALL.slate_size


def test_slate_labels_consistent_with_true_probs(
    small_dataset: tuple[SyntheticDataset, SyntheticFintechDataGenerator],
) -> None:
    ds, _ = small_dataset
    p_click = np.array([v for s in ds.slates for v in s.p_click])
    y_click = np.array([v for s in ds.slates for v in s.y_click])
    for s in ds.slates:
        assert len(s.candidate_item_ids) == SMALL.slate_size
        assert len(set(s.candidate_item_ids)) == SMALL.slate_size
        for elig, pa in zip(s.eligible, s.p_approve, strict=True):
            if not elig:
                assert pa == 0.0
        # Users with thin eligible sets are backfilled with ineligible candidates.
        assert sum(1 for e in s.eligible if not e) >= SMALL.ineligible_per_slate
    assert any(sum(1 for e in s.eligible if not e) == SMALL.ineligible_per_slate for s in ds.slates)
    assert abs(y_click.mean() - p_click.mean()) < 0.05
    assert 0.0 < y_click.mean() < 0.5


def test_item_counts_and_features(
    small_dataset: tuple[SyntheticDataset, SyntheticFintechDataGenerator],
) -> None:
    ds, _ = small_dataset
    assert ds.catalog_features.shape == (ds.num_items + 1, PRODUCT_FEATURE_DIM)
    assert np.all(ds.catalog_features[0] == 0)
    assert ds.user_features.shape == (ds.num_users, USER_FEATURE_DIM)
    # user_features is rebuilt after the pending context is filled in
    idx = next(i for i, u in enumerate(ds.users) if u.pending_family_ids)
    assert ds.user_features[idx, 21 + ds.users[idx].pending_family_ids[0]] == 1.0
    assert ds.item_counts.shape == (ds.num_items + 1,)
    assert ds.item_counts[0] == 0
    total = sum(len(r.history) + 1 for r in ds.interactions)
    n_score = sum(1 for r in ds.interactions for e in r.history if e.item_id == 0)
    assert ds.item_counts.sum() == total - n_score


def test_roundtrip_save_load(
    tmp_path: Path, small_dataset: tuple[SyntheticDataset, SyntheticFintechDataGenerator]
) -> None:
    ds, _ = small_dataset
    save_dataset(ds, tmp_path, SMALL)
    back = load_dataset(tmp_path)
    assert back.products == ds.products
    assert back.users == ds.users
    assert back.interactions == ds.interactions
    assert back.slates == ds.slates
    assert np.allclose(back.catalog_features, ds.catalog_features)
    assert np.array_equal(back.item_counts, ds.item_counts)
    assert back.snapshot_at_days == ds.snapshot_at_days
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["format_version"] == FORMAT_VERSION == 2
    assert meta["snapshot_at_days"] == ds.snapshot_at_days


def test_load_rejects_format_version_1(tmp_path: Path) -> None:
    ds = SyntheticFintechDataGenerator(SMALL).generate()
    save_dataset(ds, tmp_path, SMALL)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["format_version"] = 1
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format_version=1"):
        load_dataset(tmp_path)
