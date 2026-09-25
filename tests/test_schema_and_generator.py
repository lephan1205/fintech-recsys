"""Schema contracts and synthetic-generator invariants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from recsys import seed_everything
from recsys.data.schema import (
    ActionType,
    CreditTier,
    FinancialProduct,
    ImpressionSlate,
    InteractionEvent,
    ProductFamily,
    SyntheticDataset,
    UserProfile,
)
from recsys.data.synthetic_generator import (
    GeneratorConfig,
    SyntheticFintechDataGenerator,
    approve_probability,
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
    assert [a.value for a in ActionType] == [0, 1, 2, 3, 4, 5]
    assert ActionType.PAD.value == 0 and ActionType.SCORE_CHANGE.value == 2


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
    assert ds.catalog_features.shape == (ds.num_items + 1, 13)
    assert np.all(ds.catalog_features[0] == 0)
    assert ds.user_features.shape == (ds.num_users, 9)
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
