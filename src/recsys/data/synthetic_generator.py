"""Synthetic credit-marketplace data generator.

Design goals
------------
* **Correlated credit attributes.** FICO drives both DTI (negatively) and income
  (positively), so tabular models have real structure to learn.
* **Hard underwriting.** Approval is impossible when any declarative gate fails
  (``min_fico``, ``max_dti``, ``min_annual_income``, ``licensed_states``) and
  otherwise follows a logistic in the FICO / DTI margins.  The *same* function
  (:func:`approve_probability`) drives both history simulation and slate labels,
  so the funnel is internally consistent.
* **Realistic sequences.** Histories mix product events with user-level
  ``SCORE_CHANGE`` events (item id 0) that drift the user's FICO, and contain
  adverse ``APPLY_DECLINED`` signals.
* **Determinism.** Everything is drawn from a single ``numpy`` generator in a
  fixed order; the same seed yields byte-identical serialized output.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from recsys.data.schema import (
    FAMILY_ORDER,
    NUM_FAMILIES,
    NUM_STATES,
    NUM_TIERS,
    POSITIVE_ACTIONS,
    PRODUCT_FEATURE_DIM,
    SCORE_CHANGE_ITEM_ID,
    STATE_INDEX,
    TIER_FICO_FLOORS,
    US_STATES,
    USER_FEATURE_DIM,
    ActionType,
    FinancialProduct,
    ImpressionSlate,
    InteractionEvent,
    InteractionRecord,
    ProductFamily,
    SyntheticDataset,
    UserProfile,
)

FORMAT_VERSION = 1

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratorConfig:
    num_users: int = 2000
    num_products: int = 10_000
    seed: int = 42
    min_history: int = 5
    max_history: int = 60
    horizon_days: float = 365.0
    mean_gap_days: float = 3.0
    score_change_prob: float = 0.12
    apply_prob: float = 0.15
    credit_pull_prob: float = 0.10
    ineligible_browse_prob: float = 0.20
    slates_per_user: int = 3
    slate_size: int = 10
    ineligible_per_slate: int = 2
    future_window_days: tuple[float, float] = (14.0, 28.0)
    max_future: int = 16

    def __post_init__(self) -> None:
        if self.min_history < 2 or self.max_history < self.min_history:
            raise ValueError("need 2 <= min_history <= max_history")
        if self.ineligible_per_slate >= self.slate_size:
            raise ValueError("ineligible_per_slate must be < slate_size")


# --------------------------------------------------------------------------- #
# Family-level parameters (index-aligned with FAMILY_ORDER)
# --------------------------------------------------------------------------- #

_FAMILY_MIX = np.array([0.40, 0.15, 0.20, 0.10, 0.15])
_TIER_MIX_BY_FAMILY = np.array(
    [
        [0.15, 0.20, 0.25, 0.25, 0.15],  # CREDIT_CARD (secured cards reach deep subprime)
        [0.00, 0.05, 0.25, 0.40, 0.30],  # BALANCE_TRANSFER_CARD
        [0.05, 0.20, 0.35, 0.25, 0.15],  # PERSONAL_LOAN
        [0.00, 0.15, 0.35, 0.30, 0.20],  # AUTO_REFINANCE
        [0.00, 0.00, 0.30, 0.40, 0.30],  # MORTGAGE (never below NEAR_PRIME)
    ]
)
_TIER_FICO_JITTER = (280, 40, 60, 60, 60)
_TIER_APR_BASE = (0.30, 0.26, 0.22, 0.18, 0.14)
_FAMILY_APR_SCALE = (1.0, 0.9, 0.8, 0.5, 0.35)
_FAMILY_DTI_RANGE = ((0.50, 0.70), (0.50, 0.70), (0.40, 0.55), (0.40, 0.55), (0.36, 0.45))
_FAMILY_LOG_INCOME_MU = (9.6, 9.8, 10.1, 10.1, 10.7)
_FAMILY_LOG_PAYOUT_MU = (5.0, 5.2, 5.5, 5.85, 7.1)
_FAMILY_LOG_AMOUNT_MU = (8.0, 8.3, 9.2, 9.9, 12.6)
_FAMILY_TERMS: tuple[tuple[int, ...], ...] = (
    (0,), (0,), (24, 36, 48, 60), (36, 48, 60, 72), (180, 360),
)  # fmt: skip
_CARD_FEES = np.array([0.0, 95.0, 250.0, 550.0])
_CARD_FEE_P = np.array([0.60, 0.25, 0.10, 0.05])
_IS_CARD = (True, True, False, False, False)


# --------------------------------------------------------------------------- #
# Approval model (single source of truth)
# --------------------------------------------------------------------------- #


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def approve_probability(
    fico: int, dti: float, annual_income: float, state: str, product: FinancialProduct
) -> float:
    """p(approve | apply).  Exactly 0 when any hard underwriting gate fails."""
    if (
        fico < product.min_fico
        or dti > product.max_dti
        or annual_income < product.min_annual_income
        or state not in product.licensed_states
    ):
        return 0.0
    z = 0.04 * (fico - product.min_fico) + 3.0 * (product.max_dti - dti) - 0.5
    return _sigmoid(z)


@dataclass
class _GateArrays:
    """Vectorized copies of the catalog gates (index = item_id; row 0 never eligible)."""

    min_fico: npt.NDArray[np.int64]
    max_dti: npt.NDArray[np.float64]
    min_income: npt.NDArray[np.float64]
    licensed: npt.NDArray[np.bool_]  # (N+1, NUM_STATES)
    family: npt.NDArray[np.int64]
    apr: npt.NDArray[np.float64]
    reward: npt.NDArray[np.float64]
    payout: npt.NDArray[np.float64]

    @classmethod
    def from_products(cls, products: list[FinancialProduct]) -> _GateArrays:
        n = len(products) + 1
        g = cls(
            min_fico=np.full(n, 10_000, dtype=np.int64),
            max_dti=np.full(n, -1.0),
            min_income=np.full(n, np.inf),
            licensed=np.zeros((n, NUM_STATES), dtype=bool),
            family=np.full(n, -1, dtype=np.int64),
            apr=np.zeros(n),
            reward=np.zeros(n),
            payout=np.zeros(n),
        )
        for p in products:
            i = p.item_id
            g.min_fico[i] = p.min_fico
            g.max_dti[i] = p.max_dti
            g.min_income[i] = p.min_annual_income
            for s in p.licensed_states:
                g.licensed[i, STATE_INDEX[s]] = True
            g.family[i] = p.family_index
            g.apr[i] = p.apr
            g.reward[i] = p.reward_rate
            g.payout[i] = p.partner_payout
        return g

    def eligible(
        self, fico: int, dti: float, income: float, state_idx: int
    ) -> npt.NDArray[np.bool_]:
        out: npt.NDArray[np.bool_] = (
            (self.min_fico <= fico)
            & (self.max_dti >= dti)
            & (self.min_income <= income)
            & self.licensed[:, state_idx]
        )
        out[0] = False
        return out

    def approve_prob(
        self, fico: int, dti: float, item_ids: npt.NDArray[np.int64], elig: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float64]:
        z = 0.04 * (fico - self.min_fico[item_ids]) + 3.0 * (self.max_dti[item_ids] - dti) - 0.5
        p: npt.NDArray[np.float64] = 1.0 / (1.0 + np.exp(-z))
        return np.where(elig, p, 0.0)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class SyntheticFintechDataGenerator:
    """Generate a complete :class:`SyntheticDataset` from a :class:`GeneratorConfig`."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        #: Per-user FICO at each history event (filled by :meth:`generate`; used by tests
        #: to assert that every APPLY_APPROVED event passed the hard gates).
        self.fico_at_event: list[npt.NDArray[np.int64]] = []

    # ------------------------------------------------------------------ products
    def generate_products(self) -> list[FinancialProduct]:
        cfg, rng = self.config, self.rng
        products: list[FinancialProduct] = []
        fam_idx = rng.choice(NUM_FAMILIES, size=cfg.num_products, p=_FAMILY_MIX)
        for item_id, f in enumerate(fam_idx.tolist(), start=1):
            tier = int(rng.choice(NUM_TIERS, p=_TIER_MIX_BY_FAMILY[f]))
            min_fico = TIER_FICO_FLOORS[tier] + int(rng.integers(0, _TIER_FICO_JITTER[tier]))
            lo, hi = _FAMILY_DTI_RANGE[f]
            max_dti = float(np.round(rng.uniform(lo, hi), 3))
            min_income = float(np.round(np.exp(_FAMILY_LOG_INCOME_MU[f] + 0.3 * rng.normal()), -2))
            if rng.random() < 0.7:
                states: frozenset[str] = frozenset(US_STATES)
            else:
                k = int(rng.integers(20, 46))
                states = frozenset(rng.choice(US_STATES, size=k, replace=False).tolist())
            apr = _TIER_APR_BASE[tier] * _FAMILY_APR_SCALE[f] + 0.02 * rng.normal()
            apr = float(np.clip(np.round(apr, 4), 0.02, 0.45))
            fee = float(rng.choice(_CARD_FEES, p=_CARD_FEE_P)) if _IS_CARD[f] else 0.0
            reward = float(np.round(rng.uniform(0.01, 0.05), 4)) if _IS_CARD[f] else 0.0
            term = int(rng.choice(_FAMILY_TERMS[f]))
            payout = float(np.round(np.exp(_FAMILY_LOG_PAYOUT_MU[f] + 0.3 * rng.normal()), 2))
            family = FAMILY_ORDER[f]
            products.append(
                FinancialProduct(
                    item_id=item_id,
                    name=f"{family.value.lower()}_{item_id}",
                    family=family,
                    min_fico=min_fico,
                    max_dti=max_dti,
                    min_annual_income=min_income,
                    licensed_states=states,
                    apr=apr,
                    annual_fee=fee,
                    reward_rate=reward,
                    term_months=term,
                    partner_payout=payout,
                )
            )
        return products

    # --------------------------------------------------------------------- users
    def generate_users(self, gates: _GateArrays) -> list[UserProfile]:
        cfg, rng = self.config, self.rng
        users: list[UserProfile] = []
        for u in range(cfg.num_users):
            fico = int(np.clip(np.round(rng.normal(690.0, 75.0)), 300, 850))
            dti = float(np.clip(0.30 - 0.0012 * (fico - 690) + rng.normal(0.0, 0.08), 0.02, 0.95))
            dti = round(dti, 4)
            income = float(np.exp(10.9 + 0.004 * (fico - 690) + rng.normal(0.0, 0.45)))
            income = float(np.round(np.clip(income, 12_000.0, 600_000.0), -2))
            state = str(rng.choice(US_STATES))
            elig = np.flatnonzero(gates.eligible(fico, dti, income, STATE_INDEX[state]))
            n_held = int(rng.integers(0, 4))
            held: tuple[int, ...] = ()
            if n_held > 0 and elig.size > 0:
                picks = rng.choice(elig, size=min(n_held, elig.size), replace=False)
                held = tuple(sorted(int(i) for i in picks))
            users.append(
                UserProfile(
                    user_index=u,
                    fico=fico,
                    dti=dti,
                    annual_income=income,
                    state=state,
                    held_product_ids=held,
                )
            )
        return users

    # ------------------------------------------------------------------ histories
    def _sample_item(
        self,
        family: int,
        elig_by_family: list[npt.NDArray[np.int64]],
        all_by_family: list[npt.NDArray[np.int64]],
    ) -> int:
        cfg, rng = self.config, self.rng
        pool = elig_by_family[family]
        if pool.size == 0 or rng.random() < cfg.ineligible_browse_prob:
            pool = all_by_family[family]
        return int(rng.choice(pool))

    def generate_interactions(
        self,
        users: list[UserProfile],
        products: list[FinancialProduct],
        gates: _GateArrays,
    ) -> tuple[list[InteractionRecord], list[npt.NDArray[np.float64]]]:
        """Returns records and, per user, the family-affinity vector (reused for slates)."""
        cfg, rng = self.config, self.rng
        all_by_family = [np.flatnonzero(gates.family == f) for f in range(NUM_FAMILIES)]
        by_id = {p.item_id: p for p in products}
        records: list[InteractionRecord] = []
        affinities: list[npt.NDArray[np.float64]] = []
        self.fico_at_event = []

        for user in users:
            affinity = rng.dirichlet(np.ones(NUM_FAMILIES))
            affinities.append(affinity)
            elig_mask = gates.eligible(user.fico, user.dti, user.annual_income, user.state_index)
            elig_by_family = [
                np.flatnonzero(elig_mask & (gates.family == f)) for f in range(NUM_FAMILIES)
            ]

            n = int(rng.integers(cfg.min_history, cfg.max_history + 1))
            gaps = rng.exponential(cfg.mean_gap_days, size=n)
            gaps[0] = 0.0
            timestamps = np.round(np.cumsum(gaps), 3)
            cut = max(1, n - int(rng.integers(2, 6)))

            is_score_change = rng.random(n) < cfg.score_change_prob
            is_score_change[cut] = False  # the target must be a product event
            drifts = np.where(is_score_change, rng.integers(-25, 26, size=n), 0)
            # Walk FICO backwards from the profile value so the path ends at user.fico.
            fico_path = user.fico - (int(drifts.sum()) - np.cumsum(drifts))
            fico_path = np.clip(fico_path, 300, 850).astype(np.int64)
            self.fico_at_event.append(fico_path)

            action_roll = rng.random(n)
            families = rng.choice(NUM_FAMILIES, size=n, p=affinity)
            events: list[InteractionEvent] = []
            for t in range(n):
                ts = float(timestamps[t])
                if is_score_change[t]:
                    events.append(
                        InteractionEvent(
                            item_id=SCORE_CHANGE_ITEM_ID,
                            action=ActionType.SCORE_CHANGE,
                            timestamp=ts,
                        )
                    )
                    continue
                item = self._sample_item(int(families[t]), elig_by_family, all_by_family)
                r = action_roll[t]
                if r < cfg.apply_prob:
                    p = approve_probability(
                        int(fico_path[t]), user.dti, user.annual_income, user.state, by_id[item]
                    )
                    action = (
                        ActionType.APPLY_APPROVED if rng.random() < p else ActionType.APPLY_DECLINED
                    )
                elif r < cfg.apply_prob + cfg.credit_pull_prob:
                    action = ActionType.CREDIT_PULL
                else:
                    action = ActionType.VIEW
                events.append(InteractionEvent(item_id=item, action=action, timestamp=ts))

            target = events[cut]
            window = float(rng.uniform(*cfg.future_window_days))
            future = tuple(
                e
                for e in events[cut:]
                if e.action in POSITIVE_ACTIONS and e.timestamp <= target.timestamp + window
            )[: cfg.max_future]
            records.append(
                InteractionRecord(
                    user_index=user.user_index,
                    history=tuple(events[:cut]),
                    target=target,
                    future_window=future,
                )
            )
        return records, affinities

    # --------------------------------------------------------------------- slates
    def generate_slates(
        self,
        users: list[UserProfile],
        gates: _GateArrays,
        affinities: list[npt.NDArray[np.float64]],
    ) -> list[ImpressionSlate]:
        cfg, rng = self.config, self.rng
        slates: list[ImpressionSlate] = []
        slate_id = 0
        n_elig_target = cfg.slate_size - cfg.ineligible_per_slate
        for user, affinity in zip(users, affinities, strict=True):
            elig_mask = gates.eligible(user.fico, user.dti, user.annual_income, user.state_index)
            elig_ids = np.flatnonzero(elig_mask)
            inelig_ids = np.flatnonzero(~elig_mask)[1:]  # drop row 0
            for _ in range(cfg.slates_per_user):
                n_e = min(n_elig_target, elig_ids.size)
                n_i = min(cfg.slate_size - n_e, inelig_ids.size)
                cand = np.concatenate(
                    [
                        rng.choice(elig_ids, size=n_e, replace=False),
                        rng.choice(inelig_ids, size=n_i, replace=False),
                    ]
                ).astype(np.int64)
                rng.shuffle(cand)
                k = cand.size
                elig = elig_mask[cand]
                fam = gates.family[cand]
                margin = user.fico - gates.min_fico[cand]
                z_click = (
                    -2.5
                    + 2.0 * (affinity[fam] - 0.2)
                    + 8.0 * gates.reward[cand]
                    - 3.0 * gates.apr[cand]
                    + 0.4 * (margin > 0)
                    + rng.normal(0.0, 0.3, size=k)
                )
                p_click = 1.0 / (1.0 + np.exp(-z_click))
                z_apply = -1.0 + 0.002 * margin - 2.0 * user.dti + rng.normal(0.0, 0.3, size=k)
                p_apply = 1.0 / (1.0 + np.exp(-z_apply))
                p_approve = gates.approve_prob(user.fico, user.dti, cand, elig)

                y_click = (rng.random(k) < p_click).astype(np.int64)
                y_apply = y_click * (rng.random(k) < p_apply).astype(np.int64)
                y_approve = y_apply * (rng.random(k) < p_approve).astype(np.int64)
                mu = np.array(_FAMILY_LOG_AMOUNT_MU)[fam] + 0.002 * (user.fico - 600)
                amounts = y_approve * np.round(np.exp(mu + 0.5 * rng.normal(size=k)), 2)

                slates.append(
                    ImpressionSlate(
                        user_index=user.user_index,
                        slate_id=slate_id,
                        candidate_item_ids=tuple(int(i) for i in cand),
                        y_click=tuple(int(v) for v in y_click),
                        y_apply=tuple(int(v) for v in y_apply),
                        y_approve=tuple(int(v) for v in y_approve),
                        p_click=tuple(float(round(v, 6)) for v in p_click),
                        p_apply=tuple(float(round(v, 6)) for v in p_apply),
                        p_approve=tuple(float(round(v, 6)) for v in p_approve),
                        payouts=tuple(float(v) for v in gates.payout[cand]),
                        amounts=tuple(float(v) for v in amounts),
                        eligible=tuple(bool(v) for v in elig),
                    )
                )
                slate_id += 1
        return slates

    # ----------------------------------------------------------------- generate
    def generate(self) -> SyntheticDataset:
        products = self.generate_products()
        gates = _GateArrays.from_products(products)
        users = self.generate_users(gates)
        interactions, affinities = self.generate_interactions(users, products, gates)
        slates = self.generate_slates(users, gates, affinities)

        n = len(products)
        catalog = np.zeros((n + 1, PRODUCT_FEATURE_DIM), dtype=np.float32)
        for p in products:
            catalog[p.item_id] = p.to_feature_vector()
        catalog_mean = catalog[1:].mean(axis=0).astype(np.float32)
        catalog_std = (catalog[1:].std(axis=0) + 1e-6).astype(np.float32)
        user_feats = np.stack([u.to_feature_vector() for u in users]).astype(np.float32)
        if user_feats.ndim != 2:
            user_feats = user_feats.reshape(len(users), USER_FEATURE_DIM)

        counts = np.zeros(n + 1, dtype=np.int64)
        for r in interactions:
            for e in r.history:
                counts[e.item_id] += 1
            counts[r.target.item_id] += 1
        counts[0] = 0

        return SyntheticDataset(
            products=products,
            users=users,
            interactions=interactions,
            slates=slates,
            catalog_features=catalog,
            user_features=user_feats,
            item_counts=counts,
            catalog_mean=catalog_mean,
            catalog_std=catalog_std,
        )


# --------------------------------------------------------------------------- #
# Serialization (JSONL + npz, no pandas)
# --------------------------------------------------------------------------- #


def _dump_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json())
            f.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_dataset(
    ds: SyntheticDataset, out_dir: str | Path, config: GeneratorConfig | None = None
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _dump_jsonl(out / "products.jsonl", ds.products)
    _dump_jsonl(out / "users.jsonl", ds.users)
    _dump_jsonl(out / "interactions.jsonl", ds.interactions)
    _dump_jsonl(out / "slates.jsonl", ds.slates)
    np.savez(
        out / "arrays.npz",
        catalog_features=ds.catalog_features,
        user_features=ds.user_features,
        item_counts=ds.item_counts,
        catalog_mean=ds.catalog_mean,
        catalog_std=ds.catalog_std,
    )
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "num_products": ds.num_items,
        "num_users": ds.num_users,
        "num_interactions": len(ds.interactions),
        "num_slates": len(ds.slates),
        "config": asdict(config) if config is not None else None,
        "sha256": {name: _sha256(out / name) for name in ("products.jsonl", "users.jsonl")},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_dataset(in_dir: str | Path) -> SyntheticDataset:
    src = Path(in_dir)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported dataset format_version={meta.get('format_version')}")

    def _load(path: Path, model: Any) -> list[Any]:
        with path.open(encoding="utf-8") as f:
            return [model.model_validate_json(line) for line in f if line.strip()]

    arrays = np.load(src / "arrays.npz")
    return SyntheticDataset(
        products=_load(src / "products.jsonl", FinancialProduct),
        users=_load(src / "users.jsonl", UserProfile),
        interactions=_load(src / "interactions.jsonl", InteractionRecord),
        slates=_load(src / "slates.jsonl", ImpressionSlate),
        catalog_features=arrays["catalog_features"],
        user_features=arrays["user_features"],
        item_counts=arrays["item_counts"],
        catalog_mean=arrays["catalog_mean"],
        catalog_std=arrays["catalog_std"],
    )


def dataset_summary(ds: SyntheticDataset) -> dict[str, float]:
    """Headline statistics used by the CLI and the docs."""
    n_events = sum(len(r.history) for r in ds.interactions)
    actions = np.array(
        [int(e.action) for r in ds.interactions for e in r.history] or [0], dtype=np.int64
    )
    y_click = np.array([v for s in ds.slates for v in s.y_click], dtype=np.float64)
    y_apply = np.array([v for s in ds.slates for v in s.y_apply], dtype=np.float64)
    y_approve = np.array([v for s in ds.slates for v in s.y_approve], dtype=np.float64)
    ficos = np.array([u.fico for u in ds.users], dtype=np.float64)
    dtis = np.array([u.dti for u in ds.users], dtype=np.float64)
    tiers = {f.value: 0 for f in ProductFamily}
    for p in ds.products:
        tiers[p.family.value] += 1
    n_imp = max(len(y_click), 1)
    return {
        "num_products": float(ds.num_items),
        "num_users": float(ds.num_users),
        "num_history_events": float(n_events),
        "frac_score_change_events": float((actions == ActionType.SCORE_CHANGE).mean()),
        "frac_apply_declined_events": float((actions == ActionType.APPLY_DECLINED).mean()),
        "frac_apply_approved_events": float((actions == ActionType.APPLY_APPROVED).mean()),
        "impressions": float(n_imp),
        "ctr": float(y_click.sum() / n_imp),
        "apply_rate_given_click": float(y_apply.sum() / max(y_click.sum(), 1.0)),
        "approval_rate_given_apply": float(y_approve.sum() / max(y_apply.sum(), 1.0)),
        "approvals_per_impression": float(y_approve.sum() / n_imp),
        "corr_fico_dti": float(np.corrcoef(ficos, dtis)[0, 1]) if len(ficos) > 2 else 0.0,
        **{f"products_{k.lower()}": float(v) for k, v in tiers.items()},
    }
