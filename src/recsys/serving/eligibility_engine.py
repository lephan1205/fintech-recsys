"""Vectorized underwriting eligibility gate (with held / pending product rules).

The engine compiles the catalog's declarative gates into dense NumPy arrays indexed
by ``item_id`` (row 0 is never eligible) and evaluates a whole ``(users x products)``
boolean matrix with four broadcast comparisons, then removes

* products the user already **holds** (always),
* products the user has an **open application** for (``pending_product_ids``, always;
  D4 item-level rule), and
* whole families under the **pending-family policy** (D4 family-level rule): by
  default one open refinance at a time (``MORTGAGE`` / ``AUTO_REFINANCE`` are masked
  while pending) whereas a second card / personal-loan application is legitimate and is
  only *penalized* at valuation time (see ``valuation/utility.py``).

It is used three times in the cascade: (1) as the per-request mask fed to the TIGER
prefix trie, (2) as the post-retrieval vectorized gate, and (3) in the final slate
assertion (``ComplianceViolation`` if anything slips through).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from recsys.data.schema import (
    FAMILY_ORDER,
    NUM_FAMILIES,
    NUM_STATES,
    NUM_TIERS,
    STATE_INDEX,
    TIER_INDEX,
    FinancialProduct,
    ProductFamily,
    UserProfile,
)

GATE_NAMES: tuple[str, ...] = ("min_fico", "max_dti", "min_annual_income", "licensed_state")
CONTEXT_RULES: tuple[str, ...] = ("not_held", "not_pending", "pending_family_policy")

PendingFamilyRule = Literal["mask", "penalize", "ignore"]
PendingFamilyPolicy = Mapping[ProductFamily, PendingFamilyRule]

DEFAULT_PENDING_FAMILY_POLICY: dict[ProductFamily, PendingFamilyRule] = {
    ProductFamily.CREDIT_CARD: "penalize",
    ProductFamily.BALANCE_TRANSFER_CARD: "penalize",
    ProductFamily.PERSONAL_LOAN: "penalize",
    ProductFamily.AUTO_REFINANCE: "mask",
    ProductFamily.MORTGAGE: "mask",
}


def policy_vector(policy: PendingFamilyPolicy, rule: PendingFamilyRule) -> npt.NDArray[np.bool_]:
    """``(NUM_FAMILIES,)`` bool: families whose rule equals ``rule``."""
    return np.array([policy.get(f, "ignore") == rule for f in FAMILY_ORDER], dtype=bool)


class ComplianceViolation(RuntimeError):
    """Raised when an ineligible product reaches a user-facing surface."""


@dataclass
class ProductGateArrays:
    min_fico: npt.NDArray[np.int64]  # (N+1,)
    max_dti: npt.NDArray[np.float64]  # (N+1,)
    min_income: npt.NDArray[np.float64]  # (N+1,)
    licensed: npt.NDArray[np.bool_]  # (N+1, NUM_STATES)
    family: npt.NDArray[np.int64]  # (N+1,), -1 for row 0
    required_tier: npt.NDArray[np.int64]  # (N+1,), NUM_TIERS for row 0
    payout: npt.NDArray[np.float64]  # (N+1,)

    @classmethod
    def from_products(cls, products: Sequence[FinancialProduct]) -> ProductGateArrays:
        n = max((p.item_id for p in products), default=0) + 1
        g = cls(
            min_fico=np.full(n, 10_000, dtype=np.int64),
            max_dti=np.full(n, -1.0, dtype=np.float64),
            min_income=np.full(n, np.inf, dtype=np.float64),
            licensed=np.zeros((n, NUM_STATES), dtype=bool),
            family=np.full(n, -1, dtype=np.int64),
            required_tier=np.full(n, NUM_TIERS, dtype=np.int64),
            payout=np.zeros(n, dtype=np.float64),
        )
        for p in products:
            i = p.item_id
            g.min_fico[i] = p.min_fico
            g.max_dti[i] = p.max_dti
            g.min_income[i] = p.min_annual_income
            for s in p.licensed_states:
                g.licensed[i, STATE_INDEX[s]] = True
            g.family[i] = p.family_index
            g.required_tier[i] = TIER_INDEX[p.required_tier]
            g.payout[i] = p.partner_payout
        return g

    @property
    def num_items(self) -> int:
        return int(self.min_fico.shape[0]) - 1


@dataclass
class UserGateArrays:
    """Per-user inputs of :meth:`EligibilityEngine.mask`, built from profiles."""

    fico: npt.NDArray[np.int64]  # (U,)
    dti: npt.NDArray[np.float64]  # (U,)
    annual_income: npt.NDArray[np.float64]  # (U,)
    state_idx: npt.NDArray[np.int64]  # (U,)
    held: npt.NDArray[np.bool_]  # (U, N+1)
    pending: npt.NDArray[np.bool_]  # (U, N+1)
    pending_families: npt.NDArray[np.bool_]  # (U, NUM_FAMILIES)

    @classmethod
    def from_users(cls, users: Sequence[UserProfile], num_items: int) -> UserGateArrays:
        u = len(users)
        held = np.zeros((u, num_items + 1), dtype=bool)
        pending = np.zeros((u, num_items + 1), dtype=bool)
        fams = np.zeros((u, NUM_FAMILIES), dtype=bool)
        for i, user in enumerate(users):
            if user.held_product_ids:
                held[i, list(user.held_product_ids)] = True
            if user.pending_product_ids:
                pending[i, list(user.pending_product_ids)] = True
            if user.pending_family_ids:
                fams[i, list(user.pending_family_ids)] = True
        return cls(
            fico=np.array([x.fico for x in users], dtype=np.int64),
            dti=np.array([x.dti for x in users], dtype=np.float64),
            annual_income=np.array([x.annual_income for x in users], dtype=np.float64),
            state_idx=np.array([x.state_index for x in users], dtype=np.int64),
            held=held,
            pending=pending,
            pending_families=fams,
        )


class EligibilityEngine:
    def __init__(
        self,
        gates: ProductGateArrays,
        pending_family_policy: PendingFamilyPolicy | None = None,
    ) -> None:
        self.gates = gates
        self.policy: dict[ProductFamily, PendingFamilyRule] = dict(
            DEFAULT_PENDING_FAMILY_POLICY
            if pending_family_policy is None
            else pending_family_policy
        )
        self._masked_families = policy_vector(self.policy, "mask")

    @classmethod
    def from_products(
        cls,
        products: Sequence[FinancialProduct],
        pending_family_policy: PendingFamilyPolicy | None = None,
    ) -> EligibilityEngine:
        return cls(ProductGateArrays.from_products(products), pending_family_policy)

    @property
    def num_items(self) -> int:
        return self.gates.num_items

    # ---------------------------------------------------------------- batch API
    def hard_gate_mask(
        self,
        fico: npt.NDArray[np.int64],
        dti: npt.NDArray[np.float64],
        annual_income: npt.NDArray[np.float64],
        state_idx: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.bool_]:
        """``(U,)`` user arrays -> ``(U, N+1)`` hard underwriting gates only."""
        g = self.gates
        out: npt.NDArray[np.bool_] = (
            (g.min_fico[None, :] <= fico[:, None])
            & (g.max_dti[None, :] >= dti[:, None])
            & (g.min_income[None, :] <= annual_income[:, None])
            & g.licensed.T[state_idx]  # (U, N+1)
        )
        out[:, 0] = False
        return out

    def mask(
        self,
        fico: npt.NDArray[np.int64],
        dti: npt.NDArray[np.float64],
        annual_income: npt.NDArray[np.float64],
        state_idx: npt.NDArray[np.int64],
        held: npt.NDArray[np.bool_] | None = None,
        pending: npt.NDArray[np.bool_] | None = None,
        pending_families: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """``(U,)`` user arrays -> ``(U, N+1)`` serving eligibility.

        ``held`` / ``pending`` are ``(U, N+1)`` product matrices (always excluded);
        ``pending_families`` is ``(U, NUM_FAMILIES)`` and excludes every product of a
        pending family whose policy rule is ``"mask"``.
        """
        out = self.hard_gate_mask(fico, dti, annual_income, state_idx)
        if held is not None:
            out &= ~held
        if pending is not None:
            out &= ~pending
        if pending_families is not None:
            fam = np.clip(self.gates.family, 0, NUM_FAMILIES - 1)  # row 0 handled below
            blocked = (pending_families & self._masked_families[None, :])[:, fam]  # (U, N+1)
            out &= ~blocked
        out[:, 0] = False
        return out

    def mask_for_users(self, users: Sequence[UserProfile]) -> npt.NDArray[np.bool_]:
        """``(U, N+1)`` serving eligibility for a list of profiles (all rules applied)."""
        a = UserGateArrays.from_users(users, self.num_items)
        return self.mask(
            a.fico, a.dti, a.annual_income, a.state_idx, a.held, a.pending, a.pending_families
        )

    # ----------------------------------------------------------- single-user API
    def mask_for_user(
        self,
        user: UserProfile,
        exclude_held: bool = True,
        exclude_pending: bool = True,
        apply_family_policy: bool = True,
    ) -> npt.NDArray[np.bool_]:
        a = UserGateArrays.from_users([user], self.num_items)
        m = self.mask(
            a.fico,
            a.dti,
            a.annual_income,
            a.state_idx,
            a.held if exclude_held else None,
            a.pending if exclude_pending else None,
            a.pending_families if apply_family_policy else None,
        )
        row: npt.NDArray[np.bool_] = m[0]
        return row

    def filter_candidates(
        self, user: UserProfile, candidate_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Post-retrieval gate: keep only serving-eligible candidates (order preserved)."""
        m = self.mask_for_user(user)
        valid = (candidate_ids > 0) & (candidate_ids <= self.num_items)
        keep = np.zeros_like(valid)
        keep[valid] = m[candidate_ids[valid]]
        out: npt.NDArray[np.int64] = candidate_ids[keep]
        return out

    def explain(self, user: UserProfile, item_id: int) -> dict[str, bool]:
        """Per-rule pass/fail for adverse-action style reasoning."""
        g = self.gates
        fam = int(g.family[item_id])
        family_rule = self.policy.get(FAMILY_ORDER[fam], "ignore") if fam >= 0 else "ignore"
        family_blocked = family_rule == "mask" and fam in user.pending_family_ids
        return {
            "min_fico": bool(g.min_fico[item_id] <= user.fico),
            "max_dti": bool(g.max_dti[item_id] >= user.dti),
            "min_annual_income": bool(g.min_income[item_id] <= user.annual_income),
            "licensed_state": bool(g.licensed[item_id, user.state_index]),
            "not_held": item_id not in user.held_product_ids,
            "not_pending": item_id not in user.pending_product_ids,
            "pending_family_policy": not family_blocked,
        }

    def assert_all_eligible(self, user: UserProfile, item_ids: Sequence[int]) -> None:
        """Final enforcement point: every served id must pass *all* rules."""
        m = self.mask_for_user(user)
        bad = [int(i) for i in item_ids if i <= 0 or i > self.num_items or not m[i]]
        if bad:
            raise ComplianceViolation(f"ineligible items served to user {user.user_index}: {bad}")
