"""Domain data contracts for the credit-marketplace recommender.

This module is the single source of truth for:

* the action vocabulary (:class:`ActionType`) and the reserved indices that
  every collator and model must respect,
* the product catalog contract (:class:`FinancialProduct`) with its *hard*
  declarative underwriting gates,
* the user contract (:class:`UserProfile`),
* the sequential training contract (:class:`InteractionRecord`), and
* the impression-space contract (:class:`ImpressionSlate`) used by the
  ranking / multi-task / re-ranking stages.

Reserved indices
----------------
``PAD_ITEM_ID == SCORE_CHANGE_ITEM_ID == 0``.  A ``SCORE_CHANGE`` event is a
user-level event (a credit score moved) with no product attached, so it
legitimately carries item id 0.  The *only* reliable way to tell a real
event from padding is therefore the action id: ``attention_mask = action_ids
!= PAD_ACTION_ID``.  Never derive a mask from ``item_ids != 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Reserved indices
# --------------------------------------------------------------------------- #

PAD_ITEM_ID: int = 0
SCORE_CHANGE_ITEM_ID: int = 0
PAD_ACTION_ID: int = 0


class ActionType(IntEnum):
    """Action vocabulary.  ``PAD`` is reserved for sequence padding only."""

    PAD = 0
    VIEW = 1
    SCORE_CHANGE = 2
    CREDIT_PULL = 3
    APPLY_APPROVED = 4
    APPLY_DECLINED = 5


NUM_ACTIONS: int = len(ActionType)

#: Actions that count as positive engagement with a product (used as retrieval
#: targets and PinnerFormer "future window" positives).
POSITIVE_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.VIEW, ActionType.CREDIT_PULL, ActionType.APPLY_APPROVED}
)


class ProductFamily(str, Enum):
    CREDIT_CARD = "CREDIT_CARD"
    BALANCE_TRANSFER_CARD = "BALANCE_TRANSFER_CARD"
    PERSONAL_LOAN = "PERSONAL_LOAN"
    AUTO_REFINANCE = "AUTO_REFINANCE"
    MORTGAGE = "MORTGAGE"


#: Fixed order used for one-hot encodings and ``family_ids`` tensors.
FAMILY_ORDER: tuple[ProductFamily, ...] = tuple(ProductFamily)
FAMILY_INDEX: dict[ProductFamily, int] = {f: i for i, f in enumerate(FAMILY_ORDER)}
NUM_FAMILIES: int = len(FAMILY_ORDER)


class CreditTier(str, Enum):
    DEEP_SUBPRIME = "DEEP_SUBPRIME"
    SUBPRIME = "SUBPRIME"
    NEAR_PRIME = "NEAR_PRIME"
    PRIME = "PRIME"
    SUPER_PRIME = "SUPER_PRIME"

    @classmethod
    def from_fico(cls, fico: int) -> CreditTier:
        if fico < 580:
            return cls.DEEP_SUBPRIME
        if fico < 620:
            return cls.SUBPRIME
        if fico < 680:
            return cls.NEAR_PRIME
        if fico < 740:
            return cls.PRIME
        return cls.SUPER_PRIME


TIER_ORDER: tuple[CreditTier, ...] = tuple(CreditTier)
TIER_INDEX: dict[CreditTier, int] = {t: i for i, t in enumerate(TIER_ORDER)}
NUM_TIERS: int = len(TIER_ORDER)
#: Lowest FICO that maps to each tier (index-aligned with ``TIER_ORDER``).
TIER_FICO_FLOORS: tuple[int, ...] = (300, 580, 620, 680, 740)

US_STATES: tuple[str, ...] = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
    "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
)  # fmt: skip
STATE_INDEX: dict[str, int] = {s: i for i, s in enumerate(US_STATES)}
NUM_STATES: int = len(US_STATES)

#: Feature-vector widths (kept as constants so collators can preallocate).
PRODUCT_FEATURE_DIM: int = 8 + NUM_FAMILIES
USER_FEATURE_DIM: int = 4 + NUM_TIERS

_FROZEN = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


class FinancialProduct(BaseModel):
    """A partner product with hard underwriting gates and economics.

    The four gate fields are *declarative*: a user who fails any one of them is
    ineligible and must never be shown the product, regardless of model score.
    """

    model_config = _FROZEN

    item_id: int = Field(ge=1, description="1-based; 0 is reserved for PAD / SCORE_CHANGE")
    name: str
    family: ProductFamily
    # --- hard underwriting gates -------------------------------------------------
    min_fico: int = Field(ge=300, le=850)
    max_dti: float = Field(gt=0.0, le=1.0)
    min_annual_income: float = Field(ge=0.0)
    licensed_states: frozenset[str]
    # --- economics ---------------------------------------------------------------
    apr: float = Field(ge=0.0, le=1.0)
    annual_fee: float = Field(ge=0.0)
    reward_rate: float = Field(ge=0.0, le=1.0)
    term_months: int = Field(ge=0)
    partner_payout: float = Field(gt=0.0, description="Revenue R_i on a funded approval")

    @model_validator(mode="after")
    def _validate_states(self) -> FinancialProduct:
        unknown = self.licensed_states - set(US_STATES)
        if unknown:
            raise ValueError(f"unknown state codes: {sorted(unknown)}")
        return self

    @property
    def required_tier(self) -> CreditTier:
        return CreditTier.from_fico(self.min_fico)

    @property
    def family_index(self) -> int:
        return FAMILY_INDEX[self.family]

    def to_feature_vector(self) -> npt.NDArray[np.float32]:
        vec = np.zeros(PRODUCT_FEATURE_DIM, dtype=np.float32)
        vec[0] = self.min_fico / 850.0
        vec[1] = self.max_dti
        vec[2] = math.log1p(self.min_annual_income) / 13.0
        vec[3] = self.apr
        vec[4] = self.annual_fee / 1000.0
        vec[5] = self.reward_rate
        vec[6] = self.term_months / 360.0
        vec[7] = math.log1p(self.partner_payout) / 8.0
        vec[8 + self.family_index] = 1.0
        return vec


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #


class UserProfile(BaseModel):
    model_config = _FROZEN

    user_index: int = Field(ge=0)
    fico: int = Field(ge=300, le=850)
    dti: float = Field(ge=0.0, le=1.0)
    annual_income: float = Field(ge=0.0)
    state: str
    held_product_ids: tuple[int, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> UserProfile:
        if self.state not in STATE_INDEX:
            raise ValueError(f"unknown state code: {self.state}")
        if any(pid < 1 for pid in self.held_product_ids):
            raise ValueError("held_product_ids must be >= 1")
        return self

    @property
    def tier(self) -> CreditTier:
        return CreditTier.from_fico(self.fico)

    @property
    def tier_index(self) -> int:
        return TIER_INDEX[self.tier]

    @property
    def state_index(self) -> int:
        return STATE_INDEX[self.state]

    def to_feature_vector(self) -> npt.NDArray[np.float32]:
        vec = np.zeros(USER_FEATURE_DIM, dtype=np.float32)
        vec[0] = self.fico / 850.0
        vec[1] = self.dti
        vec[2] = math.log1p(self.annual_income) / 13.0
        vec[3] = min(len(self.held_product_ids), 5) / 5.0
        vec[4 + self.tier_index] = 1.0
        return vec


# --------------------------------------------------------------------------- #
# Sequential interactions
# --------------------------------------------------------------------------- #


class InteractionEvent(BaseModel):
    """One event in a user's timeline.  ``timestamp`` is in days."""

    model_config = _FROZEN

    item_id: int = Field(ge=0)
    action: ActionType
    timestamp: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate(self) -> InteractionEvent:
        if self.action == ActionType.PAD:
            raise ValueError("PAD is reserved for padding and cannot be a real event")
        is_score_change = self.action == ActionType.SCORE_CHANGE
        if is_score_change != (self.item_id == SCORE_CHANGE_ITEM_ID):
            raise ValueError(
                "item_id must be 0 if and only if action is SCORE_CHANGE "
                f"(got item_id={self.item_id}, action={self.action.name})"
            )
        return self


class InteractionRecord(BaseModel):
    """History -> target -> future window, all in chronological order."""

    model_config = _FROZEN

    user_index: int = Field(ge=0)
    history: tuple[InteractionEvent, ...] = Field(min_length=1)
    target: InteractionEvent
    future_window: tuple[InteractionEvent, ...] = ()

    @model_validator(mode="after")
    def _validate(self) -> InteractionRecord:
        if self.target.item_id == SCORE_CHANGE_ITEM_ID:
            raise ValueError("target must be a product event (item_id > 0)")
        ts = [e.timestamp for e in self.history]
        if any(b < a for a, b in zip(ts, ts[1:], strict=False)):
            raise ValueError("history timestamps must be non-decreasing")
        if self.target.timestamp < ts[-1]:
            raise ValueError("target must not precede the last history event")
        fts = [e.timestamp for e in self.future_window]
        if any(t < self.target.timestamp for t in fts):
            raise ValueError("future_window events must not precede the target")
        return self


# --------------------------------------------------------------------------- #
# Impression space
# --------------------------------------------------------------------------- #


class ImpressionSlate(BaseModel):
    """A served slate with binary funnel labels and the generator's true probabilities.

    Funnel semantics: ``y_apply`` is only observable when ``y_click == 1`` and
    ``y_approve`` only when ``y_apply == 1``.  ``p_apply`` and ``p_approve`` are
    the *conditional* probabilities p(apply | click) and p(approve | apply).
    """

    model_config = _FROZEN

    user_index: int = Field(ge=0)
    slate_id: int = Field(ge=0)
    candidate_item_ids: tuple[int, ...] = Field(min_length=1)
    y_click: tuple[int, ...]
    y_apply: tuple[int, ...]
    y_approve: tuple[int, ...]
    p_click: tuple[float, ...]
    p_apply: tuple[float, ...]
    p_approve: tuple[float, ...]
    payouts: tuple[float, ...]
    amounts: tuple[float, ...]
    eligible: tuple[bool, ...]

    @property
    def size(self) -> int:
        return len(self.candidate_item_ids)

    @model_validator(mode="after")
    def _validate(self) -> ImpressionSlate:
        k = self.size
        fields = (
            self.y_click, self.y_apply, self.y_approve, self.p_click, self.p_apply,
            self.p_approve, self.payouts, self.amounts, self.eligible,
        )  # fmt: skip
        if any(len(f) != k for f in fields):
            raise ValueError("all per-candidate fields must have the same length")
        if any(i < 1 for i in self.candidate_item_ids):
            raise ValueError("candidate_item_ids must be >= 1")
        for i in range(k):
            yc, ya, yp = self.y_click[i], self.y_apply[i], self.y_approve[i]
            if yc not in (0, 1) or ya not in (0, 1) or yp not in (0, 1):
                raise ValueError("labels must be binary")
            if ya > yc or yp > ya:
                raise ValueError("funnel labels must satisfy y_approve <= y_apply <= y_click")
            for p in (self.p_click[i], self.p_apply[i], self.p_approve[i]):
                if not 0.0 <= p <= 1.0:
                    raise ValueError("probabilities must be in [0, 1]")
            if (self.amounts[i] > 0.0) != (yp == 1):
                raise ValueError("amount must be > 0 if and only if the application was approved")
            if not self.eligible[i] and self.p_approve[i] != 0.0:
                raise ValueError("ineligible candidates must have p_approve == 0")
        return self


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #


@dataclass
class SyntheticDataset:
    """In-memory dataset produced by the generator (plain dataclass, no validation)."""

    products: list[FinancialProduct]
    users: list[UserProfile]
    interactions: list[InteractionRecord]
    slates: list[ImpressionSlate]
    catalog_features: npt.NDArray[np.float32]  # (N+1, PRODUCT_FEATURE_DIM), row 0 zero
    user_features: npt.NDArray[np.float32]  # (U, USER_FEATURE_DIM)
    item_counts: npt.NDArray[np.int64]  # (N+1,) history frequency for logQ
    catalog_mean: npt.NDArray[np.float32] = field(default_factory=lambda: np.zeros(0, np.float32))
    catalog_std: npt.NDArray[np.float32] = field(default_factory=lambda: np.ones(0, np.float32))

    @property
    def num_items(self) -> int:
        return len(self.products)

    @property
    def num_users(self) -> int:
        return len(self.users)

    def family_by_item(self) -> npt.NDArray[np.int64]:
        """(N+1,) family index per item id; row 0 is -1."""
        out = np.full(self.num_items + 1, -1, dtype=np.int64)
        for p in self.products:
            out[p.item_id] = p.family_index
        return out

    def required_tier_by_item(self) -> npt.NDArray[np.int64]:
        """(N+1,) required tier index per item id; row 0 is NUM_TIERS (never eligible)."""
        out = np.full(self.num_items + 1, NUM_TIERS, dtype=np.int64)
        for p in self.products:
            out[p.item_id] = TIER_INDEX[p.required_tier]
        return out
