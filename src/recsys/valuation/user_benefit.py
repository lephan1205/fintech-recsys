"""Net user benefit ``NB`` per product family (D5), pure NumPy, vectorized over ``(B, K)``.

``NB`` is a *deterministic, auditable* function of product economics and the user's
financial state — it is never learned and never enters a training loss.  A learned or
loss-embedded version would (i) entangle a business weight with the behavioural
probabilities, (ii) bake the trade-off into model parameters so changing it means
retraining, and (iii) make "why was this recommended?" unanswerable, whereas these
formulas yield "this saves you $X".

Formulas (horizon ``H`` years, default 2; loans use ``E[amount]`` from ZILN capped by the
debt being refinanced)::

    BALANCE_TRANSFER_CARD  B · r_rev · m_intro/12 − f_bt · B − annual_fee · min(H, m_intro/12)
                           B = min(revolving_balance, E[amount])
    CREDIT_CARD            reward_rate · annual_spend · H + signup_bonus − annual_fee · H
    PERSONAL_LOAN          (r_other − apr) · A · (term/12) / 2 − f_orig · A
                           A = min(E[amount], other_debt_balance)
    AUTO_REFINANCE         Σ payment difference over n = min(remaining, term)
                           − (balance still owed after n months: new − old)
                           − f_orig · balance
    MORTGAGE (refi)        PV of payment difference over n = min(remaining, 12·H_hold)
                           − PV(balance still owed after n months: new − old)
                           − closing_costs − f_orig · balance,  H_hold = 7 years

The "balance still owed" term keeps refinance comparisons honest: stretching a loan into
a longer term lowers the payment but leaves more principal outstanding at the end of the
window, so payment relief alone would flatter a costlier loan.

minus the **hard-pull cost** ``c_pull`` ($15 equivalent), doubled when
``recent_hard_pulls_30d >= 2`` (application fatigue).  Refinance products are only
sensible when ``apr < current_rate`` and a loan exists (``refinance_ok``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from recsys.data.schema import FAMILY_INDEX, FinancialProduct, ProductFamily, UserProfile

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FAM_CC = FAMILY_INDEX[ProductFamily.CREDIT_CARD]
FAM_BT = FAMILY_INDEX[ProductFamily.BALANCE_TRANSFER_CARD]
FAM_PL = FAMILY_INDEX[ProductFamily.PERSONAL_LOAN]
FAM_AUTO = FAMILY_INDEX[ProductFamily.AUTO_REFINANCE]
FAM_MORT = FAMILY_INDEX[ProductFamily.MORTGAGE]


@dataclass(frozen=True)
class UserBenefitConfig:
    horizon_years: float = 2.0  # H
    hold_years_mortgage: float = 7.0  # H_hold
    hard_pull_cost: float = 15.0  # c_pull, dollars
    fatigue_pulls: int = 2  # recent_hard_pulls_30d >= this doubles the cost
    fatigue_multiplier: float = 2.0


@dataclass
class ProductEconomics:
    """Catalog economics indexed by ``item_id`` (row 0 = zeros)."""

    family: IntArray
    apr: FloatArray
    annual_fee: FloatArray
    reward_rate: FloatArray
    term_months: FloatArray
    intro_apr_months: FloatArray
    balance_transfer_fee_rate: FloatArray
    origination_fee_rate: FloatArray
    signup_bonus_value: FloatArray
    closing_costs: FloatArray
    partner_payout: FloatArray

    @classmethod
    def from_products(cls, products: Sequence[FinancialProduct]) -> ProductEconomics:
        n = max((p.item_id for p in products), default=0) + 1
        z = np.zeros(n)
        econ = cls(
            family=np.full(n, -1, dtype=np.int64),
            apr=z.copy(),
            annual_fee=z.copy(),
            reward_rate=z.copy(),
            term_months=z.copy(),
            intro_apr_months=z.copy(),
            balance_transfer_fee_rate=z.copy(),
            origination_fee_rate=z.copy(),
            signup_bonus_value=z.copy(),
            closing_costs=z.copy(),
            partner_payout=z.copy(),
        )
        for p in products:
            i = p.item_id
            econ.family[i] = p.family_index
            econ.apr[i] = p.apr
            econ.annual_fee[i] = p.annual_fee
            econ.reward_rate[i] = p.reward_rate
            econ.term_months[i] = p.term_months
            econ.intro_apr_months[i] = p.intro_apr_months
            econ.balance_transfer_fee_rate[i] = p.balance_transfer_fee_rate
            econ.origination_fee_rate[i] = p.origination_fee_rate
            econ.signup_bonus_value[i] = p.signup_bonus_value
            econ.closing_costs[i] = p.closing_costs
            econ.partner_payout[i] = p.partner_payout
        return econ


@dataclass
class UserFinancialState:
    """Per-user financial state, row index = ``user_index``."""

    revolving_balance: FloatArray
    revolving_apr: FloatArray
    annual_card_spend: FloatArray
    other_debt_balance: FloatArray
    other_debt_apr: FloatArray
    auto_loan_balance: FloatArray
    auto_loan_rate: FloatArray
    auto_remaining_months: FloatArray
    mortgage_balance: FloatArray
    mortgage_rate: FloatArray
    mortgage_remaining_months: FloatArray
    recent_hard_pulls_30d: IntArray

    @classmethod
    def from_users(cls, users: Sequence[UserProfile]) -> UserFinancialState:
        def f(name: str) -> FloatArray:
            return np.array([float(getattr(u, name)) for u in users], dtype=np.float64)

        return cls(
            revolving_balance=f("revolving_balance"),
            revolving_apr=f("revolving_apr"),
            annual_card_spend=f("annual_card_spend"),
            other_debt_balance=f("other_debt_balance"),
            other_debt_apr=f("other_debt_apr"),
            auto_loan_balance=f("auto_loan_balance"),
            auto_loan_rate=f("auto_loan_rate"),
            auto_remaining_months=f("auto_remaining_months"),
            mortgage_balance=f("mortgage_balance"),
            mortgage_rate=f("mortgage_rate"),
            mortgage_remaining_months=f("mortgage_remaining_months"),
            recent_hard_pulls_30d=np.array(
                [u.recent_hard_pulls_30d for u in users], dtype=np.int64
            ),
        )


def monthly_payment(
    principal: FloatArray, annual_rate: FloatArray, months: FloatArray
) -> FloatArray:
    """Amortized payment; ``0`` when ``months == 0``; ``principal / months`` at zero rate."""
    r = annual_rate / 12.0
    n = np.maximum(months, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        amort = principal * r / (1.0 - (1.0 + r) ** (-n))
        flat = np.where(n > 0, principal / np.maximum(n, 1.0), 0.0)
    out: FloatArray = np.where(n <= 0, 0.0, np.where(r > 0, amort, flat))
    return out


def remaining_balance(
    principal: FloatArray,
    annual_rate: FloatArray,
    total_months: FloatArray,
    months_paid: FloatArray,
) -> FloatArray:
    """Principal still owed after ``months_paid`` of an amortizing loan (0 when paid off)."""
    r = annual_rate / 12.0
    n = np.maximum(total_months, 0.0)
    k = np.clip(months_paid, 0.0, n)
    pmt = monthly_payment(principal, annual_rate, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (1.0 + r) ** k
        amort = principal * growth - pmt * (growth - 1.0) / r
        flat = principal * (1.0 - k / np.maximum(n, 1.0))
    out: FloatArray = np.where(n <= 0, 0.0, np.where(r > 0, amort, flat))
    return np.maximum(out, 0.0)


def present_value_of_annuity(
    payment: FloatArray, annual_rate: FloatArray, months: FloatArray
) -> FloatArray:
    """PV of ``payment`` per month for ``months`` at monthly rate ``annual_rate / 12``."""
    r = annual_rate / 12.0
    n = np.maximum(months, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pv = payment * (1.0 - (1.0 + r) ** (-n)) / r
    out: FloatArray = np.where(r > 0, pv, payment * n)
    return out


@dataclass
class NetBenefit:
    nb: FloatArray  # (B, K) net user benefit in dollars (gross - hard-pull cost)
    gross: FloatArray  # (B, K) before the hard-pull cost
    hard_pull_cost: FloatArray  # (B, K)
    refinance_ok: FloatArray  # (B, K) bool as float: refinance products only if apr < current rate


def net_user_benefit(
    item_ids: npt.ArrayLike,
    user_indices: npt.ArrayLike,
    expected_amount: npt.ArrayLike,
    econ: ProductEconomics,
    state: UserFinancialState,
    config: UserBenefitConfig | None = None,
) -> NetBenefit:
    """``item_ids (B, K)``, ``user_indices (B,)``, ``expected_amount (B, K)`` -> ``NetBenefit``.

    Row-0 (empty slot) items get ``0`` gross benefit.
    """
    cfg = config or UserBenefitConfig()
    ids = np.asarray(item_ids, dtype=np.int64)
    uidx = np.asarray(user_indices, dtype=np.int64)
    amt = np.asarray(expected_amount, dtype=np.float64)
    if ids.ndim != 2 or amt.shape != ids.shape or uidx.shape != (ids.shape[0],):
        raise ValueError("item_ids / expected_amount must be (B, K) and user_indices (B,)")
    fam = econ.family[ids]
    h = cfg.horizon_years

    def u(arr: FloatArray) -> FloatArray:
        """Per-user column ``(B, 1)`` for broadcasting against ``(B, K)``."""
        col: FloatArray = arr[uidx][:, None]
        return col

    # balance-transfer card
    intro_years = econ.intro_apr_months[ids] / 12.0
    b_bt = np.minimum(u(state.revolving_balance), amt)
    bt = (
        b_bt * u(state.revolving_apr) * intro_years
        - econ.balance_transfer_fee_rate[ids] * b_bt
        - econ.annual_fee[ids] * np.minimum(h, intro_years)
    )
    # credit card
    cc = (
        econ.reward_rate[ids] * u(state.annual_card_spend) * h
        + econ.signup_bonus_value[ids]
        - econ.annual_fee[ids] * h
    )
    # personal loan (consolidation)
    a_pl = np.minimum(amt, u(state.other_debt_balance))
    pl = (u(state.other_debt_apr) - econ.apr[ids]) * a_pl * (econ.term_months[ids] / 12.0) / 2.0
    pl = pl - econ.origination_fee_rate[ids] * a_pl
    # auto refinance
    auto_bal = u(state.auto_loan_balance)
    auto_n = np.minimum(u(state.auto_remaining_months), econ.term_months[ids])
    old_auto = monthly_payment(auto_bal, u(state.auto_loan_rate), u(state.auto_remaining_months))
    new_auto = monthly_payment(auto_bal, econ.apr[ids], econ.term_months[ids])
    owed_old_auto = remaining_balance(
        auto_bal, u(state.auto_loan_rate), u(state.auto_remaining_months), auto_n
    )
    owed_new_auto = remaining_balance(auto_bal, econ.apr[ids], econ.term_months[ids], auto_n)
    auto = (
        (old_auto - new_auto) * auto_n
        - (owed_new_auto - owed_old_auto)
        - econ.origination_fee_rate[ids] * auto_bal
    )
    # mortgage refinance
    m_bal = u(state.mortgage_balance)
    m_n = np.minimum(u(state.mortgage_remaining_months), 12.0 * cfg.hold_years_mortgage)
    old_m = monthly_payment(m_bal, u(state.mortgage_rate), u(state.mortgage_remaining_months))
    new_m = monthly_payment(m_bal, econ.apr[ids], econ.term_months[ids])
    owed_old_m = remaining_balance(
        m_bal, u(state.mortgage_rate), u(state.mortgage_remaining_months), m_n
    )
    owed_new_m = remaining_balance(m_bal, econ.apr[ids], econ.term_months[ids], m_n)
    discount = (1.0 + econ.apr[ids] / 12.0) ** (-m_n)
    mort = (
        present_value_of_annuity(old_m - new_m, econ.apr[ids], m_n)
        - (owed_new_m - owed_old_m) * discount
        - econ.closing_costs[ids]
        - econ.origination_fee_rate[ids] * m_bal
    )

    gross = np.zeros_like(amt)
    gross = np.where(fam == FAM_BT, bt, gross)
    gross = np.where(fam == FAM_CC, cc, gross)
    gross = np.where(fam == FAM_PL, pl, gross)
    gross = np.where(fam == FAM_AUTO, auto, gross)
    gross = np.where(fam == FAM_MORT, mort, gross)
    gross = np.where(ids > 0, gross, 0.0)

    fatigue = (state.recent_hard_pulls_30d[uidx] >= cfg.fatigue_pulls)[:, None]
    pull = np.where(fatigue, cfg.fatigue_multiplier, 1.0) * cfg.hard_pull_cost * (ids > 0)
    refi_auto = (fam == FAM_AUTO) & ((auto_bal <= 0) | (econ.apr[ids] >= u(state.auto_loan_rate)))
    refi_mort = (fam == FAM_MORT) & ((m_bal <= 0) | (econ.apr[ids] >= u(state.mortgage_rate)))
    refinance_ok = ~(refi_auto | refi_mort)
    return NetBenefit(
        nb=gross - pull,
        gross=gross,
        hard_pull_cost=pull,
        refinance_ok=refinance_ok.astype(np.float64),
    )
