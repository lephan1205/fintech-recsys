"""The Stage-4 valuation step shared by serving, PRM training and evaluation.

    raw logits -> calibrate -> P_funded / EV / E[amount] -> NB -> U + guardrails

One implementation so training data for PRM, the offline tables and the live pipeline
cannot drift apart.  Everything after calibration is NumPy (dollars, auditable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from recsys.data.impression_collator import ImpressionBatch
from recsys.data.schema import NUM_FAMILIES
from recsys.models.ranker import RankerOutput
from recsys.serving.calibration import CalibratorSet
from recsys.valuation.expected_value import expected_amount, expected_value, p_funded
from recsys.valuation.user_benefit import (
    ProductEconomics,
    UserBenefitConfig,
    UserFinancialState,
    net_user_benefit,
)
from recsys.valuation.utility import GuardrailResult, UtilityConfig, apply_guardrails

FloatArray = npt.NDArray[np.float64]


@dataclass
class ValuedCandidates:
    p1: FloatArray  # (B, K) calibrated p(click)
    p2: FloatArray  # (B, K) calibrated p(apply | click)
    p3: FloatArray  # (B, K) calibrated p(approve | apply)
    p_funded: FloatArray  # (B, K)
    ev: FloatArray  # (B, K) expected partner revenue, $
    expected_amount: FloatArray  # (B, K) ZILN E[amount], $
    nb: FloatArray  # (B, K) net user benefit after the pending-family penalty, $
    utility: FloatArray  # (B, K) U after guardrails (-inf on excluded slots)
    keep: npt.NDArray[np.bool_]  # (B, K) guardrail survivors
    refinance_ok: npt.NDArray[np.bool_]  # (B, K)
    guardrails: GuardrailResult
    raw: RankerOutput


class ValuationStage:
    def __init__(
        self,
        econ: ProductEconomics,
        state: UserFinancialState,
        penalize_families: npt.NDArray[np.bool_],
        user_benefit: UserBenefitConfig | None = None,
        utility: UtilityConfig | None = None,
    ) -> None:
        if penalize_families.shape != (NUM_FAMILIES,):
            raise ValueError("penalize_families must be (NUM_FAMILIES,)")
        self.econ = econ
        self.state = state
        self.penalize_families = penalize_families
        self.user_benefit = user_benefit or UserBenefitConfig()
        self.utility = utility or UtilityConfig()

    def value(
        self, batch: ImpressionBatch, out: RankerOutput, calibrators: CalibratorSet
    ) -> ValuedCandidates:
        """``out`` must come from ``ranker.predict`` (down-sampling-corrected logits)."""
        b, k = batch.candidate_item_ids.shape
        z1, z2, z3 = (t.detach().cpu().numpy().reshape(-1) for t in (out.z1, out.z2, out.z3))
        p1, p2, p3 = (p.reshape(b, k) for p in calibrators.calibrate(z1, z2, z3))
        pf = p_funded(p1, p2, p3)
        payout = batch.payouts.cpu().numpy().astype(np.float64)
        ev = expected_value(pf, payout)
        amount = (
            expected_amount(out.amount_logits)
            if out.amount_logits is not None
            else np.zeros((b, k))
        )
        ids = batch.candidate_item_ids.cpu().numpy()
        nb_res = net_user_benefit(
            ids, batch.user_indices.cpu().numpy(), amount, self.econ, self.state, self.user_benefit
        )
        family = batch.family_ids.cpu().numpy()
        mask = batch.candidate_mask.cpu().numpy()
        pending_pen = (
            batch.pending_family_mask.cpu().numpy()
            & self.penalize_families[np.clip(family, 0, NUM_FAMILIES - 1)]
        )
        g = apply_guardrails(
            pf, payout, nb_res.nb, family, mask, nb_res.refinance_ok, pending_pen, self.utility
        )
        return ValuedCandidates(
            p1=p1, p2=p2, p3=p3, p_funded=pf, ev=ev, expected_amount=amount, nb=g.nb,
            utility=g.utility, keep=g.keep, refinance_ok=nb_res.refinance_ok.astype(bool),
            guardrails=g, raw=out,
        )  # fmt: skip
