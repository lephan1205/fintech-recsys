"""Impression-space collator for the ranking / multi-task / re-ranking stages.

An :class:`ImpressionSlate` is a served list of ``K`` candidates with binary
funnel labels.  The collator turns a list of slates into dense ``(B, K, ...)``
tensors, tiling user features across the slate so tabular models can consume
``tabular = [user_features ‖ candidate_features]``.

Observed labels (format v2)
---------------------------
Slates store the *oracle* outcome.  The collator derives the **observed** view at
``snapshot_at_days``: on ``PENDING`` rows ``y_approve`` and ``amounts`` are zeroed
and ``approve_observed`` is False; ``approve_weight`` implements the configured
``pending_policy`` (D1).  The oracle values are kept as ``y_approve_oracle`` /
``amounts_oracle`` for ablations and tests only — never as training labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt
import torch

from recsys.data.collator import SequenceBatch
from recsys.data.delayed_feedback import (
    PendingPolicy,
    approve_observed,
    approve_weights,
    observed_status,
)
from recsys.data.schema import NUM_FAMILIES, DelayConfig, ImpressionSlate


@dataclass
class ImpressionBatch:
    user_indices: torch.Tensor  # (B,) int64
    user_features: torch.Tensor  # (B, F_u) float32
    candidate_item_ids: torch.Tensor  # (B, K) int64, 0 = empty slot
    candidate_features: torch.Tensor  # (B, K, F_p) float32
    family_ids: torch.Tensor  # (B, K) int64, -1 for empty slots
    tabular: torch.Tensor  # (B, K, F_u + F_p) float32
    y_click: torch.Tensor  # (B, K) float32
    y_apply: torch.Tensor  # (B, K) float32
    y_approve: torch.Tensor  # (B, K) float32  OBSERVED: 0 on pending rows
    p_click: torch.Tensor  # (B, K) float32 (generator truth, evaluation only)
    p_apply: torch.Tensor  # (B, K) float32
    p_approve: torch.Tensor  # (B, K) float32
    payouts: torch.Tensor  # (B, K) float32
    amounts: torch.Tensor  # (B, K) float32  OBSERVED: 0 on pending rows
    eligible: torch.Tensor  # (B, K) bool
    candidate_mask: torch.Tensor  # (B, K) bool, False for empty slots
    # --- delayed feedback (v2) --------------------------------------------------------
    served_at_days: torch.Tensor  # (B,) float32
    elapsed_days: torch.Tensor  # (B, K) float32 = snapshot - served_at (tiled)
    status: torch.Tensor  # (B, K) int64 ApplicationStatus
    approve_observed: torch.Tensor  # (B, K) bool, status in {NOT_APPLIED, APPROVED, DECLINED}
    approve_weight: torch.Tensor  # (B, K) float32 per pending_policy
    pending_family_mask: torch.Tensor  # (B, K) bool, candidate family in user.pending_family_ids
    y_approve_oracle: torch.Tensor  # (B, K) float32 eventual outcome (ablation / tests only)
    amounts_oracle: torch.Tensor  # (B, K) float32
    sequence: SequenceBatch | None = None

    @property
    def batch_size(self) -> int:
        return int(self.candidate_item_ids.shape[0])

    @property
    def slate_size(self) -> int:
        return int(self.candidate_item_ids.shape[1])

    def with_sequence(self, sequence: SequenceBatch) -> ImpressionBatch:
        if sequence.batch_size != self.batch_size:
            raise ValueError("sequence batch size must match impression batch size")
        kwargs = {f.name: getattr(self, f.name) for f in fields(self)}
        kwargs["sequence"] = sequence
        return ImpressionBatch(**kwargs)

    def to(self, device: torch.device | str) -> ImpressionBatch:
        kwargs = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor | SequenceBatch):
                v = v.to(device)
            kwargs[f.name] = v
        return ImpressionBatch(**kwargs)


_FLOAT_FIELDS = (
    "y_click", "y_apply", "y_approve", "p_click", "p_apply", "p_approve",
    "payouts", "amounts", "decision_delay_days",
)  # fmt: skip


class ImpressionCollator:
    """Slates -> :class:`ImpressionBatch`.

    ``snapshot_at_days`` and ``delay_config`` are keyword-required: there is no safe
    default for the training cut-off (§0.2).  ``user_pending_families`` is the
    ``(U, NUM_FAMILIES)`` multi-hot of each user's pending families (row index =
    ``user_index``); pass ``None`` to disable the pending-family mask.
    """

    def __init__(
        self,
        catalog_features: npt.NDArray[np.float32],
        user_features: npt.NDArray[np.float32],
        family_by_item: npt.NDArray[np.int64],
        *,
        snapshot_at_days: float,
        delay_config: DelayConfig,
        pending_policy: PendingPolicy = "drop",
        w_floor: float = 0.05,
        user_pending_families: npt.NDArray[np.bool_] | None = None,
        payout_by_item: npt.NDArray[np.float64] | None = None,
        slate_size: int | None = None,
    ) -> None:
        if pending_policy not in ("drop", "ipw", "negative"):
            raise ValueError(f"unknown pending_policy {pending_policy!r}")
        self.catalog_features = torch.as_tensor(catalog_features, dtype=torch.float32)
        self.user_features = torch.as_tensor(user_features, dtype=torch.float32)
        self.family_by_item = torch.as_tensor(family_by_item, dtype=torch.int64)
        self.snapshot_at_days = float(snapshot_at_days)
        self.delay_config = delay_config
        self.pending_policy: PendingPolicy = pending_policy
        self.w_floor = w_floor
        if user_pending_families is None:
            user_pending_families = np.zeros((self.user_features.shape[0], NUM_FAMILIES), bool)
        self.user_pending_families = torch.as_tensor(user_pending_families, dtype=torch.bool)
        self.payout_by_item = (
            None if payout_by_item is None else torch.as_tensor(payout_by_item, dtype=torch.float32)
        )
        self.slate_size = slate_size

    def __call__(self, slates: Sequence[ImpressionSlate]) -> ImpressionBatch:
        b = len(slates)
        k = self.slate_size or max(s.size for s in slates)
        ids = torch.zeros((b, k), dtype=torch.int64)
        mask = torch.zeros((b, k), dtype=torch.bool)
        flt = {n: torch.zeros((b, k), dtype=torch.float32) for n in _FLOAT_FIELDS}
        eligible = torch.zeros((b, k), dtype=torch.bool)
        user_idx = torch.zeros(b, dtype=torch.int64)
        served = torch.zeros(b, dtype=torch.float32)

        for i, s in enumerate(slates):
            n = min(s.size, k)
            ids[i, :n] = torch.tensor(s.candidate_item_ids[:n], dtype=torch.int64)
            mask[i, :n] = True
            for name, buf in flt.items():
                buf[i, :n] = torch.tensor(getattr(s, name)[:n], dtype=torch.float32)
            eligible[i, :n] = torch.tensor(s.eligible[:n], dtype=torch.bool)
            user_idx[i] = s.user_index
            served[i] = s.served_at_days

        cand_feat = self.catalog_features[ids]  # (B, K, F_p); row 0 is zero
        user_feat = self.user_features[user_idx]  # (B, F_u)
        family = torch.where(mask, self.family_by_item[ids], torch.full_like(ids, -1))
        tabular = torch.cat([user_feat.unsqueeze(1).expand(-1, k, -1), cand_feat], dim=-1)

        # --- observed view at the snapshot ------------------------------------------
        y_apply_np = flt["y_apply"].numpy()
        y_approve_oracle = flt["y_approve"]
        amounts_oracle = flt["amounts"]
        elapsed = (self.snapshot_at_days - served).unsqueeze(1).expand(-1, k).contiguous()
        status_np = observed_status(
            y_apply_np,
            served.unsqueeze(1).expand(-1, k).numpy(),
            flt["decision_delay_days"].numpy(),
            self.snapshot_at_days,
            y_approve_oracle.numpy(),
        )
        resolved_np = approve_observed(status_np)
        resolved = torch.as_tensor(resolved_np, dtype=torch.bool)
        # observed view: the oracle label / amount of a pending row must never leak
        y_approve = torch.where(resolved, y_approve_oracle, torch.zeros_like(y_approve_oracle))
        amounts = torch.where(resolved, amounts_oracle, torch.zeros_like(amounts_oracle))
        observed_np = resolved_np
        if self.pending_policy == "negative":
            # baseline (wrong by construction): a pending row enters the approval terms as
            # an *observed* decline (y_approve = 0, weight 1) instead of being masked out
            observed_np = np.ones_like(resolved_np)
        observed = torch.as_tensor(observed_np, dtype=torch.bool)
        weight_np = approve_weights(
            status_np,
            elapsed.numpy(),
            np.clip(family.numpy(), 0, NUM_FAMILIES - 1),
            y_approve.numpy(),
            self.delay_config,
            self.pending_policy,
            self.w_floor,
        )
        approve_weight = torch.as_tensor(weight_np, dtype=torch.float32) * mask.to(torch.float32)
        pending_fam = self.user_pending_families[user_idx]  # (B, NUM_FAMILIES)
        fam_clamped = torch.clamp(family, min=0)
        pending_family_mask = pending_fam.gather(1, fam_clamped) & mask & (family >= 0)

        return ImpressionBatch(
            user_indices=user_idx,
            user_features=user_feat,
            candidate_item_ids=ids,
            candidate_features=cand_feat,
            family_ids=family,
            tabular=tabular,
            y_click=flt["y_click"],
            y_apply=flt["y_apply"],
            y_approve=y_approve,
            p_click=flt["p_click"],
            p_apply=flt["p_apply"],
            p_approve=flt["p_approve"],
            payouts=flt["payouts"],
            amounts=amounts,
            eligible=eligible,
            candidate_mask=mask,
            served_at_days=served,
            elapsed_days=elapsed,
            status=torch.as_tensor(status_np, dtype=torch.int64),
            approve_observed=observed,
            approve_weight=approve_weight,
            pending_family_mask=pending_family_mask,
            y_approve_oracle=y_approve_oracle,
            amounts_oracle=amounts_oracle,
        )

    def collate_candidates(
        self,
        user_indices: npt.NDArray[np.int64] | torch.Tensor,
        candidate_ids: npt.NDArray[np.int64] | torch.Tensor,
        served_at_days: float | None = None,
    ) -> ImpressionBatch:
        """Label-free batch for *serving*: ``(B,)`` users x ``(B, K)`` retrieved candidates.

        Labels / statuses are zeros, ``candidate_mask`` is ``ids > 0``, payouts come from
        ``payout_by_item`` (zeros if not provided) and ``served_at_days`` defaults to the
        snapshot ("now").
        """
        user_idx = torch.as_tensor(np.asarray(user_indices), dtype=torch.int64)
        ids = torch.as_tensor(np.asarray(candidate_ids), dtype=torch.int64)
        b, k = ids.shape
        mask = ids > 0
        zeros = torch.zeros((b, k), dtype=torch.float32)
        served = torch.full(
            (b,), self.snapshot_at_days if served_at_days is None else served_at_days
        )
        cand_feat = self.catalog_features[ids]
        user_feat = self.user_features[user_idx]
        family = torch.where(mask, self.family_by_item[ids], torch.full_like(ids, -1))
        tabular = torch.cat([user_feat.unsqueeze(1).expand(-1, k, -1), cand_feat], dim=-1)
        payouts = zeros if self.payout_by_item is None else self.payout_by_item[ids] * mask
        pending_fam = self.user_pending_families[user_idx]
        pending_family_mask = pending_fam.gather(1, torch.clamp(family, min=0)) & mask
        return ImpressionBatch(
            user_indices=user_idx,
            user_features=user_feat,
            candidate_item_ids=ids,
            candidate_features=cand_feat,
            family_ids=family,
            tabular=tabular,
            y_click=zeros.clone(),
            y_apply=zeros.clone(),
            y_approve=zeros.clone(),
            p_click=zeros.clone(),
            p_apply=zeros.clone(),
            p_approve=zeros.clone(),
            payouts=payouts,
            amounts=zeros.clone(),
            eligible=mask.clone(),
            candidate_mask=mask,
            served_at_days=served,
            elapsed_days=(self.snapshot_at_days - served).unsqueeze(1).expand(-1, k).contiguous(),
            status=torch.zeros((b, k), dtype=torch.int64),
            approve_observed=torch.ones((b, k), dtype=torch.bool),
            approve_weight=mask.to(torch.float32),
            pending_family_mask=pending_family_mask,
            y_approve_oracle=zeros.clone(),
            amounts_oracle=zeros.clone(),
        )
