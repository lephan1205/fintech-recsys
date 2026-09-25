"""Impression-space collator for the ranking / multi-task / re-ranking stages.

An :class:`ImpressionSlate` is a served list of ``K`` candidates with binary
funnel labels.  The collator turns a list of slates into dense ``(B, K, ...)``
tensors, tiling user features across the slate so tabular models (DCN-v2,
PLE, ESMM) can consume ``tabular = [user_features ‖ candidate_features]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt
import torch

from recsys.data.collator import SequenceBatch
from recsys.data.schema import ImpressionSlate


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
    y_approve: torch.Tensor  # (B, K) float32
    p_click: torch.Tensor  # (B, K) float32 (generator truth, evaluation only)
    p_apply: torch.Tensor  # (B, K) float32
    p_approve: torch.Tensor  # (B, K) float32
    payouts: torch.Tensor  # (B, K) float32
    amounts: torch.Tensor  # (B, K) float32
    eligible: torch.Tensor  # (B, K) bool
    candidate_mask: torch.Tensor  # (B, K) bool, False for empty slots
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


class ImpressionCollator:
    def __init__(
        self,
        catalog_features: npt.NDArray[np.float32],
        user_features: npt.NDArray[np.float32],
        family_by_item: npt.NDArray[np.int64],
        slate_size: int | None = None,
    ) -> None:
        self.catalog_features = torch.as_tensor(catalog_features, dtype=torch.float32)
        self.user_features = torch.as_tensor(user_features, dtype=torch.float32)
        self.family_by_item = torch.as_tensor(family_by_item, dtype=torch.int64)
        self.slate_size = slate_size

    def __call__(self, slates: Sequence[ImpressionSlate]) -> ImpressionBatch:
        b = len(slates)
        k = self.slate_size or max(s.size for s in slates)
        ids = torch.zeros((b, k), dtype=torch.int64)
        mask = torch.zeros((b, k), dtype=torch.bool)
        flt = {
            n: torch.zeros((b, k), dtype=torch.float32)
            for n in (
                "y_click", "y_apply", "y_approve", "p_click", "p_apply", "p_approve",
                "payouts", "amounts",
            )
        }  # fmt: skip
        eligible = torch.zeros((b, k), dtype=torch.bool)
        user_idx = torch.zeros(b, dtype=torch.int64)

        for i, s in enumerate(slates):
            n = min(s.size, k)
            ids[i, :n] = torch.tensor(s.candidate_item_ids[:n], dtype=torch.int64)
            mask[i, :n] = True
            for name, buf in flt.items():
                buf[i, :n] = torch.tensor(getattr(s, name)[:n], dtype=torch.float32)
            eligible[i, :n] = torch.tensor(s.eligible[:n], dtype=torch.bool)
            user_idx[i] = s.user_index

        cand_feat = self.catalog_features[ids]  # (B, K, F_p); row 0 is zero
        user_feat = self.user_features[user_idx]  # (B, F_u)
        family = torch.where(mask, self.family_by_item[ids], torch.full_like(ids, -1))
        tabular = torch.cat([user_feat.unsqueeze(1).expand(-1, k, -1), cand_feat], dim=-1)

        return ImpressionBatch(
            user_indices=user_idx,
            user_features=user_feat,
            candidate_item_ids=ids,
            candidate_features=cand_feat,
            family_ids=family,
            tabular=tabular,
            y_click=flt["y_click"],
            y_apply=flt["y_apply"],
            y_approve=flt["y_approve"],
            p_click=flt["p_click"],
            p_apply=flt["p_apply"],
            p_approve=flt["p_approve"],
            payouts=flt["payouts"],
            amounts=flt["amounts"],
            eligible=eligible,
            candidate_mask=mask,
        )
