"""``HSTUPLERanker``: HSTU backbone + PLE funnel towers, one batched pass per slate.

The ranker owns two serving-time corrections that must never be forgotten:

* **Negative down-sampling logit correction** (D2): training keeps all clicked
  impressions and a fraction ``r`` of the non-clicked ones, so the click tower learns
  ``z_train``; at inference ``z_true = z_train + log r``.  ``logit_correction`` is a
  buffer set from the training rate and applied in :meth:`predict` only — not in
  :meth:`forward`, which is what the loss sees.
* **Tabular standardization**: the fusion vector mixes ``log1p`` scales with one-hots;
  ``tabular_mean / tabular_std`` are fitted on the training split and applied inside
  the model so serving cannot drift from training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from recsys.data.impression_collator import ImpressionBatch
from recsys.layers.hstu import HSTUConfig
from recsys.models.hstu.model import HSTUBackbone
from recsys.models.ple.model import PLE, PLEConfig


@dataclass
class RankerOutput:
    z1: torch.Tensor  # (B, K) click logit (raw in forward, +log r in predict)
    z2: torch.Tensor  # (B, K) apply | click logit
    z3: torch.Tensor  # (B, K) approve | apply logit
    amount_logits: torch.Tensor | None  # (B, K, 3) ZILN
    imputation_logits: torch.Tensor | None  # (B, K) for ssb_mode="dr"
    h_user: torch.Tensor  # (B, d)
    h_cand: torch.Tensor  # (B, K, d)
    fused: torch.Tensor  # (B, K, fusion_dim)

    @property
    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.z1), torch.sigmoid(self.z2), torch.sigmoid(self.z3)


class HSTUPLERanker(nn.Module):
    def __init__(
        self,
        hstu_config: HSTUConfig,
        ple_config: PLEConfig,
        downsample_rate: float = 0.25,
    ) -> None:
        super().__init__()
        if ple_config.input_dim != hstu_config.fusion_dim:
            raise ValueError(
                f"PLE input_dim {ple_config.input_dim} != HSTU fusion_dim {hstu_config.fusion_dim}"
            )
        if not 0.0 < downsample_rate <= 1.0:
            raise ValueError("downsample_rate must be in (0, 1]")
        self.backbone = HSTUBackbone(hstu_config)
        self.ple = PLE(ple_config)
        self.register_buffer("logit_correction", torch.tensor(math.log(downsample_rate)))
        self.register_buffer("tabular_mean", torch.zeros(hstu_config.tabular_dim))
        self.register_buffer("tabular_std", torch.ones(hstu_config.tabular_dim))
        self.logit_correction: torch.Tensor
        self.tabular_mean: torch.Tensor
        self.tabular_std: torch.Tensor

    @property
    def downsample_rate(self) -> float:
        return float(torch.exp(self.logit_correction))

    @torch.no_grad()
    def fit_tabular_stats(self, tabular: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Fit standardization on ``(..., F)`` rows (optionally masked) from the train split."""
        rows = tabular.reshape(-1, tabular.shape[-1])
        if mask is not None:
            rows = rows[mask.reshape(-1)]
        self.tabular_mean.copy_(rows.mean(dim=0))
        self.tabular_std.copy_(rows.std(dim=0, unbiased=False).clamp(min=1e-6))

    def _standardize(self, tabular: torch.Tensor) -> torch.Tensor:
        return (tabular - self.tabular_mean) / self.tabular_std

    def forward(self, batch: ImpressionBatch) -> RankerOutput:
        """Raw (training) logits.  ``batch.sequence`` must be set."""
        if batch.sequence is None:
            raise ValueError("ImpressionBatch.sequence is required (use with_sequence)")
        enc = self.backbone.score_candidates(batch.sequence, batch.candidate_item_ids)
        fused = self.backbone.fusion(enc.h_user, enc.h_cand, self._standardize(batch.tabular))
        out = self.ple(fused)
        return RankerOutput(
            z1=out.click_logit,
            z2=out.apply_logit,
            z3=out.approve_logit,
            amount_logits=out.amount_logits,
            imputation_logits=out.logits.get("imputation"),
            h_user=enc.h_user,
            h_cand=enc.h_cand,
            fused=fused,
        )

    @torch.no_grad()
    def predict(self, batch: ImpressionBatch) -> RankerOutput:
        """Serving logits: the click logit is corrected for negative down-sampling."""
        out = self.forward(batch)
        out.z1 = out.z1 + self.logit_correction
        return out
