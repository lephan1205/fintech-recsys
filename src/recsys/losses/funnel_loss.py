"""Unified Funnel Loss (D3): conditional task terms + log-space entire-space terms.

Notation: PLE towers produce logits ``z1, z2, z3`` for ``p1 = p(Click)``,
``p2 = p(Apply | Click)``, ``p3 = p(Approve | Apply)``; ``s_k = logsigmoid(z_k)``;
``m_c = y_click``, ``m_a = y_apply``, ``o`` = approval-observed mask, ``w`` = D1 weight.

Conditional (task) terms — biased sample spaces, sharp signal::

    L_click   = mean_all   [ BCE(z1, y_click) ]
    L_apply   = mean_{m_c} [ BCE(z2, y_apply) ]               # clicked impressions only
    L_approve = mean_{m_a·o} [ w · BCE(z3, y_approve) ]       # resolved applications only

Entire-space terms — unbiased sample space, log-space on the SAME towers::

    log p_ctcvr  = s1 + s2           log(1 - p_ctcvr)  = log1mexp(s1 + s2)
    log p_ctcavr = s1 + s2 + s3      log(1 - p_ctcavr) = log1mexp(s1 + s2 + s3)
    L_ctcvr  = mean_all [ -( y_apply · log p_ctcvr + (1 - y_apply) · log(1 - p_ctcvr) ) ]
    L_ctcavr = mean_all [ -w' ( y_approve · log p_ctcavr + (1 - y_approve) · log(1 - p_ctcavr) ) ]
               w' = 1 for non-applied rows, w for resolved applications, 0 for pending

    L = λ_click L_click + λ_apply L_apply + λ_approve L_approve
        + μ_ctcvr L_ctcvr + μ_ctcavr L_ctcavr (+ λ_amount · ZILN on resolved *approved* rows)

This *is* the ESMM sample-selection-bias correction without a second network: the
multiplicative terms supervise ``p2`` and ``p3`` on every impression (including
non-clicked ones, through the product with ``p1``) while the conditional terms keep
each tower sharp on its observed sub-population.  Every masked term is normalized by
its mask sum, not the batch size, so a batch with 3 applies does not vanish.

Business objectives never enter these losses (§0.3): they are proper scoring rules only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from recsys.losses.stable import bce_from_log_prob, log1mexp, logsigmoid, masked_mean
from recsys.losses.ziln_loss import ziln_loss

LossBalancing = Literal["fixed", "running_mean", "uncertainty"]
SSBMode = Literal["none", "ips", "dr"]

TERM_NAMES: tuple[str, ...] = ("click", "apply", "approve", "ctcvr", "ctcavr", "amount")


@dataclass(frozen=True)
class FunnelLossConfig:
    lambda_click: float = 1.0
    lambda_apply: float = 1.0
    lambda_approve: float = 1.0
    mu_ctcvr: float = 1.0
    mu_ctcavr: float = 1.0
    lambda_amount: float = 0.1
    loss_balancing: LossBalancing = "fixed"
    ssb_mode: SSBMode = "none"
    detach_upstream: bool = False
    ips_eps: float = 0.05  # clip for 1 / p1 in "ips" / "dr"
    running_mean_decay: float = 0.99

    @property
    def weights(self) -> tuple[float, ...]:
        return (
            self.lambda_click,
            self.lambda_apply,
            self.lambda_approve,
            self.mu_ctcvr,
            self.mu_ctcavr,
            self.lambda_amount,
        )


@dataclass
class FunnelLossTerms:
    total: torch.Tensor
    click: torch.Tensor
    apply: torch.Tensor
    approve: torch.Tensor
    ctcvr: torch.Tensor
    ctcavr: torch.Tensor
    amount: torch.Tensor
    imputation: torch.Tensor
    num_rows: int
    num_clicked: int
    num_resolved_applications: int
    num_amount_rows: int
    effective_weights: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        out = {
            name: float(getattr(self, name).detach())
            for name in ("total", *TERM_NAMES, "imputation")
        }
        out.update(
            num_rows=float(self.num_rows),
            num_clicked=float(self.num_clicked),
            num_resolved_applications=float(self.num_resolved_applications),
            num_amount_rows=float(self.num_amount_rows),
        )
        return out


class UnifiedFunnelLoss(nn.Module):
    def __init__(self, config: FunnelLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or FunnelLossConfig()
        n = len(TERM_NAMES)
        # Kendall et al. homoscedastic-uncertainty weights (used only for "uncertainty")
        self.log_var = nn.Parameter(
            torch.zeros(n), requires_grad=self.config.loss_balancing == "uncertainty"
        )
        self.register_buffer("ema", torch.full((n,), -1.0))
        self.ema: torch.Tensor

    # ------------------------------------------------------------------ terms
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        y_click: torch.Tensor,
        y_apply: torch.Tensor,
        y_approve: torch.Tensor,
        approve_weight: torch.Tensor,
        approve_observed: torch.Tensor,
        candidate_mask: torch.Tensor,
        train_mask: torch.Tensor | None = None,
        amount_logits: torch.Tensor | None = None,
        amounts: torch.Tensor | None = None,
        imputation_logits: torch.Tensor | None = None,
    ) -> FunnelLossTerms:
        """All tensors ``(B, K)`` except ``amount_logits (B, K, 3)``.

        ``train_mask`` is the negative down-sampling keep mask (all True when absent).
        ``y_approve`` / ``amounts`` are the *observed* labels (0 on pending rows).
        """
        cfg = self.config
        m_all = candidate_mask if train_mask is None else candidate_mask & train_mask
        m_c = m_all & (y_click > 0.5)
        m_a = m_all & (y_apply > 0.5)
        o = approve_observed.to(torch.bool)
        w = approve_weight.to(z1.dtype)

        # --- conditional terms ------------------------------------------------------
        click = masked_mean(
            F.binary_cross_entropy_with_logits(z1, y_click, reduction="none"), m_all
        )
        apply_row = F.binary_cross_entropy_with_logits(z2, y_apply, reduction="none")
        ips_w = torch.ones_like(z1)
        if cfg.ssb_mode in ("ips", "dr"):
            ips_w = 1.0 / torch.sigmoid(z1).detach().clamp(min=cfg.ips_eps, max=1.0)
        apply = masked_mean(apply_row * ips_w, m_c)
        approve_row = F.binary_cross_entropy_with_logits(z3, y_approve, reduction="none")
        approve = masked_mean(w * approve_row, m_a & o)

        # --- entire-space terms (log space, same towers) ---------------------------
        s1, s2, s3 = logsigmoid(z1), logsigmoid(z2), logsigmoid(z3)
        s1_up = s1.detach() if cfg.detach_upstream else s1
        s2_up = s2.detach() if cfg.detach_upstream else s2
        log_ctcvr = s1_up + s2
        ctcvr = masked_mean(bce_from_log_prob(log_ctcvr, y_apply), m_all)
        log_ctcavr = s1_up + s2_up + s3
        w_prime = torch.where(y_apply > 0.5, w * o.to(z1.dtype), torch.ones_like(w))
        ctcavr = masked_mean(w_prime * bce_from_log_prob(log_ctcavr, y_approve), m_all)

        # --- amount tower (resolved approved rows only) ----------------------------
        m_amt = m_a & o & (y_approve > 0.5)
        if amount_logits is not None and amounts is not None:
            amount = ziln_loss(amount_logits, amounts, m_amt).total
        else:
            amount = z1.sum() * 0.0

        # --- doubly-robust correction (ESCM²-DR) -----------------------------------
        imputation = z1.sum() * 0.0
        if cfg.ssb_mode == "dr":
            if imputation_logits is None:
                raise ValueError("ssb_mode='dr' needs imputation_logits")
            imp = F.softplus(imputation_logits)  # predicted apply loss, >= 0
            err = apply_row.detach() - imp
            # DR estimate of the impression-space apply loss + imputation fit on clicked rows
            dr = masked_mean(imp, m_all) + masked_mean(m_c.to(z1.dtype) * ips_w * err, m_all)
            imputation = masked_mean((imp - apply_row.detach()) ** 2, m_c)
            apply = dr + imputation

        terms = [click, apply, approve, ctcvr, ctcavr, amount]
        total, eff = self._combine(terms)
        return FunnelLossTerms(
            total=total,
            click=click,
            apply=apply,
            approve=approve,
            ctcvr=ctcvr,
            ctcavr=ctcavr,
            amount=amount,
            imputation=imputation,
            num_rows=int(m_all.sum()),
            num_clicked=int(m_c.sum()),
            num_resolved_applications=int((m_a & o).sum()),
            num_amount_rows=int(m_amt.sum()),
            effective_weights=eff,
        )

    def _combine(self, terms: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.config
        base = cfg.weights
        eff: dict[str, float] = {}
        if cfg.loss_balancing == "fixed":
            total = sum((wt * t for wt, t in zip(base, terms, strict=True)), terms[0] * 0.0)
            eff = dict(zip(TERM_NAMES, base, strict=True))
            return total, eff
        if cfg.loss_balancing == "running_mean":
            values = torch.stack([t.detach() for t in terms])
            if self.training:
                with torch.no_grad():
                    unset = self.ema < 0
                    self.ema = torch.where(
                        unset,
                        values,
                        cfg.running_mean_decay * self.ema + (1.0 - cfg.running_mean_decay) * values,
                    )
            scale = torch.where(self.ema > 0, self.ema, torch.ones_like(self.ema)).clamp(min=1e-8)
            total = terms[0] * 0.0
            for i, (wt, t) in enumerate(zip(base, terms, strict=True)):
                total = total + wt * t / scale[i]
                eff[TERM_NAMES[i]] = float(wt / scale[i])
            return total, eff
        # uncertainty (Kendall et al. 2018): sum exp(-s_i) L_i + s_i
        total = terms[0] * 0.0
        for i, (wt, t) in enumerate(zip(base, terms, strict=True)):
            precision = torch.exp(-self.log_var[i])
            total = total + wt * (precision * t + 0.5 * self.log_var[i])
            eff[TERM_NAMES[i]] = float(wt * precision.detach())
        return total, eff


def log_ctcvr_from_logits(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """``log p(CTCVR) = logsigmoid(z1) + logsigmoid(z2)`` (helper for tests / serving)."""
    return logsigmoid(z1) + logsigmoid(z2)


def naive_entire_space_bce(p_product: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference implementation in probability space (only valid at moderate logits)."""
    return -(y * torch.log(p_product) + (1 - y) * torch.log(1 - p_product))


__all__ = [
    "FunnelLossConfig",
    "FunnelLossTerms",
    "LossBalancing",
    "SSBMode",
    "TERM_NAMES",
    "UnifiedFunnelLoss",
    "log1mexp",
    "log_ctcvr_from_logits",
    "naive_entire_space_bce",
]
