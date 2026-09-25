"""Listwise (ListNet) loss and the family-cannibalization regularizer for the PRM (D5).

Target distribution ``q`` over the slate is either ``q_i ∝ y_click_i``
(``prm_target = "click"``, default) or ``q_i = softmax(U_i / τ)`` (``"utility"``, which
stays well defined when some ``U_i < 0``).  Slates with no positive (or no finite
utility) contribute zero.  The cannibalization regularizer is
``Σ_{i≠j} 1[fam_i = fam_j] p_i p_j`` on the softmax of the scores.
"""

from __future__ import annotations

from typing import Literal

import torch

PRMTarget = Literal["click", "utility"]


def same_family_pairs(family_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(B, K, K)`` bool: both slots real, same family, ``i != j``."""
    same = family_ids.unsqueeze(-1) == family_ids.unsqueeze(-2)
    both = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    eye = torch.eye(family_ids.shape[-1], dtype=torch.bool, device=family_ids.device)
    return same & both & ~eye


def same_family_penalty(
    scores: torch.Tensor, family_ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``sum_{i != j} 1[fam_i == fam_j] p_i p_j`` with ``p = softmax(scores)``; ``(B,)``."""
    p = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    pairs = same_family_pairs(family_ids, mask).to(p.dtype)
    return torch.einsum("bi,bij,bj->b", p, pairs, p)


def prm_target_distribution(
    labels: torch.Tensor,
    mask: torch.Tensor,
    target: PRMTarget = "click",
    utility: torch.Tensor | None = None,
    temperature: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q (B, K), has_target (B,))`` — the listwise target and which slates count."""
    m = mask.to(labels.dtype)
    if target == "click":
        t = labels * m
        total = t.sum(dim=-1, keepdim=True)
        q = torch.where(total > 0, t / total.clamp(min=1e-12), torch.zeros_like(t))
        return q, total.squeeze(-1) > 0
    if utility is None:
        raise ValueError("prm_target='utility' needs the utility tensor")
    u = utility.masked_fill(~mask, float("-inf")) / temperature
    finite = torch.isfinite(u) & mask
    has = finite.any(dim=-1)
    q = torch.softmax(torch.where(finite, u, torch.full_like(u, float("-inf"))), dim=-1)
    q = torch.nan_to_num(q, nan=0.0)
    return q, has


def listnet_loss(
    scores: torch.Tensor,
    target_dist: torch.Tensor,
    mask: torch.Tensor,
    has_target: torch.Tensor,
) -> torch.Tensor:
    """Softmax cross-entropy between ``softmax(scores)`` and ``target_dist`` over real slots."""
    log_p = torch.log_softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    per_slate = -(target_dist * torch.nan_to_num(log_p, neginf=0.0)).sum(dim=-1)
    if not bool(has_target.any()):
        return torch.nan_to_num(scores, neginf=0.0, posinf=0.0).sum() * 0.0
    return per_slate[has_target].mean()


def prm_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    family_ids: torch.Tensor,
    cannibalization_weight: float = 0.0,
    target: PRMTarget = "click",
    utility: torch.Tensor | None = None,
    temperature: float = 25.0,
) -> torch.Tensor:
    """ListNet loss + ``λ · same_family_penalty`` (mean over the batch)."""
    q, has = prm_target_distribution(labels, mask, target, utility, temperature)
    loss = listnet_loss(scores, q, mask, has)
    if cannibalization_weight > 0.0:
        loss = loss + cannibalization_weight * same_family_penalty(scores, family_ids, mask).mean()
    return loss
