"""Personalized Re-ranking Model (Pei et al. 2019) for the final Top-10 slate.

Pointwise scores treat every candidate in isolation.  A slate is not a set of
independent decisions: two balance-transfer cards side by side split the same click
(*substitution*), and a flashy card next to a mortgage changes how the mortgage looks
(*context*).  PRM runs a small Pre-LN transformer over the ``K = 10`` slots so every
score is conditioned on the whole slate, adds the incoming rank as a *slot position
embedding*, and concatenates the user vector so the interactions are personalised.

Input per slot: ``[h_cand ‖ p̂1 ‖ p̂2 ‖ p̂3 ‖ EV ‖ NB ‖ U ‖ family one-hot ‖ position]``
(dollar features scaled by ``1/100``).  PRM outputs an **ordering**, not probabilities:
the calibrated ``p̂`` computed before it are what gets logged and audited, so nothing is
re-calibrated after PRM.  The family-cannibalization penalty is applied greedily at
inference (:func:`rerank`) with weight ``β`` — a serving policy, kept out of the loss so
it can be tuned without retraining (the loss has its own optional regularizer).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from recsys.data.schema import NUM_FAMILIES
from recsys.layers.transformer_blocks import TransformerEncoder, TransformerEncoderConfig
from recsys.losses.listwise_loss import PRMTarget

DOLLAR_SCALE = 1.0 / 100.0


@dataclass(frozen=True)
class PRMConfig:
    cand_dim: int  # d of h_cand
    user_dim: int  # d of h_user
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    d_ff: int = 64
    slate_size: int = 10
    dropout: float = 0.0
    cannibalization_weight: float = 0.5  # β, greedy discount at inference (score units)
    loss_cannibalization_weight: float = 0.0  # optional listwise regularizer
    prm_target: PRMTarget = "click"
    utility_temperature: float = 25.0  # τ (dollars) for prm_target="utility"

    @property
    def feature_dim(self) -> int:
        """``h_cand + 3 probs + EV + NB + U + family one-hot + position``."""
        return self.cand_dim + 3 + 3 + NUM_FAMILIES + 1


def build_prm_features(
    h_cand: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    ev: torch.Tensor,
    nb: torch.Tensor,
    u: torch.Tensor,
    family_ids: torch.Tensor,
) -> torch.Tensor:
    """``(B, K, d)`` + seven ``(B, K)`` signals + ``(B, K)`` family ids -> ``(B, K, feature_dim)``.

    Position is the incoming rank ``k / K``; ``-inf`` utilities (excluded slots) become 0.
    """
    b, k, _ = h_cand.shape
    fam = torch.nn.functional.one_hot(family_ids.clamp(min=0), NUM_FAMILIES).to(h_cand.dtype)
    fam = fam * (family_ids >= 0).unsqueeze(-1).to(h_cand.dtype)
    pos = (torch.arange(k, device=h_cand.device, dtype=h_cand.dtype) / max(k, 1)).expand(b, k)
    u_safe = torch.nan_to_num(u, neginf=0.0, posinf=0.0)
    dollars = torch.stack([ev, nb, u_safe], dim=-1) * DOLLAR_SCALE
    probs = torch.stack([p1, p2, p3], dim=-1)
    return torch.cat([h_cand, probs, dollars, fam, pos.unsqueeze(-1)], dim=-1)


@dataclass
class PRMOutput:
    scores: torch.Tensor  # (B, K), -inf on masked slots
    hidden: torch.Tensor  # (B, K, D)


class PRM(nn.Module):
    def __init__(self, config: PRMConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.feature_dim + config.user_dim, config.d_model)
        self.slot_pos_emb = nn.Embedding(config.slate_size, config.d_model)
        self.encoder = TransformerEncoder(
            TransformerEncoderConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                dropout=config.dropout,
                causal=False,
            )
        )
        self.head = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.slot_pos_emb.weight, std=0.02)

    def forward(
        self,
        slot_features: torch.Tensor,
        user_vec: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> PRMOutput:
        """``(B, K, feature_dim)``, ``(B, user_dim)``, ``(B, K)`` bool -> scores ``(B, K)``.

        Slot ``k`` is the candidate's position in the *incoming* (utility) ranking.
        """
        b, k, _ = slot_features.shape
        if k > self.config.slate_size:
            raise ValueError(f"slate of {k} exceeds slate_size {self.config.slate_size}")
        u = user_vec.unsqueeze(1).expand(-1, k, -1)
        x = self.input_proj(torch.cat([slot_features, u], dim=-1))
        x = x + self.slot_pos_emb(torch.arange(k, device=x.device)).unsqueeze(0)
        x = x * candidate_mask.unsqueeze(-1).to(x.dtype)
        hidden = self.encoder(x, candidate_mask)
        scores = self.head(hidden).squeeze(-1).masked_fill(~candidate_mask, float("-inf"))
        return PRMOutput(scores=scores, hidden=hidden)


@torch.no_grad()
def rerank(
    scores: torch.Tensor,
    family_ids: torch.Tensor,
    mask: torch.Tensor,
    cannibalization_weight: float = 0.0,
) -> torch.Tensor:
    """Greedy slate construction with a same-family discount.

    Pick the best remaining candidate, then subtract ``cannibalization_weight`` from every
    remaining candidate of the same family, repeat.  Returns ``(B, K)`` slot indices in
    display order; masked slots are always last.  With weight 0 this is a plain argsort.
    """
    b, k = scores.shape
    working = torch.nan_to_num(scores.clone(), neginf=-1e30).masked_fill(~mask, -1e30)
    taken = torch.zeros((b, k), dtype=torch.bool, device=scores.device)
    order = torch.zeros((b, k), dtype=torch.int64, device=scores.device)
    for step in range(k):
        pick = working.masked_fill(taken, float("-inf")).argmax(dim=-1)  # (B,)
        order[:, step] = pick
        taken.scatter_(1, pick.unsqueeze(1), True)
        if cannibalization_weight > 0.0:
            picked_family = family_ids.gather(1, pick.unsqueeze(1))  # (B, 1)
            same = (family_ids == picked_family) & mask & ~taken
            working = working - cannibalization_weight * same.to(working.dtype)
    return order
