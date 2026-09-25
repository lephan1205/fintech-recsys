"""Hierarchical Sequential Transduction Unit (HSTU, Zhai et al. 2024) layers.

One HSTU layer over ``X (B, S, d)``::

    [U, V, Q, K] = split(SiLU(f1(LN(X))))                      # pointwise projections
    A            = SiLU(Q Kᵀ / sqrt(d_h) + rab_pos + rab_time)  # *no* softmax
    A            = A ⊙ mask / n_valid_keys                       # normalized by valid keys
    Y            = X + f2(LN(A V) ⊙ U)                          # gated residual update

Pointwise SiLU attention (instead of softmax) keeps the *intensity* of preferences —
a user with many card views should look different from one with a single view —
which softmax normalization erases.  ``rab_pos`` / ``rab_time`` are learned relative
position / relative time biases (per head), bucketized.

M-FALCON batched scoring: the sequence is ``[history (L, left-padded, causal) ‖ K
candidate tokens]``.  :func:`build_mfalcon_mask` lets history attend causally to real
history, each candidate attend to all real history and to itself, never to other
candidates, and history never to candidates.  One pass of length ``L + K`` scores all
candidates, and each candidate's representation is provably independent of which
other candidates were retrieved (tested in ``tests/test_hstu.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from recsys.data.collator import PaddingSide
from recsys.data.schema import NUM_ACTIONS, PRODUCT_FEATURE_DIM, USER_FEATURE_DIM


@dataclass(frozen=True)
class HSTUConfig:
    num_items: int
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    max_len: int = 64  # L: history tokens
    max_candidates: int = 100  # K
    num_actions: int = NUM_ACTIONS
    num_time_buckets: int = 32
    max_time_days: float = 365.0
    tabular_dim: int = USER_FEATURE_DIM + PRODUCT_FEATURE_DIM
    dropout: float = 0.0
    padding_side: PaddingSide = "left"

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def max_rel_pos(self) -> int:
        return self.max_len + self.max_candidates

    @property
    def fusion_dim(self) -> int:
        """``[h_cand ‖ h_user ‖ h_cand ⊙ h_user ‖ tabular]``."""
        return 3 * self.d_model + self.tabular_dim


def bucketize_time(days: torch.Tensor, num_buckets: int, max_days: float) -> torch.Tensor:
    """Log-spaced buckets over ``log1p(days)`` up to ``max_days``; ``(...)`` -> int64."""
    scaled = torch.log1p(days.clamp(min=0.0)) / math.log1p(max_days)
    idx = torch.floor(scaled * (num_buckets - 1)).to(torch.int64)
    return idx.clamp(0, num_buckets - 1)


def build_mfalcon_mask(attention_mask: torch.Tensor, num_candidates: int) -> torch.Tensor:
    """``(B, L)`` history mask -> ``(B, L+K, L+K)`` bool attention mask (query x key).

    history -> history: causal and key-real; candidate -> real history and itself;
    candidate -> other candidate: never; history -> candidate: never.  A fully padded
    history row still attends to itself so no normalizer is zero.
    """
    b, ell = attention_mask.shape
    s = ell + num_candidates
    device = attention_mask.device
    out = torch.zeros((b, s, s), dtype=torch.bool, device=device)
    causal = torch.tril(torch.ones((ell, ell), dtype=torch.bool, device=device))
    hist_keys = attention_mask.unsqueeze(1)  # (B, 1, L)
    out[:, :ell, :ell] = causal.unsqueeze(0) & hist_keys
    out[:, ell:, :ell] = hist_keys.expand(-1, num_candidates, -1)
    eye = torch.eye(s, dtype=torch.bool, device=device)
    out |= eye.unsqueeze(0)
    return out


class HSTULayer(nn.Module):
    def __init__(self, config: HSTUConfig) -> None:
        super().__init__()
        self.config = config
        d, h = config.d_model, config.n_heads
        self.ln_in = nn.LayerNorm(d)
        self.f1 = nn.Linear(d, 4 * d)
        self.ln_attn = nn.LayerNorm(d)
        self.f2 = nn.Linear(d, d)
        self.rab_pos = nn.Embedding(2 * config.max_rel_pos + 1, h)
        self.rab_time = nn.Embedding(config.num_time_buckets, h)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.rab_pos.weight, std=0.02)
        nn.init.normal_(self.rab_time.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rel_pos: torch.Tensor,
        time_bucket: torch.Tensor,
    ) -> torch.Tensor:
        """``x (B, S, d)``, ``attn_mask (B, S, S)`` bool, ``rel_pos (B, S, S)`` int64 indices
        into ``rab_pos``, ``time_bucket (B, S, S)`` int64 -> ``(B, S, d)``."""
        b, s, d = x.shape
        h, dh = self.config.n_heads, self.config.head_dim
        u, v, q, k = F.silu(self.f1(self.ln_in(x))).split(d, dim=-1)
        q = q.view(b, s, h, dh).transpose(1, 2)  # (B, h, S, dh)
        k = k.view(b, s, h, dh).transpose(1, 2)
        v = v.view(b, s, h, dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # (B, h, S, S)
        bias = self.rab_pos(rel_pos) + self.rab_time(time_bucket)  # (B, S, S, h)
        scores = scores + bias.permute(0, 3, 1, 2)
        a = F.silu(scores) * attn_mask.unsqueeze(1).to(scores.dtype)
        n_valid = attn_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(scores.dtype)  # (B, S, 1)
        a = a / n_valid.unsqueeze(1)
        ctx = (a @ v).transpose(1, 2).reshape(b, s, d)  # (B, S, d)
        gated = self.ln_attn(ctx) * u
        out: torch.Tensor = x + self.dropout(self.f2(gated))
        return out


@dataclass
class HSTUEncoded:
    hidden: torch.Tensor  # (B, L+K, d)
    h_user: torch.Tensor  # (B, d) at the most recent real history event
    h_cand: torch.Tensor  # (B, K, d)
