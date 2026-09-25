"""Pre-LN transformer blocks shared by every sequential model in the repo.

* :func:`build_attention_mask` turns a ``(B, L)`` key-validity mask into the
  ``(B, 1, L, L)`` boolean mask expected by ``scaled_dot_product_attention``.
  The diagonal is always allowed so that fully padded query rows never
  produce a NaN softmax (their outputs are garbage but are never read).
* :func:`gather_last_real` extracts the "user state" vector under either
  padding side.  Under left padding this is simply ``hidden[:, -1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TransformerEncoderConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float = 0.0
    causal: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


def build_attention_mask(attention_mask: torch.Tensor, causal: bool) -> torch.Tensor:
    """``(B, L)`` bool key mask -> ``(B, 1, L, L)`` bool (True = may attend)."""
    b, L = attention_mask.shape
    key_valid = attention_mask[:, None, None, :]  # (B, 1, 1, L)
    if causal:
        tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=attention_mask.device))
        allowed = key_valid & tril[None, None]
    else:
        allowed = key_valid.expand(b, 1, L, L)
    eye = torch.eye(L, dtype=torch.bool, device=attention_mask.device)[None, None]
    out: torch.Tensor = allowed | eye
    return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, L, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, L, dh)
        k = k.view(b, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, L, self.n_heads, self.d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        merged = attn.transpose(1, 2).reshape(b, L, d)
        out: torch.Tensor = self.out(merged)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class PreLNTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), attn_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of Pre-LN blocks with a final LayerNorm.  ``causal`` is fixed at construction."""

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                PreLNTransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = build_attention_mask(attention_mask, self.config.causal)
        for block in self.blocks:
            x = block(x, attn_mask)
        out: torch.Tensor = self.final_ln(x)
        return out


def gather_last_real(
    hidden: torch.Tensor, attention_mask: torch.Tensor, padding_side: Literal["left", "right"]
) -> torch.Tensor:
    """``(B, L, D)`` -> ``(B, D)`` at the most recent real event."""
    if padding_side == "left":
        return hidden[:, -1, :]
    if padding_side != "right":
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    last = (attention_mask.to(torch.int64).sum(dim=-1) - 1).clamp(min=0)  # (B,)
    idx = last.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    return hidden.gather(1, idx).squeeze(1)
