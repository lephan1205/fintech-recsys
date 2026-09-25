"""Action-aware multi-modal credit token embedding.

.. math::

    E_t = W_{item}(i_t) + W_{action}(a_t) + W_{time}(\\ln(1 + \\Delta t_t)) + P_t

* ``W_item`` has ``padding_idx=0`` so the PAD / SCORE_CHANGE row is a frozen zero.
  A ``SCORE_CHANGE`` token is therefore represented purely by its action, time
  and position embeddings, while a PAD token is exactly zero (all four terms
  are zeroed by the mask).
* Position ids are counted over **real** tokens
  (``(mask.cumsum(-1) - 1).clamp(min=0)``) so that the same events, padded on
  either side, receive identical positional embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from recsys.data.schema import NUM_ACTIONS, PAD_ACTION_ID, PAD_ITEM_ID


@dataclass(frozen=True)
class MultiModalEmbeddingConfig:
    num_items: int
    d_model: int
    max_len: int
    num_actions: int = NUM_ACTIONS
    dropout: float = 0.0


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``(B, L)`` 0-based position among real tokens; PAD slots are clamped to 0."""
    pos: torch.Tensor = (attention_mask.to(torch.int64).cumsum(dim=-1) - 1).clamp(min=0)
    return pos


class MultiModalEmbedding(nn.Module):
    def __init__(self, config: MultiModalEmbeddingConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.item_emb = nn.Embedding(config.num_items + 1, d, padding_idx=PAD_ITEM_ID)
        self.action_emb = nn.Embedding(config.num_actions, d, padding_idx=PAD_ACTION_ID)
        self.time_proj = nn.Linear(1, d)
        self.pos_emb = nn.Embedding(config.max_len, d)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.action_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD_ITEM_ID].zero_()
            self.action_emb.weight[PAD_ACTION_ID].zero_()

    def forward(
        self,
        item_ids: torch.Tensor,
        action_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if item_ids.shape[1] > self.config.max_len:
            raise ValueError(f"sequence length {item_ids.shape[1]} > max_len {self.config.max_len}")
        log_dt = torch.log1p(time_deltas.clamp(min=0.0)).unsqueeze(-1)  # (B, L, 1)
        pos = position_ids_from_mask(attention_mask)
        e: torch.Tensor = (
            self.item_emb(item_ids)
            + self.action_emb(action_ids)
            + self.time_proj(log_dt)
            + self.pos_emb(pos)
        )
        e = e * attention_mask.unsqueeze(-1).to(e.dtype)
        out: torch.Tensor = self.dropout(e)
        return out
