"""HSTU scoring backbone with M-FALCON batched candidate scoring.

Token embedding::

    history   E_t = W_item(i_t) + W_action(a_t) + W_time(bucket(Δt_t))
    candidate E_c = W_item(c)   + W_candtype     + W_time(bucket(0))

(shared item table, ``padding_idx = 0``).  Absolute positions are not embedded: the
HSTU layers carry learned *relative* position and time biases.  Candidate tokens sit
at "now": position ``last_history_position + 1`` and time ``t_last``.

``score_candidates`` runs one forward of length ``L + K`` and returns ``h_user`` (the
user state at the most recent real event) and ``h_cand (B, K, d)``; ``fusion`` builds
``[h_cand ‖ h_user ‖ h_cand ⊙ h_user ‖ tabular]`` for the PLE funnel.
"""

from __future__ import annotations

import torch
from torch import nn

from recsys.data.collator import SequenceBatch
from recsys.data.schema import PAD_ACTION_ID, PAD_ITEM_ID
from recsys.layers.hstu import (
    HSTUConfig,
    HSTUEncoded,
    HSTULayer,
    bucketize_time,
    build_mfalcon_mask,
)
from recsys.layers.multi_modal_embedding import position_ids_from_mask
from recsys.layers.transformer_blocks import gather_last_real


class HSTUBackbone(nn.Module):
    def __init__(self, config: HSTUConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.item_emb = nn.Embedding(config.num_items + 1, d, padding_idx=PAD_ITEM_ID)
        self.action_emb = nn.Embedding(config.num_actions, d, padding_idx=PAD_ACTION_ID)
        self.time_emb = nn.Embedding(config.num_time_buckets, d)
        self.cand_type = nn.Parameter(torch.zeros(d))
        self.layers = nn.ModuleList([HSTULayer(config) for _ in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        for emb in (self.item_emb, self.action_emb, self.time_emb):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.cand_type, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[PAD_ITEM_ID].zero_()
            self.action_emb.weight[PAD_ACTION_ID].zero_()

    # ------------------------------------------------------------------ pieces
    def _check(self, seq: SequenceBatch, candidate_ids: torch.Tensor) -> None:
        if seq.padding_side != self.config.padding_side:
            raise ValueError(
                f"batch is {seq.padding_side}-padded but the model expects "
                f"{self.config.padding_side}"
            )
        if seq.seq_len > self.config.max_len:
            raise ValueError(f"history length {seq.seq_len} > max_len {self.config.max_len}")
        if candidate_ids.shape[1] > self.config.max_candidates:
            raise ValueError("more candidates than max_candidates")
        if candidate_ids.shape[0] != seq.batch_size:
            raise ValueError("candidate batch size must match the sequence batch")

    def embed(self, seq: SequenceBatch, candidate_ids: torch.Tensor) -> torch.Tensor:
        """``(B, L+K, d)`` token embeddings; PAD history slots are exactly zero."""
        cfg = self.config
        dt_bucket = bucketize_time(seq.time_deltas, cfg.num_time_buckets, cfg.max_time_days)
        hist: torch.Tensor = (
            self.item_emb(seq.item_ids) + self.action_emb(seq.action_ids) + self.time_emb(dt_bucket)
        )
        hist = hist * seq.attention_mask.unsqueeze(-1).to(hist.dtype)
        zero_bucket = torch.zeros_like(candidate_ids)
        cand: torch.Tensor = (
            self.item_emb(candidate_ids) + self.cand_type + self.time_emb(zero_bucket)
        )
        out: torch.Tensor = self.dropout(torch.cat([hist, cand], dim=1))
        return out

    def relative_indices(
        self, seq: SequenceBatch, num_candidates: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rel_pos, time_bucket)``, both ``(B, S, S)`` int64, for ``S = L + K``."""
        cfg = self.config
        mask = seq.attention_mask
        pos_hist = position_ids_from_mask(mask)  # (B, L), real-token positions
        last_pos = pos_hist.max(dim=1).values  # (B,)
        pos_cand = (last_pos + 1).unsqueeze(1).expand(-1, num_candidates)
        pos = torch.cat([pos_hist, pos_cand], dim=1)  # (B, S)
        rel = (pos.unsqueeze(2) - pos.unsqueeze(1)).clamp(-cfg.max_rel_pos, cfg.max_rel_pos)
        rel_idx = rel + cfg.max_rel_pos

        t_abs = torch.cumsum(seq.time_deltas * mask.to(seq.time_deltas.dtype), dim=1)  # (B, L)
        t_last = gather_last_real(t_abs.unsqueeze(-1), mask, seq.padding_side).squeeze(-1)
        t_cand = t_last.unsqueeze(1).expand(-1, num_candidates)
        t_all = torch.cat([t_abs, t_cand], dim=1)  # (B, S)
        dt = (t_all.unsqueeze(2) - t_all.unsqueeze(1)).abs()
        return rel_idx, bucketize_time(dt, cfg.num_time_buckets, cfg.max_time_days)

    def encode(self, seq: SequenceBatch, candidate_ids: torch.Tensor) -> torch.Tensor:
        """``(B, L+K, d)`` hidden states after all HSTU layers."""
        k = candidate_ids.shape[1]
        x = self.embed(seq, candidate_ids)
        attn_mask = build_mfalcon_mask(seq.attention_mask, k)
        rel_idx, time_bucket = self.relative_indices(seq, k)
        for layer in self.layers:
            x = layer(x, attn_mask, rel_idx, time_bucket)
        out: torch.Tensor = self.final_ln(x)
        return out

    # ---------------------------------------------------------------- scoring
    def score_candidates(self, seq: SequenceBatch, candidate_ids: torch.Tensor) -> HSTUEncoded:
        """One batched pass: ``(B, L)`` history + ``(B, K)`` candidates -> user + candidate reps."""
        self._check(seq, candidate_ids)
        ell = seq.seq_len
        hidden = self.encode(seq, candidate_ids)
        h_user = gather_last_real(hidden[:, :ell], seq.attention_mask, seq.padding_side)
        return HSTUEncoded(hidden=hidden, h_user=h_user, h_cand=hidden[:, ell:])

    def fusion(
        self, h_user: torch.Tensor, h_cand: torch.Tensor, tabular: torch.Tensor
    ) -> torch.Tensor:
        """``[h_cand ‖ h_user ‖ h_cand ⊙ h_user ‖ tabular]`` -> ``(B, K, fusion_dim)``."""
        k = h_cand.shape[1]
        hu = h_user.unsqueeze(1).expand(-1, k, -1)
        out: torch.Tensor = torch.cat([h_cand, hu, h_cand * hu, tabular], dim=-1)
        if out.shape[-1] != self.config.fusion_dim:
            raise ValueError(f"fusion dim {out.shape[-1]} != config {self.config.fusion_dim}")
        return out
