"""TIGER: generative retrieval over Semantic IDs with underwriting-constrained decoding.

Design choice: **decoder-only**.  History tokens and target tokens live in the same
Semantic-ID vocabulary, so a single causal stack over

    [TIER, STATE, <history item tokens ...>, c_1, ..., c_T]

needs one embedding table, reuses :class:`~recsys.layers.transformer_blocks.TransformerEncoder`,
and makes beam search a plain next-token loop.  The two *user-context tokens* (credit
tier, state) sit physically first, so under left padding they precede the PAD run;
PAD keys are masked and positions are counted over real tokens, so every real token
can attend to them.

Training signal: next-Semantic-ID cross-entropy.  The target item's ``T`` codes are
predicted from the last real history token onward (exactly the pattern used at
inference), and — with ``train_all_positions`` — every history item's codes are also
predicted from its prefix, which turns one record per user into ``~L`` training
examples.  Each level's cross-entropy is restricted to that level's token range,
matching trie-masked decoding.

Compliance: at every decoding step the prefix trie yields, for the user's eligible-item
mask, the set of allowed next codes; everything else is set to ``-inf`` before the
top-k, so generated IDs are provably eligible.  Beam search is fully tensorized:
beams live in a ``(B, W, l)`` tensor and each level is one forward pass over ``B·W`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from recsys.data.collator import PaddingSide
from recsys.data.schema import (
    NUM_ACTIONS,
    NUM_STATES,
    NUM_TIERS,
    PAD_ACTION_ID,
    SCORE_CHANGE_ITEM_ID,
)
from recsys.layers.multi_modal_embedding import position_ids_from_mask
from recsys.layers.prefix_trie import SemanticIdTrie
from recsys.layers.transformer_blocks import TransformerEncoder, TransformerEncoderConfig

PAD_TOK = 0
SCORE_CHANGE_TOK = 1
TIER_TOK_OFFSET = 2
STATE_TOK_OFFSET = TIER_TOK_OFFSET + NUM_TIERS
NUM_SPECIAL_TOKENS = STATE_TOK_OFFSET + NUM_STATES
NUM_CONTEXT_TOKENS = 2


@dataclass(frozen=True)
class TIGERConfig:
    num_items: int
    level_sizes: tuple[int, ...]  # per-level codebook sizes, dedup level included
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 128
    max_history_items: int = 20
    num_actions: int = NUM_ACTIONS
    dropout: float = 0.0
    padding_side: PaddingSide = "left"
    train_all_positions: bool = True
    beam_size: int = 100

    @property
    def num_levels(self) -> int:
        return len(self.level_sizes)

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL_TOKENS + sum(self.level_sizes)

    @property
    def level_offsets(self) -> tuple[int, ...]:
        offs, acc = [], NUM_SPECIAL_TOKENS
        for s in self.level_sizes:
            offs.append(acc)
            acc += s
        return tuple(offs)

    @property
    def max_seq_len(self) -> int:
        return NUM_CONTEXT_TOKENS + self.max_history_items * self.num_levels + self.num_levels


class SemanticIdTokenizer:
    """Maps item ids <-> Semantic-ID token sequences and builds context tokens."""

    def __init__(
        self, item_codes: npt.NDArray[np.int64] | torch.Tensor, config: TIGERConfig
    ) -> None:
        codes = torch.as_tensor(np.asarray(item_codes), dtype=torch.int64)
        if codes.shape != (config.num_items + 1, config.num_levels):
            expected = (config.num_items + 1, config.num_levels)
            raise ValueError(f"item_codes must have shape {expected}, got {tuple(codes.shape)}")
        for level, size in enumerate(config.level_sizes):
            if int(codes[1:, level].max()) >= size:
                raise ValueError(f"codes at level {level} exceed level size {size}")
        self.config = config
        self.item_codes = codes
        self.offsets = torch.tensor(config.level_offsets, dtype=torch.int64)

    def token_of(self, level: int, code: int) -> int:
        return int(self.offsets[level]) + code

    def codes_to_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """``(..., l)`` codes for levels ``0..l-1`` -> ``(..., l)`` token ids."""
        n = codes.shape[-1]
        return codes + self.offsets[:n].to(codes.device)

    def tokens_to_codes(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[-1]
        return tokens - self.offsets[:n].to(tokens.device)

    def item_tokens(self, item_ids: torch.Tensor) -> torch.Tensor:
        """``(...,)`` item ids -> ``(..., T)`` tokens (SCORE_CHANGE -> repeated special token)."""
        toks = self.codes_to_tokens(self.item_codes.to(item_ids.device)[item_ids])
        sc = (item_ids == SCORE_CHANGE_ITEM_ID).unsqueeze(-1)
        return torch.where(sc, torch.full_like(toks, SCORE_CHANGE_TOK), toks)

    @staticmethod
    def context_tokens(tier_idx: torch.Tensor, state_idx: torch.Tensor) -> torch.Tensor:
        """``(B,)`` tier / state indices -> ``(B, 2)`` context tokens."""
        return torch.stack([TIER_TOK_OFFSET + tier_idx, STATE_TOK_OFFSET + state_idx], dim=1)

    def encode_history(
        self,
        item_ids: torch.Tensor,
        action_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tier_idx: torch.Tensor,
        state_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(B, L)`` events -> ``(B, 2 + L*T)`` tokens / actions / mask.

        Context tokens come first (mask True, action PAD); the history keeps its
        padding side.
        """
        if item_ids.shape[1] > self.config.max_history_items:
            raise ValueError("history longer than max_history_items")
        t = self.config.num_levels
        toks = self.item_tokens(item_ids)  # (B, L, T)
        mask = attention_mask.unsqueeze(-1).expand(-1, -1, t)
        toks = torch.where(mask, toks, torch.full_like(toks, PAD_TOK))
        acts = action_ids.unsqueeze(-1).expand(-1, -1, t)
        acts = torch.where(mask, acts, torch.full_like(acts, PAD_ACTION_ID))
        b = item_ids.shape[0]
        ctx = self.context_tokens(tier_idx, state_idx).to(item_ids.device)
        ctx_mask = torch.ones((b, NUM_CONTEXT_TOKENS), dtype=torch.bool, device=item_ids.device)
        ctx_act = torch.full((b, NUM_CONTEXT_TOKENS), PAD_ACTION_ID, dtype=torch.int64)
        return (
            torch.cat([ctx, toks.reshape(b, -1)], dim=1),
            torch.cat([ctx_act.to(item_ids.device), acts.reshape(b, -1)], dim=1),
            torch.cat([ctx_mask, mask.reshape(b, -1)], dim=1),
        )


@dataclass
class TIGEROutput:
    hidden: torch.Tensor  # (B, S, D)
    logits: torch.Tensor  # (B, S, V)


@dataclass
class TIGERLoss:
    total: torch.Tensor
    target: torch.Tensor  # next-SID CE on the target item (inference pattern)
    history: torch.Tensor  # next-SID CE on every history position (0 if disabled)
    num_history_predictions: int


@dataclass
class TIGERGeneration:
    item_ids: torch.Tensor  # (B, beam) int64, -1 for dead beams
    codes: torch.Tensor  # (B, beam, T) int64, -1 for dead beams
    log_probs: torch.Tensor  # (B, beam) float32, -inf for dead beams


def _append_tokens(
    tokens: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    new_tokens: torch.Tensor,
    padding_side: PaddingSide,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append ``new_tokens (B, t)`` as real tokens after the last real token.

    Returns ``(tokens, actions, mask, positions)`` where ``positions (B, t)`` are the
    indices of the appended tokens.  Under left padding this is a plain concat.
    ``t`` may be 0, in which case the inputs are returned with empty positions.
    """
    b, t = new_tokens.shape
    device = tokens.device
    new_actions = torch.zeros_like(new_tokens)  # appended decoder tokens carry no action
    if padding_side == "left":
        pos = torch.arange(tokens.shape[1], tokens.shape[1] + t, device=device)
        pos = pos.unsqueeze(0).expand(b, -1)
        return (
            torch.cat([tokens, new_tokens], dim=1),
            torch.cat([actions, new_actions], dim=1),
            torch.cat([mask, torch.ones_like(new_tokens, dtype=torch.bool)], dim=1),
            pos,
        )
    lengths = mask.to(torch.int64).sum(dim=1)  # (B,)
    pos = lengths.unsqueeze(1) + torch.arange(t, device=device).unsqueeze(0)  # (B, t)
    out_tokens = torch.cat(
        [tokens, torch.full((b, t), PAD_TOK, dtype=tokens.dtype, device=device)], 1
    )
    out_actions = torch.cat([actions, torch.zeros((b, t), dtype=actions.dtype, device=device)], 1)
    out_mask = torch.cat([mask, torch.zeros((b, t), dtype=torch.bool, device=device)], dim=1)
    if t > 0:
        out_tokens.scatter_(1, pos, new_tokens)
        out_actions.scatter_(1, pos, new_actions)
        out_mask.scatter_(1, pos, torch.ones_like(new_tokens, dtype=torch.bool))
    return out_tokens, out_actions, out_mask, pos


def _last_real_position(mask: torch.Tensor, padding_side: PaddingSide) -> torch.Tensor:
    """``(B,)`` index of the most recent real token."""
    if padding_side == "left":
        return torch.full(
            (mask.shape[0],), mask.shape[1] - 1, dtype=torch.int64, device=mask.device
        )
    return (mask.to(torch.int64).sum(dim=1) - 1).clamp(min=0)


class TIGER(nn.Module):
    def __init__(self, config: TIGERConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.tok_emb = nn.Embedding(config.vocab_size, d, padding_idx=PAD_TOK)
        self.action_emb = nn.Embedding(config.num_actions, d, padding_idx=PAD_ACTION_ID)
        self.pos_emb = nn.Embedding(config.max_seq_len, d)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = TransformerEncoder(
            TransformerEncoderConfig(
                d_model=d,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                dropout=config.dropout,
                causal=True,
            )
        )
        self.head = nn.Linear(d, config.vocab_size)
        for emb in (self.tok_emb, self.action_emb, self.pos_emb):
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[PAD_TOK].zero_()
            self.action_emb.weight[PAD_ACTION_ID].zero_()

    # ------------------------------------------------------------------ pieces
    def embed(
        self, tokens: torch.Tensor, token_actions: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if tokens.shape[1] > self.config.max_seq_len:
            raise ValueError(f"sequence length {tokens.shape[1]} > max_seq_len")
        pos = position_ids_from_mask(attention_mask)
        e: torch.Tensor = self.tok_emb(tokens) + self.action_emb(token_actions) + self.pos_emb(pos)
        e = e * attention_mask.unsqueeze(-1).to(e.dtype)
        out: torch.Tensor = self.dropout(e)
        return out

    def encode(self, emb: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.encoder(emb, attention_mask)
        return out

    def forward(
        self, tokens: torch.Tensor, token_actions: torch.Tensor, attention_mask: torch.Tensor
    ) -> TIGEROutput:
        hidden = self.encode(self.embed(tokens, token_actions, attention_mask), attention_mask)
        return TIGEROutput(hidden=hidden, logits=self.head(hidden))

    def _level_log_probs(self, step_logits: torch.Tensor, level: int) -> torch.Tensor:
        """``(M, V)`` logits -> ``(M, level_size)`` log-probs restricted to the level's range."""
        off, size = self.config.level_offsets[level], self.config.level_sizes[level]
        return F.log_softmax(step_logits[:, off : off + size], dim=-1)

    def _level_ce(self, step_logits: torch.Tensor, codes: torch.Tensor, level: int) -> torch.Tensor:
        """Per-row cross-entropy ``(M,)`` at ``level`` (restricted to the level's token range)."""
        lp = self._level_log_probs(step_logits, level)
        return -lp.gather(1, codes.unsqueeze(1)).squeeze(1)

    # ---------------------------------------------------------------- training
    def next_sid_loss(
        self,
        hist_tokens: torch.Tensor,
        hist_actions: torch.Tensor,
        hist_mask: torch.Tensor,
        target_codes: torch.Tensor,
        tokenizer: SemanticIdTokenizer,
    ) -> TIGERLoss:
        """Next-Semantic-ID cross-entropy.

        *Target term*: ``c_1`` is predicted from the last real history token and ``c_k``
        from ``c_{k-1}`` (appended with action PAD), exactly as decoding runs.
        *History term* (``train_all_positions``): for every history slot ``j+1`` that is
        a real product event whose predecessor slot ``j`` is real, its ``T`` codes are
        predicted from slot ``j``'s last token and its own preceding tokens.
        """
        cfg = self.config
        b, t = target_codes.shape
        tgt_tokens = tokenizer.codes_to_tokens(target_codes)  # (B, T)
        dec_in = tgt_tokens[:, :-1]  # (B, T-1)
        tokens, actions, mask, pos = _append_tokens(
            hist_tokens, hist_actions, hist_mask, dec_in, cfg.padding_side
        )
        logits = self.forward(tokens, actions, mask).logits  # (B, S, V)
        start = _last_real_position(hist_mask, cfg.padding_side)  # (B,)
        pred_pos = torch.cat([start.unsqueeze(1), pos], dim=1)  # (B, T)
        idx = pred_pos.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        step_logits = logits.gather(1, idx)  # (B, T, V)
        target_terms = [self._level_ce(step_logits[:, k], target_codes[:, k], k) for k in range(t)]
        target_loss = torch.stack(target_terms, dim=1).mean()

        history_loss = torch.zeros((), device=logits.device)
        n_hist = 0
        if cfg.train_all_positions:
            s_h = hist_tokens.shape[1] - NUM_CONTEXT_TOKENS
            n_slots = s_h // t
            hist_logits = logits[:, NUM_CONTEXT_TOKENS : NUM_CONTEXT_TOKENS + s_h]  # (B, L*T, V)
            hist_tok = hist_tokens[:, NUM_CONTEXT_TOKENS:].reshape(b, n_slots, t)
            slot_real = hist_mask[:, NUM_CONTEXT_TOKENS:].reshape(b, n_slots, t)[:, :, 0]
            slot_product = slot_real & (hist_tok[:, :, 0] != SCORE_CHANGE_TOK)
            valid = slot_real[:, :-1] & slot_product[:, 1:]  # (B, L-1): predict slot j+1
            if bool(valid.any()):
                per_level = []
                hist_logits_slots = hist_logits.reshape(b, n_slots, t, -1)
                codes = tokenizer.tokens_to_codes(hist_tok)  # (B, L, T) codes (SC rows garbage)
                for k in range(t):
                    if k == 0:
                        pred = hist_logits_slots[:, :-1, t - 1]  # last token of slot j
                    else:
                        pred = hist_logits_slots[:, 1:, k - 1]  # token k-1 of slot j+1
                    tgt = codes[:, 1:, k].clamp(min=0)
                    ce = self._level_ce(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), k)
                    per_level.append(ce.reshape(b, n_slots - 1))
                ce_all = torch.stack(per_level, dim=-1).mean(dim=-1)  # (B, L-1)
                history_loss = (ce_all * valid).sum() / valid.sum()
                n_hist = int(valid.sum())
        total = target_loss + history_loss
        return TIGERLoss(
            total=total, target=target_loss, history=history_loss, num_history_predictions=n_hist
        )

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(
        self,
        hist_tokens: torch.Tensor,
        hist_actions: torch.Tensor,
        hist_mask: torch.Tensor,
        trie: SemanticIdTrie,
        tokenizer: SemanticIdTokenizer,
        allowed_items: torch.Tensor,
        beam_size: int | None = None,
    ) -> TIGERGeneration:
        """Tensorized, trie-constrained beam search.

        ``allowed_items (B, N+1) bool`` is the user's serving-eligibility mask (row 0
        False).  Every returned item is an allowed leaf of ``trie``; dead beams are
        ``-1`` with ``-inf`` log-prob, so ``|C_u| <= beam_size`` (beam yield).
        """
        cfg = self.config
        w = cfg.beam_size if beam_size is None else beam_size
        b = hist_tokens.shape[0]
        t = cfg.num_levels
        device = hist_tokens.device
        if allowed_items.shape != (b, cfg.num_items + 1):
            raise ValueError("allowed_items must be (B, num_items + 1)")

        # level 0: one forward over the plain histories
        logits = self.forward(hist_tokens, hist_actions, hist_mask).logits
        start = _last_real_position(hist_mask, cfg.padding_side)
        step = logits[torch.arange(b, device=device), start]  # (B, V)
        lp = self._level_log_probs(step, 0)  # (B, s0)
        empty = torch.zeros((b, 0), dtype=torch.int64, device=device)
        child = trie.allowed_children_batch(empty, allowed_items)
        lp = lp.masked_fill(~child, float("-inf"))
        k0 = min(w, lp.shape[1])
        scores, idx = lp.topk(k0, dim=1)  # (B, k0)
        beams = idx.unsqueeze(-1)  # (B, k0, 1)

        for level in range(1, t):
            wc = beams.shape[1]
            flat_beams = beams.reshape(b * wc, level)
            dec = tokenizer.codes_to_tokens(flat_beams)
            tokens, actions, mask, pos = _append_tokens(
                hist_tokens.repeat_interleave(wc, dim=0),
                hist_actions.repeat_interleave(wc, dim=0),
                hist_mask.repeat_interleave(wc, dim=0),
                dec,
                cfg.padding_side,
            )
            out = self.forward(tokens, actions, mask).logits  # (B*wc, S, V)
            last = out.gather(1, pos[:, -1:].unsqueeze(-1).expand(-1, -1, out.shape[-1]))
            lp = self._level_log_probs(last.squeeze(1), level)  # (B*wc, size)
            child = trie.allowed_children_batch(
                flat_beams, allowed_items.repeat_interleave(wc, dim=0)
            )
            lp = lp.masked_fill(~child, float("-inf"))
            size = lp.shape[1]
            cand = (scores.reshape(b * wc, 1) + lp).reshape(b, wc * size)
            k = min(w, wc * size)
            scores, flat_idx = cand.topk(k, dim=1)  # (B, k)
            parent = flat_idx // size
            code = flat_idx % size
            beams = torch.cat(
                [beams.gather(1, parent.unsqueeze(-1).expand(-1, -1, level)), code.unsqueeze(-1)],
                dim=-1,
            )

        alive = torch.isfinite(scores)
        items = trie.items_for_batch(beams)  # (B, k)
        items = torch.where(alive, items, torch.full_like(items, -1))
        codes_out = torch.where(alive.unsqueeze(-1), beams, torch.full_like(beams, -1))
        # pad to the requested beam width
        kk = items.shape[1]
        if kk < w:
            pad_i = torch.full((b, w - kk), -1, dtype=torch.int64, device=device)
            pad_c = torch.full((b, w - kk, t), -1, dtype=torch.int64, device=device)
            pad_s = torch.full((b, w - kk), float("-inf"), device=device)
            items = torch.cat([items, pad_i], dim=1)
            codes_out = torch.cat([codes_out, pad_c], dim=1)
            scores = torch.cat([scores, pad_s], dim=1)
        return TIGERGeneration(item_ids=items, codes=codes_out, log_probs=scores)
