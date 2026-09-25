"""Sequence collator with a **left-padding** default.

Why left padding?
-----------------
With left padding the most recent real event always sits at index ``L-1``.
Every sequential model can therefore read its "user state" as
``hidden[:, -1, :]`` with no per-row gather, and causal recency indexing
(``position L-1`` is "now", ``L-2`` is "one event ago", ...) is aligned
across the whole batch.  Right padding is still supported through
``padding_side="right"`` and :func:`recsys.layers.transformer_blocks.gather_last_real`.

Masking rule
------------
``attention_mask = action_ids != PAD_ACTION_ID``.  ``SCORE_CHANGE`` events
carry ``item_id == 0`` (they are user-level, not product-level), so a mask
derived from ``item_ids`` would silently drop them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal

import torch

from recsys.data.schema import (
    PAD_ACTION_ID,
    PAD_ITEM_ID,
    POSITIVE_ACTIONS,
    ActionType,
    InteractionRecord,
)

PaddingSide = Literal["left", "right"]


@dataclass
class SequenceBatch:
    """Fixed-shape batch of user timelines.  All sequence tensors are ``(B, L)``."""

    item_ids: torch.Tensor  # (B, L) int64, PAD = 0
    action_ids: torch.Tensor  # (B, L) int64, PAD = 0
    time_deltas: torch.Tensor  # (B, L) float32, days since previous kept event, PAD = 0
    attention_mask: torch.Tensor  # (B, L) bool, == action_ids != 0
    lengths: torch.Tensor  # (B,) int64, number of real events
    target_item_ids: torch.Tensor  # (B,) int64
    target_action_ids: torch.Tensor  # (B,) int64
    future_item_ids: torch.Tensor  # (B, F) int64, 0-padded
    future_mask: torch.Tensor  # (B, F) bool
    user_indices: torch.Tensor  # (B,) int64
    padding_side: PaddingSide = "left"

    @property
    def batch_size(self) -> int:
        return int(self.item_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.item_ids.shape[1])

    @property
    def target_is_pending(self) -> torch.Tensor:
        """(B,) bool: the target is an application whose decision is not yet observed."""
        out: torch.Tensor = self.target_action_ids == int(ActionType.APPLY_PENDING)
        return out

    def to(self, device: torch.device | str) -> SequenceBatch:
        kwargs = {
            f.name: (
                getattr(self, f.name).to(device)
                if isinstance(getattr(self, f.name), torch.Tensor)
                else getattr(self, f.name)
            )
            for f in fields(self)
        }
        return SequenceBatch(**kwargs)


class SequenceCollator:
    """Turn :class:`InteractionRecord` objects into a padded :class:`SequenceBatch`."""

    def __init__(
        self, max_len: int, padding_side: PaddingSide = "left", max_future: int = 16
    ) -> None:
        if max_len < 1 or max_future < 1:
            raise ValueError("max_len and max_future must be >= 1")
        if padding_side not in ("left", "right"):
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
        self.max_len = max_len
        self.padding_side: PaddingSide = padding_side
        self.max_future = max_future

    def __call__(
        self,
        records: Sequence[InteractionRecord],
        cutoff_days: Sequence[float] | None = None,
    ) -> SequenceBatch:
        """Collate ``records``; ``cutoff_days[i]`` (optional) drops every history event of
        record ``i`` with ``timestamp > cutoff`` so a slate only sees what preceded it.
        A record whose history is emptied by the cutoff keeps its single earliest event,
        so every row has at least one real token (the mask is never all-False)."""
        b, L, F = len(records), self.max_len, self.max_future
        if cutoff_days is not None and len(cutoff_days) != b:
            raise ValueError("cutoff_days must have one entry per record")
        item_ids = torch.full((b, L), PAD_ITEM_ID, dtype=torch.int64)
        action_ids = torch.full((b, L), PAD_ACTION_ID, dtype=torch.int64)
        time_deltas = torch.zeros((b, L), dtype=torch.float32)
        lengths = torch.zeros(b, dtype=torch.int64)
        target_items = torch.zeros(b, dtype=torch.int64)
        target_actions = torch.zeros(b, dtype=torch.int64)
        future_items = torch.full((b, F), PAD_ITEM_ID, dtype=torch.int64)
        future_mask = torch.zeros((b, F), dtype=torch.bool)
        user_indices = torch.zeros(b, dtype=torch.int64)

        for i, rec in enumerate(records):
            history = rec.history
            if cutoff_days is not None:
                kept = tuple(e for e in history if e.timestamp <= cutoff_days[i])
                history = kept if kept else history[:1]
            events = history[-L:]  # keep the most recent L events
            n = len(events)
            lengths[i] = n
            items = [e.item_id for e in events]
            actions = [int(e.action) for e in events]
            ts = [e.timestamp for e in events]
            deltas = [0.0] + [max(ts[j] - ts[j - 1], 0.0) for j in range(1, n)]
            start = L - n if self.padding_side == "left" else 0
            item_ids[i, start : start + n] = torch.tensor(items, dtype=torch.int64)
            action_ids[i, start : start + n] = torch.tensor(actions, dtype=torch.int64)
            time_deltas[i, start : start + n] = torch.tensor(deltas, dtype=torch.float32)

            target_items[i] = rec.target.item_id
            target_actions[i] = int(rec.target.action)
            user_indices[i] = rec.user_index
            fut = [e.item_id for e in rec.future_window if e.action in POSITIVE_ACTIONS][:F]
            if fut:
                future_items[i, : len(fut)] = torch.tensor(fut, dtype=torch.int64)
                future_mask[i, : len(fut)] = True

        return SequenceBatch(
            item_ids=item_ids,
            action_ids=action_ids,
            time_deltas=time_deltas,
            attention_mask=action_ids != PAD_ACTION_ID,
            lengths=lengths,
            target_item_ids=target_items,
            target_action_ids=target_actions,
            future_item_ids=future_items,
            future_mask=future_mask,
            user_indices=user_indices,
            padding_side=self.padding_side,
        )
