"""Prefix trie over Semantic IDs with per-request eligibility masking.

Generative retrieval decodes an item one code at a time.  Without constraints the
decoder can emit a tuple that (a) is not a real item or (b) is an item the user is
not eligible for.  The trie is built **once** from the full catalog; per request the
caller supplies the user's eligible-item mask and :meth:`allowed_children_batch`
returns, for a batch of prefixes, exactly the codes whose continuation leads to an
*eligible* leaf.  Everything else is set to ``-inf`` before the top-k, so generated
IDs are eligible by construction.  This turns underwriting compliance into a hard
constraint on the logits rather than a post-filter (the post-filter still exists
as the second enforcement point).

The dict-based walk (``allowed_next`` / ``item_for``) is kept for tests and
explanations; the vectorized methods are what serving uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch


class _Node:
    __slots__ = ("children", "item_id")

    def __init__(self) -> None:
        self.children: dict[int, _Node] = {}
        self.item_id: int | None = None


class SemanticIdTrie:
    def __init__(self, level_sizes: Sequence[int]) -> None:
        self.level_sizes = tuple(int(s) for s in level_sizes)
        self.root = _Node()
        self._num_leaves = 0
        self.codes: torch.Tensor | None = None  # (N+1, T) int64, full catalog (row 0 = PAD)
        self._keys: torch.Tensor | None = None  # sorted flat keys of inserted rows
        self._key_items: torch.Tensor | None = None

    # ------------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        codes: npt.NDArray[np.int64] | torch.Tensor,
        item_ids: npt.NDArray[np.int64] | torch.Tensor,
        level_sizes: Sequence[int],
        keep: npt.NDArray[np.bool_] | torch.Tensor | None = None,
    ) -> SemanticIdTrie:
        """Insert row ``i`` iff ``keep[i]`` (defaults to all rows).

        ``codes`` is ``(N, T)`` for ``item_ids`` ``(N,)``.  When the rows are the
        whole catalog (``item_ids == 1..N``), the vectorized per-request masking
        becomes available.
        """
        codes_np = np.asarray(codes.cpu() if isinstance(codes, torch.Tensor) else codes)
        ids_np = np.asarray(item_ids.cpu() if isinstance(item_ids, torch.Tensor) else item_ids)
        if codes_np.shape[1] != len(level_sizes):
            raise ValueError("codes must have one column per level")
        keep_np = (
            np.ones(codes_np.shape[0], dtype=bool)
            if keep is None
            else np.asarray(keep.cpu() if isinstance(keep, torch.Tensor) else keep, dtype=bool)
        )
        trie = cls(level_sizes)
        for row, item, k in zip(codes_np, ids_np, keep_np, strict=True):
            if k:
                trie.insert(tuple(int(c) for c in row), int(item))
        n = int(ids_np.max()) if ids_np.size else 0
        full = torch.zeros((n + 1, len(level_sizes)), dtype=torch.int64)
        full[torch.as_tensor(ids_np, dtype=torch.int64)] = torch.as_tensor(
            codes_np, dtype=torch.int64
        )
        trie.codes = full
        kept_ids = torch.as_tensor(ids_np[keep_np], dtype=torch.int64)
        keys = trie.flat_keys(full[kept_ids])
        order = torch.argsort(keys)
        trie._keys = keys[order]
        trie._key_items = kept_ids[order]
        return trie

    def insert(self, codes: tuple[int, ...], item_id: int) -> None:
        if len(codes) != len(self.level_sizes):
            raise ValueError("codes length must equal the number of levels")
        node = self.root
        for level, c in enumerate(codes):
            if not 0 <= c < self.level_sizes[level]:
                raise ValueError(f"code {c} out of range at level {level}")
            node = node.children.setdefault(c, _Node())
        if node.item_id is None:
            self._num_leaves += 1
        node.item_id = item_id

    # ------------------------------------------------------------- dict walk
    def _walk(self, prefix: Sequence[int]) -> _Node | None:
        node = self.root
        for c in prefix:
            nxt = node.children.get(int(c))
            if nxt is None:
                return None
            node = nxt
        return node

    def contains_prefix(self, prefix: Sequence[int]) -> bool:
        return self._walk(prefix) is not None

    def allowed_next(self, prefix: Sequence[int]) -> npt.NDArray[np.bool_]:
        """Boolean mask over codes at level ``len(prefix)``; all False for unknown prefixes."""
        level = len(prefix)
        if level >= len(self.level_sizes):
            raise ValueError("prefix is already a full ID")
        mask = np.zeros(self.level_sizes[level], dtype=bool)
        node = self._walk(prefix)
        if node is not None:
            for c in node.children:
                mask[c] = True
        return mask

    def allowed_token_mask(
        self,
        prefixes: Sequence[Sequence[int]],
        vocab_size: int,
        level_offsets: Sequence[int],
    ) -> torch.Tensor:
        """``(B, vocab_size)`` bool: which *token ids* may follow each prefix (dict walk)."""
        out = torch.zeros((len(prefixes), vocab_size), dtype=torch.bool)
        for i, prefix in enumerate(prefixes):
            level = len(prefix)
            allowed = self.allowed_next(prefix)
            off = level_offsets[level]
            out[i, off : off + allowed.shape[0]] = torch.from_numpy(allowed)
        return out

    def item_for(self, codes: Sequence[int]) -> int | None:
        node = self._walk(codes)
        return None if node is None or len(codes) != len(self.level_sizes) else node.item_id

    @property
    def num_leaves(self) -> int:
        return self._num_leaves

    def __len__(self) -> int:
        return self._num_leaves

    def leaf_item_ids(self) -> set[int]:
        out: set[int] = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.item_id is not None:
                out.add(node.item_id)
            stack.extend(node.children.values())
        return out

    # ------------------------------------------------------ vectorized masking
    def flat_keys(self, codes: torch.Tensor) -> torch.Tensor:
        """``(..., T)`` codes -> ``(...,)`` mixed-radix integer keys."""
        key = torch.zeros(codes.shape[:-1], dtype=torch.int64, device=codes.device)
        for level, size in enumerate(self.level_sizes):
            key = key * size + codes[..., level]
        return key

    def _require_codes(self) -> torch.Tensor:
        if self.codes is None:
            raise RuntimeError("vectorized masking needs a trie built with SemanticIdTrie.build")
        return self.codes

    def allowed_children_batch(
        self, prefixes: torch.Tensor, allowed_items: torch.Tensor
    ) -> torch.Tensor:
        """``prefixes (P, l)`` + ``allowed_items (P, N+1) bool`` -> ``(P, level_sizes[l]) bool``.

        Code ``c`` is allowed after prefix ``p`` iff some *allowed* item has Semantic ID
        starting with ``p + (c,)``.  Row 0 of ``allowed_items`` (PAD) must be False.
        """
        codes = self._require_codes().to(prefixes.device)
        p, l_ = prefixes.shape
        if l_ >= len(self.level_sizes):
            raise ValueError("prefix is already a full ID")
        if allowed_items.shape != (p, codes.shape[0]):
            raise ValueError("allowed_items must be (P, N+1)")
        if l_ == 0:
            match = allowed_items
        else:
            match = (codes[None, :, :l_] == prefixes[:, None, :]).all(dim=-1) & allowed_items
        child = codes[:, l_].unsqueeze(0).expand(p, -1)  # (P, N+1)
        counts = torch.zeros((p, self.level_sizes[l_]), dtype=torch.float32, device=codes.device)
        counts.scatter_add_(1, child, match.to(torch.float32))
        out: torch.Tensor = counts > 0
        return out

    def logit_mask(
        self,
        prefixes: torch.Tensor,
        allowed_items: torch.Tensor,
        vocab_size: int,
        level_offsets: Sequence[int],
    ) -> torch.Tensor:
        """``(P, vocab_size)`` bool token mask for the level following each prefix."""
        level = int(prefixes.shape[1])
        allowed = self.allowed_children_batch(prefixes, allowed_items)
        out = torch.zeros((prefixes.shape[0], vocab_size), dtype=torch.bool, device=allowed.device)
        off = level_offsets[level]
        out[:, off : off + allowed.shape[1]] = allowed
        return out

    def items_for_batch(self, codes: torch.Tensor) -> torch.Tensor:
        """``(..., T)`` full codes -> ``(...,)`` item ids of inserted leaves, ``-1`` if absent."""
        if self._keys is None or self._key_items is None:
            raise RuntimeError("items_for_batch needs a trie built with SemanticIdTrie.build")
        keys = self.flat_keys(codes)
        flat = keys.reshape(-1)
        pos = torch.searchsorted(self._keys, flat).clamp(max=max(self._keys.numel() - 1, 0))
        found = (
            self._keys[pos] == flat
            if self._keys.numel()
            else torch.zeros_like(flat, dtype=torch.bool)
        )
        items = torch.where(found, self._key_items[pos], torch.full_like(flat, -1))
        return items.reshape(keys.shape)
