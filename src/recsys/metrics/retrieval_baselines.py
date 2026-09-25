"""Retrieval baselines for D7: eligible-popularity and eligible-random top-K.

Both use the *same* eligible mask and the same number of slots as the TIGER beam, so
the comparison isolates what the generative retriever adds.  Popularity counts come
from the **training split only** (positive actions in histories and targets).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from recsys.data.schema import POSITIVE_ACTIONS, InteractionRecord


def positive_item_counts(
    records: Sequence[InteractionRecord], num_items: int
) -> npt.NDArray[np.int64]:
    """``(N+1,)`` count of positive-action events (history + target) per item; row 0 is 0."""
    counts = np.zeros(num_items + 1, dtype=np.int64)
    for r in records:
        for e in r.history:
            if e.action in POSITIVE_ACTIONS and e.item_id > 0:
                counts[e.item_id] += 1
        if r.target.action in POSITIVE_ACTIONS:
            counts[r.target.item_id] += 1
    counts[0] = 0
    return counts


def eligible_popularity_top_k(
    counts: npt.NDArray[np.int64], allowed: npt.NDArray[np.bool_], k: int
) -> npt.NDArray[np.int64]:
    """``(B, N+1)`` allowed -> ``(B, k)`` most popular eligible ids (``-1`` when fewer)."""
    score = np.where(allowed, counts[None, :].astype(np.float64), -np.inf)
    score[:, 0] = -np.inf
    # stable tie-break by item id so results are deterministic
    order = np.lexsort(
        (np.arange(score.shape[1])[None, :].repeat(score.shape[0], 0), -score), axis=1
    )
    top = order[:, :k]
    picked = np.take_along_axis(score, top, axis=1)
    out: npt.NDArray[np.int64] = np.where(np.isfinite(picked), top, -1).astype(np.int64)
    return out


def eligible_random_top_k(
    allowed: npt.NDArray[np.bool_], k: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """``(B, N+1)`` allowed -> ``(B, k)`` uniformly random eligible ids (``-1`` when fewer)."""
    b = allowed.shape[0]
    out = np.full((b, k), -1, dtype=np.int64)
    for i in range(b):
        ids = np.flatnonzero(allowed[i])
        ids = ids[ids > 0]
        if ids.size:
            pick = rng.choice(ids, size=min(k, ids.size), replace=False)
            out[i, : pick.size] = pick
    return out
