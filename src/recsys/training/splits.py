"""User-level train / val / calib / test split.

Records and slates follow their user, so no user's behaviour leaks across splits:
``val`` drives early stopping and checkpoint selection, ``calib`` fits the calibrators
(disjoint from anything the models were trained or selected on), ``test`` reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from recsys.data.schema import ImpressionSlate, InteractionRecord, SyntheticDataset
from recsys.training.config import SplitConfig

SplitName = Literal["train", "val", "calib", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "calib", "test")


@dataclass
class Splits:
    assignment: npt.NDArray[np.int64]  # (U,) index into SPLIT_NAMES

    def users(self, name: SplitName) -> npt.NDArray[np.int64]:
        out: npt.NDArray[np.int64] = np.flatnonzero(self.assignment == SPLIT_NAMES.index(name))
        return out.astype(np.int64)

    def member(self, name: SplitName) -> npt.NDArray[np.bool_]:
        out: npt.NDArray[np.bool_] = self.assignment == SPLIT_NAMES.index(name)
        return out

    def records(self, ds: SyntheticDataset, name: SplitName) -> list[InteractionRecord]:
        m = self.member(name)
        return [r for r in ds.interactions if m[r.user_index]]

    def slates(self, ds: SyntheticDataset, name: SplitName) -> list[ImpressionSlate]:
        m = self.member(name)
        return [s for s in ds.slates if m[s.user_index]]

    def sizes(self) -> dict[str, int]:
        return {n: int(self.member(n).sum()) for n in SPLIT_NAMES}


def split_by_user(num_users: int, config: SplitConfig | None = None) -> Splits:
    cfg = config or SplitConfig()
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(num_users)
    bounds = np.cumsum([cfg.train, cfg.val, cfg.calib]) * num_users
    cuts = np.round(bounds).astype(int)
    assignment = np.zeros(num_users, dtype=np.int64)
    assignment[perm[cuts[0] : cuts[1]]] = 1
    assignment[perm[cuts[1] : cuts[2]]] = 2
    assignment[perm[cuts[2] :]] = 3
    return Splits(assignment)


def records_by_user(records: Sequence[InteractionRecord]) -> dict[int, InteractionRecord]:
    return {r.user_index: r for r in records}
