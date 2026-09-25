"""fintech-recsys: a 4-stage cascade FinTech recommender (TIGER, HSTU, PLE, calibrated PRM).

Reserved indices are defined once in :mod:`recsys.data.schema` and re-exported here.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from recsys.data.schema import PAD_ACTION_ID, PAD_ITEM_ID, SCORE_CHANGE_ITEM_ID

__all__ = ["PAD_ACTION_ID", "PAD_ITEM_ID", "SCORE_CHANGE_ITEM_ID", "seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments and tests."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
