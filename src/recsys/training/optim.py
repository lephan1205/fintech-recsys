"""Optimizer construction and the warmup-cosine schedule (D8).

``build_optimizer`` builds two parameter groups: decoupled weight decay on weight
matrices only — **no decay on embeddings, biases, or normalization gains** (anything
with ``ndim < 2`` or owned by an ``nn.Embedding``).  Buffers (RQ-VAE codebooks, EMA
statistics, logit corrections) are never optimizer parameters.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from recsys.training.config import OptimizerConfig


def split_decay_params(model: nn.Module) -> tuple[list[str], list[str]]:
    """``(decay_names, no_decay_names)`` over trainable parameters."""
    embedding_params: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embedding_params.update(id(p) for p in module.parameters(recurse=False))
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or id(p) in embedding_params:
            no_decay.append(name)
        else:
            decay.append(name)
    return decay, no_decay


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    decay_names, no_decay_names = split_decay_params(model)
    params = dict(model.named_parameters())
    groups = [
        {"params": [params[n] for n in decay_names], "weight_decay": cfg.weight_decay},
        {"params": [params[n] for n in no_decay_names], "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    if cfg.name == "adam":
        return torch.optim.Adam(groups, lr=cfg.lr, betas=cfg.betas)
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas)


def lr_at(step: int, cfg: OptimizerConfig) -> float:
    """Linear warmup to ``lr`` at ``warmup_steps``, then cosine to ``lr * final_lr_fraction``
    at ``total_steps`` (constant when ``final_lr_fraction == 1``)."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.final_lr_fraction >= 1.0:
        return cfg.lr
    span = max(cfg.total_steps - cfg.warmup_steps, 1)
    progress = min(max(step - cfg.warmup_steps, 0) / span, 1.0)
    floor = cfg.lr * cfg.final_lr_fraction
    return floor + 0.5 * (cfg.lr - floor) * (1.0 + math.cos(math.pi * progress))


class WarmupCosine:
    """Sets the optimizer lr from :func:`lr_at`; call ``step()`` after ``optimizer.step()``."""

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: OptimizerConfig) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.current_step = 0
        self._apply()

    def _apply(self) -> None:
        lr = lr_at(self.current_step, self.cfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def step(self) -> None:
        self.current_step += 1
        self._apply()

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])


def clip_gradients(model: nn.Module, cfg: OptimizerConfig) -> float:
    """Global-norm clipping when configured; returns the pre-clip norm."""
    if cfg.grad_clip is None:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        return float(torch.norm(torch.stack([g.norm() for g in grads]))) if grads else 0.0
    return float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
