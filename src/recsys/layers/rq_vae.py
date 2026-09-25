"""Residual-Quantized VAE for hierarchical Semantic IDs (TIGER, Rajput et al. 2023).

A standardized product feature vector ``x`` (18-dim) is encoded to ``z``, then
quantized level by level: ``r_0 = z``, ``c_d = argmin_k ||r_d - C_d[k]||``,
``r_{d+1} = r_d - C_d[c_d]``.  The tuple ``(c_1, ..., c_D)`` is a coarse-to-fine
Semantic ID; items with similar underwriting / economics share prefixes, which is
what makes prefix-trie masking meaningful.  A 4th *disambiguation* level makes IDs
unique (:meth:`RQVAE.assign_semantic_ids`).

Training (D8): the only gradient loss is ``recon + beta * commit`` (straight-through
estimator on the decoder path).  **Codebooks are not optimizer parameters**: they are
buffers updated by exponential moving averages (decay 0.99) of the residuals assigned
to each code, with Laplace smoothing, and dead codes (EMA count below
``dead_code_threshold``) are re-seeded from random encoder outputs every
``dead_code_reset_steps`` steps.  Plain Adam with no weight decay: decay would shrink
the encoder outputs toward the origin while the EMA codebooks follow them, contracting
the residuals at depths 2-3 and collapsing utilization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RQVAEConfig:
    input_dim: int
    latent_dim: int = 16
    num_levels: int = 3
    codebook_size: int = 32
    encoder_hidden: tuple[int, ...] = (32,)
    beta_commit: float = 0.25
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5
    kmeans_init: bool = True
    kmeans_iters: int = 10
    dead_code_reset_steps: int = 100
    dead_code_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.num_levels < 1 or self.codebook_size < 2:
            raise ValueError("need num_levels >= 1 and codebook_size >= 2")


@dataclass
class RQVAEOutput:
    recon: torch.Tensor  # (N, F)
    codes: torch.Tensor  # (N, num_levels) int64
    quantized: torch.Tensor  # (N, latent)
    loss_recon: torch.Tensor
    loss_commit: torch.Tensor
    loss_total: torch.Tensor


def _mlp(dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class RQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.encoder = _mlp((c.input_dim, *c.encoder_hidden, c.latent_dim))
        self.decoder = _mlp((c.latent_dim, *reversed(c.encoder_hidden), c.input_dim))
        init = torch.randn(c.num_levels, c.codebook_size, c.latent_dim) * 0.1
        self.register_buffer("codebooks", init)
        self.register_buffer("ema_count", torch.ones(c.num_levels, c.codebook_size))
        self.register_buffer("ema_sum", init.clone())
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.int64))
        self.codebooks: torch.Tensor
        self.ema_count: torch.Tensor
        self.ema_sum: torch.Tensor
        self.initialized: torch.Tensor
        self.steps: torch.Tensor

    # ------------------------------------------------------------ quantization
    def _nearest(self, residual: torch.Tensor, level: int) -> torch.Tensor:
        dist = torch.cdist(residual, self.codebooks[level])  # (N, K)
        return dist.argmin(dim=-1)

    def quantize(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Returns ``(z_q, codes, residuals, quantized_per_level)``."""
        residual = z
        codes, residuals, quantized = [], [], []
        z_q = torch.zeros_like(z)
        for level in range(self.config.num_levels):
            idx = self._nearest(residual.detach(), level)
            q = self.codebooks[level][idx]  # (N, latent)
            codes.append(idx)
            residuals.append(residual)
            quantized.append(q)
            z_q = z_q + q
            residual = residual - q.detach()
        return z_q, torch.stack(codes, dim=1), residuals, quantized

    def forward(self, x: torch.Tensor) -> RQVAEOutput:
        if self.training and self.config.kmeans_init and not bool(self.initialized):
            self.init_codebooks_kmeans(x)
        z = self.encoder(x)
        z_q, codes, residuals, quantized = self.quantize(z)
        z_st = z + (z_q - z).detach()
        recon = self.decoder(z_st)

        loss_recon = F.mse_loss(recon, x)
        loss_commit = self.config.beta_commit * sum(
            (F.mse_loss(r, q.detach()) for r, q in zip(residuals, quantized, strict=True)),
            torch.zeros((), device=x.device),
        )
        loss_total = loss_recon + loss_commit

        if self.training:
            with torch.no_grad():
                self._ema_update(codes, residuals)
                self.steps += 1
                if (
                    self.config.dead_code_reset_steps > 0
                    and int(self.steps) % self.config.dead_code_reset_steps == 0
                ):
                    self.reset_dead_codes(x)
        return RQVAEOutput(
            recon=recon,
            codes=codes,
            quantized=z_q,
            loss_recon=loss_recon,
            loss_commit=loss_commit,
            loss_total=loss_total,
        )

    # ------------------------------------------------------------- EMA codebooks
    @torch.no_grad()
    def _ema_update(self, codes: torch.Tensor, residuals: list[torch.Tensor]) -> None:
        c = self.config
        for level in range(c.num_levels):
            onehot = F.one_hot(codes[:, level], c.codebook_size).to(residuals[level].dtype)
            count = onehot.sum(dim=0)  # (K,)
            summed = onehot.t() @ residuals[level].detach()  # (K, latent)
            self.ema_count[level].mul_(c.ema_decay).add_(count, alpha=1.0 - c.ema_decay)
            self.ema_sum[level].mul_(c.ema_decay).add_(summed, alpha=1.0 - c.ema_decay)
            n = self.ema_count[level].sum()
            smoothed = (self.ema_count[level] + c.ema_epsilon) / (
                n + c.codebook_size * c.ema_epsilon
            )
            smoothed = smoothed * n
            self.codebooks[level].copy_(self.ema_sum[level] / smoothed.unsqueeze(1))

    # ----------------------------------------------------------- collapse fixes
    @torch.no_grad()
    def init_codebooks_kmeans(self, x: torch.Tensor) -> None:
        """Level-wise Lloyd's algorithm on the residuals each level will quantize."""
        residual = self.encoder(x)
        n, k = residual.shape[0], self.config.codebook_size
        gen = torch.Generator(device="cpu").manual_seed(int(self.steps) + 1)
        for level in range(self.config.num_levels):
            perm = torch.randperm(n, generator=gen)
            centers = residual[perm[:k]].clone()
            if n < k:  # not enough points: pad with jittered copies
                extra = residual[perm[torch.randint(0, n, (k - n,), generator=gen)]]
                centers = torch.cat([centers, extra + 0.01 * torch.randn_like(extra)], dim=0)
            for _ in range(self.config.kmeans_iters):
                assign = torch.cdist(residual, centers).argmin(dim=-1)
                for j in range(k):
                    members = residual[assign == j]
                    if members.shape[0] > 0:
                        centers[j] = members.mean(dim=0)
            self.codebooks[level].copy_(centers)
            self.ema_sum[level].copy_(centers)
            self.ema_count[level].fill_(1.0)
            residual = residual - centers[torch.cdist(residual, centers).argmin(dim=-1)]
        self.initialized.fill_(True)

    @torch.no_grad()
    def reset_dead_codes(self, x: torch.Tensor) -> None:
        """Re-seed codes whose EMA count fell below the threshold with random encoder
        residuals (plus a little noise) and restart their EMA statistics."""
        residual = self.encoder(x)
        gen = torch.Generator(device="cpu").manual_seed(int(self.steps))
        for level in range(self.config.num_levels):
            dead = torch.nonzero(self.ema_count[level] < self.config.dead_code_threshold).flatten()
            if dead.numel() > 0:
                pick = torch.randint(0, residual.shape[0], (dead.numel(),), generator=gen)
                fresh = residual[pick] + 0.01 * torch.randn(
                    dead.numel(), residual.shape[1], generator=gen
                )
                self.codebooks[level][dead] = fresh
                self.ema_sum[level][dead] = fresh
                self.ema_count[level][dead] = 1.0
            idx = torch.cdist(residual, self.codebooks[level]).argmin(dim=-1)
            residual = residual - self.codebooks[level][idx]

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, F)`` -> ``(N, num_levels)`` int64 codes."""
        _, codes, _, _ = self.quantize(self.encoder(x))
        return codes

    @torch.no_grad()
    def assign_semantic_ids(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, F)`` -> ``(N, num_levels + 1)``: codes plus a dedup level (unique IDs)."""
        codes = self.encode(x)
        seen: dict[tuple[int, ...], int] = {}
        dedup = torch.zeros(codes.shape[0], dtype=torch.int64)
        for i, row in enumerate(codes.tolist()):
            key = tuple(row)
            count = seen.get(key, 0)
            dedup[i] = count
            seen[key] = count + 1
        return torch.cat([codes, dedup.unsqueeze(1)], dim=1)

    def codebook_usage_fraction(self) -> torch.Tensor:
        """``(num_levels,)`` fraction of codes whose EMA count is above the dead threshold."""
        return (self.ema_count >= self.config.dead_code_threshold).float().mean(dim=1)

    @torch.no_grad()
    def codebook_utilization(self, x: torch.Tensor) -> torch.Tensor:
        """``(num_levels,)`` fraction of codes actually used when encoding ``x``."""
        codes = self.encode(x)
        used = [
            torch.bincount(codes[:, level], minlength=self.config.codebook_size) > 0
            for level in range(self.config.num_levels)
        ]
        return torch.stack([u.float().mean() for u in used])
