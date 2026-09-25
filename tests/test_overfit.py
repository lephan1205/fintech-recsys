"""Overfit oracles.

Every model must drive its training loss toward ~0 on a tiny fixed batch.  This
proves gradient flow end to end (no detached graph, no dead mask) and is the
cheapest possible regression test for a refactor.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from recsys import seed_everything
from recsys.layers.prefix_trie import SemanticIdTrie
from recsys.layers.rq_vae import RQVAE, RQVAEConfig
from recsys.metrics.ranking_metrics import recall_at_k
from recsys.models.tiger.model import TIGER, SemanticIdTokenizer, TIGERConfig

MAX_STEPS = 300


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


def run_overfit(
    params: list[torch.nn.Parameter],
    loss_fn: Callable[[], torch.Tensor],
    threshold: float,
    lr: float = 1e-2,
    max_steps: int = MAX_STEPS,
) -> tuple[float, float]:
    """Adam loop with early exit.  Returns ``(initial_loss, final_loss)``."""
    opt = torch.optim.Adam(params, lr=lr)
    initial = final = float("nan")
    for _ in range(max_steps):
        loss = loss_fn()
        if initial != initial:  # first step
            initial = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        final = float(loss.detach())
        if final < threshold:
            break
    return initial, final


# ------------------------------------------------------------------------ RQ-VAE


def test_rqvae_reconstructs_with_ema_codebooks() -> None:
    gen = torch.Generator().manual_seed(0)
    # rank-3 catalog geometry in 8 dims (random Gaussians cannot be compressed losslessly)
    x = torch.randn(64, 3, generator=gen) @ torch.randn(3, 8, generator=gen)
    model = RQVAE(
        RQVAEConfig(input_dim=8, latent_dim=8, num_levels=3, codebook_size=16, encoder_hidden=(16,))
    ).train()
    # codebooks are buffers, not parameters (EMA-updated), so no optimizer sees them
    assert "codebooks" not in {n for n, _ in model.named_parameters()}
    assert "codebooks" in {n for n, _ in model.named_buffers()}
    before = model.codebooks.clone()

    def loss_fn() -> torch.Tensor:
        out: torch.Tensor = model(x).loss_total
        return out

    _, final = run_overfit(list(model.parameters()), loss_fn, threshold=0.05, lr=1e-2)
    model.eval()
    recon = torch.nn.functional.mse_loss(model(x).recon, x)
    assert float(recon) < 0.05 and final < 0.06
    assert not torch.equal(before, model.codebooks)  # EMA moved the codebooks
    util = model.codebook_utilization(x)
    assert util.shape == (3,) and float(util.min()) > 0.25
    assert model.codebook_usage_fraction().shape == (3,)
    sids = model.assign_semantic_ids(x)
    assert sids.shape == (64, 4) and len({tuple(r) for r in sids.tolist()}) == 64


# ------------------------------------------------------------------------- TIGER


def test_tiger_overfits_and_reaches_full_recall_on_memorized_examples() -> None:
    gen = torch.Generator().manual_seed(0)
    n, base_sizes = 12, (4, 4, 4)
    codes = torch.stack([torch.randint(0, s, (n + 1,), generator=gen) for s in base_sizes], 1)
    seen: dict[tuple[int, ...], int] = {}
    dedup = torch.zeros(n + 1, 1, dtype=torch.int64)
    for i in range(1, n + 1):
        key = tuple(codes[i].tolist())
        dedup[i, 0] = seen.get(key, 0)
        seen[key] = int(dedup[i, 0]) + 1
    codes = torch.cat([codes, dedup], dim=1)
    codes[0] = 0
    level_sizes: tuple[int, ...] = (*base_sizes, int(dedup.max()) + 1)
    cfg = TIGERConfig(
        num_items=n, level_sizes=level_sizes, d_model=16, n_heads=2, n_layers=1, d_ff=32,
        max_history_items=5,
    )  # fmt: skip
    tok = SemanticIdTokenizer(codes, cfg)
    model = TIGER(cfg).train()
    items = torch.randint(1, n + 1, (4, 5), generator=gen)
    acts = torch.randint(1, 7, (4, 5), generator=gen)
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[0, :2] = False
    items[0, :2] = 0
    acts[0, :2] = 0
    tier = torch.tensor([0, 1, 2, 3])
    state = torch.tensor([4, 4, 5, 6])
    ht, ha, hm = tok.encode_history(items, acts, mask, tier, state)
    targets = torch.tensor([1, 5, 9, 12])

    def loss_fn() -> torch.Tensor:
        return model.next_sid_loss(ht, ha, hm, codes[targets], tok).total

    initial, final = run_overfit(list(model.parameters()), loss_fn, threshold=0.10)
    assert final < 0.10 and final < 0.2 * initial
    model.eval()
    trie = SemanticIdTrie.build(codes[1:], np.arange(1, n + 1), level_sizes)
    allowed = torch.ones((4, n + 1), dtype=torch.bool)
    allowed[:, 0] = False
    out = model.generate(ht, ha, hm, trie, tok, allowed, beam_size=2)
    assert out.item_ids[:, 0].tolist() == targets.tolist()
    # Recall@100 == 1.0 on the memorized examples (beam wider than the catalog)
    wide = model.generate(ht, ha, hm, trie, tok, allowed, beam_size=100)
    r = recall_at_k(wide.item_ids.numpy(), targets.numpy()[:, None], np.ones((4, 1), bool), 100)
    assert r.mean == 1.0 and r.num_users == 4
    assert wide.item_ids.shape == (4, 100) and (wide.item_ids[:, n:] == -1).all()
