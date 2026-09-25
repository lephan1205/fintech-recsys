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
from recsys.data.collator import SequenceCollator
from recsys.data.impression_collator import ImpressionBatch, ImpressionCollator
from recsys.data.schema import (
    PRODUCT_FEATURE_DIM,
    USER_FEATURE_DIM,
    ActionType,
    DelayConfig,
    ImpressionSlate,
    InteractionEvent,
    InteractionRecord,
)
from recsys.layers.hstu import HSTUConfig
from recsys.layers.prefix_trie import SemanticIdTrie
from recsys.layers.rq_vae import RQVAE, RQVAEConfig
from recsys.losses.funnel_loss import UnifiedFunnelLoss
from recsys.losses.ziln_loss import ZILNHead, ziln_expected_value, ziln_loss
from recsys.metrics.ranking_metrics import recall_at_k
from recsys.models.ple.model import PLEConfig
from recsys.models.ranker import HSTUPLERanker
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


# ------------------------------------------------------------------- HSTU + PLE


def _ev(item: int, action: ActionType, ts: float) -> InteractionEvent:
    return InteractionEvent(item_id=item, action=action, timestamp=ts)


def funnel_batch(num_items: int = 30) -> ImpressionBatch:
    """3 slates x 4 candidates with a monotone funnel, one resolved approval, one pending."""
    records = [
        InteractionRecord(
            user_index=u,
            history=tuple(
                _ev(1 + (u * 3 + i) % num_items, ActionType.VIEW, float(i)) for i in range(4)
            ),
            target=_ev(2 + u, ActionType.VIEW, 5.0),
        )
        for u in range(3)
    ]
    slates = [
        ImpressionSlate(
            user_index=0, slate_id=0, candidate_item_ids=(1, 2, 3, 4),
            y_click=(1, 1, 0, 0), y_apply=(1, 0, 0, 0), y_approve=(1, 0, 0, 0),
            p_click=(0.5,) * 4, p_apply=(0.5,) * 4, p_approve=(0.5,) * 4,
            payouts=(100.0,) * 4, amounts=(8000.0, 0.0, 0.0, 0.0), eligible=(True,) * 4,
            served_at_days=300.0, decision_delay_days=(1.0, 0.0, 0.0, 0.0),
        ),
        ImpressionSlate(
            user_index=1, slate_id=1, candidate_item_ids=(5, 6, 7, 8),
            y_click=(0, 1, 0, 1), y_apply=(0, 1, 0, 0), y_approve=(0, 1, 0, 0),
            p_click=(0.5,) * 4, p_apply=(0.5,) * 4, p_approve=(0.5,) * 4,
            payouts=(100.0,) * 4, amounts=(0.0, 12000.0, 0.0, 0.0), eligible=(True,) * 4,
            served_at_days=360.0, decision_delay_days=(0.0, 30.0, 0.0, 0.0),  # pending
        ),
        ImpressionSlate(
            user_index=2, slate_id=2, candidate_item_ids=(9, 10, 11, 12),
            y_click=(1, 0, 0, 0), y_apply=(1, 0, 0, 0), y_approve=(0, 0, 0, 0),
            p_click=(0.5,) * 4, p_apply=(0.5,) * 4, p_approve=(0.5,) * 4,
            payouts=(100.0,) * 4, amounts=(0.0,) * 4, eligible=(True,) * 4,
            served_at_days=350.0, decision_delay_days=(2.0, 0.0, 0.0, 0.0),
        ),
    ]  # fmt: skip
    gen = torch.Generator().manual_seed(0)
    catalog = torch.rand(num_items + 1, PRODUCT_FEATURE_DIM, generator=gen).numpy()
    catalog[0] = 0
    users = torch.rand(3, USER_FEATURE_DIM, generator=gen).numpy()
    family = np.array([-1] + [i % 5 for i in range(num_items)], dtype=np.int64)
    coll = ImpressionCollator(
        catalog, users, family, snapshot_at_days=365.0, delay_config=DelayConfig()
    )
    batch = coll(slates)
    return batch.with_sequence(SequenceCollator(max_len=6)(records))


def test_hstu_ple_ranker_overfits_unified_funnel_loss() -> None:
    batch = funnel_batch()
    assert batch.status[1, 1].item() == 1 and not batch.approve_observed[1, 1]
    hstu = HSTUConfig(
        num_items=30, d_model=16, n_heads=2, n_layers=1, max_len=6, max_candidates=4,
        num_time_buckets=8,
    )  # fmt: skip
    ple = PLEConfig(input_dim=hstu.fusion_dim, expert_hidden=(16,), expert_dim=8, tower_hidden=(8,))
    model = HSTUPLERanker(hstu, ple, downsample_rate=0.25).train()
    model.fit_tabular_stats(batch.tabular, batch.candidate_mask)
    loss = UnifiedFunnelLoss()

    def loss_fn() -> torch.Tensor:
        out = model(batch)
        terms = loss(
            out.z1, out.z2, out.z3, batch.y_click, batch.y_apply, batch.y_approve,
            batch.approve_weight, batch.approve_observed, batch.candidate_mask,
            amount_logits=out.amount_logits, amounts=batch.amounts,
        )  # fmt: skip
        total: torch.Tensor = terms.total
        return total

    # the ZILN NLL has a data-dependent floor (log y), so threshold the probability terms
    run_overfit(list(model.parameters()), loss_fn, threshold=0.3, lr=1e-2, max_steps=400)
    model.eval()
    out = model(batch)
    terms = loss(
        out.z1, out.z2, out.z3, batch.y_click, batch.y_apply, batch.y_approve,
        batch.approve_weight, batch.approve_observed, batch.candidate_mask,
        amount_logits=out.amount_logits, amounts=batch.amounts,
    )  # fmt: skip
    prob_terms = terms.click + terms.apply + terms.approve + terms.ctcvr + terms.ctcavr
    assert float(prob_terms) < 0.15
    assert terms.num_resolved_applications == 2 and terms.num_amount_rows == 1
    assert out.amount_logits is not None
    ev = ziln_expected_value(out.amount_logits)
    assert abs(float(ev[0, 0]) - 8000.0) / 8000.0 < 0.5  # approved row's amount recovered
    # predict() adds log r to the click logit only
    pred = model.predict(batch)
    assert torch.allclose(pred.z1, out.z1 + np.log(0.25)) and torch.allclose(pred.z2, out.z2)


# -------------------------------------------------------------------------- ZILN


def test_ziln_overfits() -> None:
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(32, 6, generator=gen)
    # positive iff x[:, 1] > 0 (separable); amount log-linear in x[:, 0]
    y = torch.where(x[:, 1] > 0, torch.exp(8.0 + 0.3 * x[:, 0]), torch.zeros(32))
    head = ZILNHead(6, mu_bias_init=8.0).train()

    def loss_fn() -> torch.Tensor:
        out: torch.Tensor = ziln_loss(head(x), y).total
        return out

    # the NLL has a data-dependent floor (log y), so the early-exit threshold never fires
    run_overfit(list(head.parameters()), loss_fn, threshold=0.0, lr=2e-2, max_steps=1000)
    full = ziln_loss(head(x), y)
    assert float(full.bce.detach()) < 0.1
    ev = ziln_expected_value(head(x))
    pos = y > 0
    assert abs(float(ev[pos].mean()) - float(y[pos].mean())) / float(y[pos].mean()) < 0.3
    assert float(ev[~pos].mean()) < 0.1 * float(y[pos].mean())
