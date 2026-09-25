"""HSTU backbone: causality under left padding, M-FALCON candidate independence,
batched == per-candidate scoring, right-padding parity via gather_last_real."""

from __future__ import annotations

import pytest
import torch

from recsys import seed_everything
from recsys.data.collator import SequenceBatch, SequenceCollator
from recsys.data.schema import ActionType, InteractionEvent, InteractionRecord
from recsys.layers.hstu import HSTUConfig, bucketize_time, build_mfalcon_mask
from recsys.models.hstu.model import HSTUBackbone

NUM_ITEMS = 30


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


def ev(item: int, action: ActionType, ts: float) -> InteractionEvent:
    return InteractionEvent(item_id=item, action=action, timestamp=ts)


def records() -> list[InteractionRecord]:
    return [
        InteractionRecord(
            user_index=0,
            history=(ev(0, ActionType.SCORE_CHANGE, 0.0), ev(7, ActionType.VIEW, 1.5)),
            target=ev(3, ActionType.APPLY_DECLINED, 2.0),
        ),
        InteractionRecord(
            user_index=1,
            history=tuple(ev(i, ActionType.VIEW, float(i) * 3.0) for i in range(1, 7)),
            target=ev(9, ActionType.CREDIT_PULL, 20.0),
        ),
        InteractionRecord(
            user_index=2,
            history=(
                ev(4, ActionType.VIEW, 0.0),
                ev(0, ActionType.SCORE_CHANGE, 3.0),
                ev(11, ActionType.APPLY_PENDING, 10.0),
            ),
            target=ev(5, ActionType.VIEW, 12.0),
        ),
    ]


def tiny(padding_side: str = "left", max_len: int = 6) -> HSTUBackbone:
    cfg = HSTUConfig(
        num_items=NUM_ITEMS, d_model=16, n_heads=2, n_layers=2, max_len=max_len,
        max_candidates=5, num_time_buckets=8, tabular_dim=4,
        padding_side=padding_side,  # type: ignore[arg-type]
    )  # fmt: skip
    return HSTUBackbone(cfg).eval()


def cands(b: int = 3, k: int = 4, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, NUM_ITEMS + 1, (b, k), generator=gen)


# ------------------------------------------------------------------ mask + buckets


def test_mfalcon_mask_structure() -> None:
    mask = torch.tensor([[False, True, True], [True, True, True]])
    m = build_mfalcon_mask(mask, 2)
    assert m.shape == (2, 5, 5)
    ell = 3
    # history -> history causal & key real
    assert m[0, 2, :ell].tolist() == [False, True, True]
    assert m[0, 1, :ell].tolist() == [False, True, False]
    assert m[1, 1, :ell].tolist() == [True, True, False]
    # history never attends to candidates
    assert not m[:, :ell, ell:].any()
    # candidates attend to all real history and themselves only
    assert m[0, 3].tolist() == [False, True, True, True, False]
    assert m[0, 4].tolist() == [False, True, True, False, True]
    # fully padded history row still has its diagonal
    assert m[0, 0, 0]


def test_time_buckets_monotone_and_bounded() -> None:
    days = torch.tensor([0.0, 0.5, 1.0, 7.0, 30.0, 365.0, 1e6, -3.0])
    b = bucketize_time(days, 32, 365.0)
    assert b.tolist()[0] == 0 and b.tolist()[-1] == 0
    assert b[5] == 31 and b[6] == 31
    assert all(x <= y for x, y in zip(b.tolist()[:6], b.tolist()[1:6], strict=False))


# --------------------------------------------------------------------- causality


def test_causality_under_left_padding() -> None:
    model = tiny()
    batch = SequenceCollator(max_len=6)(records())
    c = cands()
    base = model.score_candidates(batch, c).hidden
    # perturb the two most recent history events of row 1 (real, positions 4-5)
    items = batch.item_ids.clone()
    items[1, 4:] = torch.tensor([13, 14])
    pert = SequenceBatch(**{**batch.__dict__, "item_ids": items})
    out = model.score_candidates(pert, c).hidden
    assert torch.allclose(base[1, :4], out[1, :4], atol=1e-6)
    assert not torch.allclose(base[1, 4:6], out[1, 4:6])
    # other rows untouched
    assert torch.allclose(base[0], out[0], atol=1e-6) and torch.allclose(base[2], out[2], atol=1e-6)


def test_pad_content_does_not_leak() -> None:
    model = tiny()
    batch = SequenceCollator(max_len=6)(records())
    c = cands()
    base = model.score_candidates(batch, c)
    items = batch.item_ids.clone()
    deltas = batch.time_deltas.clone()
    items[0, :4] = 17  # PAD slots of row 0 (length 2)
    deltas[0, :4] = 99.0
    pert = SequenceBatch(**{**batch.__dict__, "item_ids": items, "time_deltas": deltas})
    out = model.score_candidates(pert, c)
    assert torch.allclose(base.h_user, out.h_user, atol=1e-6)
    assert torch.allclose(base.h_cand, out.h_cand, atol=1e-6)
    assert torch.allclose(base.hidden[0, 4:6], out.hidden[0, 4:6], atol=1e-6)


def test_gradient_of_past_wrt_future_is_zero() -> None:
    model = tiny()
    batch = SequenceCollator(max_len=6)(records())
    c = cands()
    emb = model.embed(batch, c).detach().requires_grad_(True)
    attn_mask = build_mfalcon_mask(batch.attention_mask, c.shape[1])
    rel, tb = model.relative_indices(batch, c.shape[1])
    x = emb
    for layer in model.layers:
        x = layer(x, attn_mask, rel, tb)
    hidden = model.final_ln(x)
    w = torch.randn(16)
    (hidden[1, 2] @ w).backward()
    assert emb.grad is not None
    assert torch.all(emb.grad[1, 3:] == 0)  # later history and all candidates
    assert torch.any(emb.grad[1, :3] != 0)


# ----------------------------------------------------------- candidate independence


def test_candidate_independence_permutation_and_removal() -> None:
    model = tiny()
    batch = SequenceCollator(max_len=6)(records())
    c = cands(k=5)
    full = model.score_candidates(batch, c)
    # permutation: representations follow the candidates
    perm = torch.tensor([4, 2, 0, 3, 1])
    out_p = model.score_candidates(batch, c[:, perm])
    assert torch.allclose(out_p.h_cand, full.h_cand[:, perm], atol=1e-6)
    assert torch.allclose(out_p.h_user, full.h_user, atol=1e-6)
    # removal: dropping candidates leaves the others unchanged
    out_r = model.score_candidates(batch, c[:, :2])
    assert torch.allclose(out_r.h_cand, full.h_cand[:, :2], atol=1e-6)
    assert torch.allclose(out_r.h_user, full.h_user, atol=1e-6)
    # history hidden states do not depend on candidates at all
    assert torch.allclose(out_r.hidden[:, :6], full.hidden[:, :6], atol=1e-6)


def test_batched_equals_per_candidate_loop() -> None:
    model = tiny()
    batch = SequenceCollator(max_len=6)(records())
    c = cands(k=5)
    full = model.score_candidates(batch, c)
    for j in range(5):
        single = model.score_candidates(batch, c[:, j : j + 1])
        assert torch.allclose(single.h_cand[:, 0], full.h_cand[:, j], atol=1e-6)


# ------------------------------------------------------------------ right padding


def test_right_padding_parity_via_gather_last_real() -> None:
    left, right = tiny("left"), tiny("right")
    right.load_state_dict(left.state_dict())
    c = cands()
    out_l = left.score_candidates(SequenceCollator(max_len=6)(records()), c)
    out_r = right.score_candidates(SequenceCollator(max_len=6, padding_side="right")(records()), c)
    assert torch.allclose(out_l.h_user, out_r.h_user, atol=1e-5)
    assert torch.allclose(out_l.h_cand, out_r.h_cand, atol=1e-5)
    with pytest.raises(ValueError):
        left.score_candidates(SequenceCollator(max_len=6, padding_side="right")(records()), c)


def test_fusion_shape_and_validation() -> None:
    model = tiny()
    batch = SequenceCollator(max_len=6)(records())
    c = cands()
    out = model.score_candidates(batch, c)
    fused = model.fusion(out.h_user, out.h_cand, torch.zeros(3, 4, 4))
    assert fused.shape == (3, 4, 3 * 16 + 4)
    assert torch.allclose(fused[:, :, :16], out.h_cand)
    with pytest.raises(ValueError):
        model.fusion(out.h_user, out.h_cand, torch.zeros(3, 4, 5))
    with pytest.raises(ValueError):
        model.score_candidates(batch, torch.ones(3, 9, dtype=torch.int64))
