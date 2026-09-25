"""Left-padding invariants.

The two facts every downstream model relies on:

1. ``attention_mask`` is derived from ``action_ids`` so a ``SCORE_CHANGE`` event
   (item id 0) is a real token, not padding.
2. Under left padding, index ``L-1`` is the most recent real event, so
   ``seq_rep == hidden[:, -1]`` with no gather.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from recsys import seed_everything
from recsys.data.collator import SequenceBatch, SequenceCollator
from recsys.data.impression_collator import ImpressionCollator
from recsys.data.schema import (
    PRODUCT_FEATURE_DIM,
    USER_FEATURE_DIM,
    ActionType,
    ImpressionSlate,
    InteractionEvent,
    InteractionRecord,
)
from recsys.layers.multi_modal_embedding import MultiModalEmbedding, MultiModalEmbeddingConfig
from recsys.layers.transformer_blocks import (
    TransformerEncoder,
    TransformerEncoderConfig,
    gather_last_real,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


def ev(item: int, action: ActionType, ts: float) -> InteractionEvent:
    return InteractionEvent(item_id=item, action=action, timestamp=ts)


def two_records() -> list[InteractionRecord]:
    short = InteractionRecord(
        user_index=0,
        history=(ev(0, ActionType.SCORE_CHANGE, 0.0), ev(7, ActionType.VIEW, 1.5)),
        target=ev(3, ActionType.APPLY_DECLINED, 2.0),
    )
    long = InteractionRecord(
        user_index=1,
        history=tuple(ev(i, ActionType.VIEW, float(i) * 2.0) for i in range(1, 7)),
        target=ev(9, ActionType.CREDIT_PULL, 14.0),
        future_window=(ev(9, ActionType.CREDIT_PULL, 14.0), ev(11, ActionType.VIEW, 15.0)),
    )
    return [short, long]


def score_change_last() -> InteractionRecord:
    return InteractionRecord(
        user_index=2,
        history=(ev(4, ActionType.VIEW, 0.0), ev(0, ActionType.SCORE_CHANGE, 3.0)),
        target=ev(5, ActionType.VIEW, 4.0),
    )


class TinyCausalEncoder(torch.nn.Module):
    """Minimal causal sequence encoder: shared embedding -> causal Pre-LN encoder.

    Stands in for any sequential model in this repo; the tests below prove the
    padding-side invariants that every such model inherits from the collator.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = MultiModalEmbedding(
            MultiModalEmbeddingConfig(num_items=20, d_model=16, max_len=6)
        )
        self.encoder = TransformerEncoder(
            TransformerEncoderConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32, causal=True)
        )

    def forward(self, batch: SequenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(batch.item_ids, batch.action_ids, batch.time_deltas, batch.attention_mask)
        hidden: torch.Tensor = self.encoder(x, batch.attention_mask)
        seq_rep = gather_last_real(hidden, batch.attention_mask, batch.padding_side)
        return hidden, seq_rep


def tiny_encoder() -> TinyCausalEncoder:
    return TinyCausalEncoder().eval()


# ------------------------------------------------------------------ padding layout


def test_left_padding_places_recent_event_last() -> None:
    batch = SequenceCollator(max_len=4, padding_side="left")(two_records())
    assert batch.item_ids.tolist() == [[0, 0, 0, 7], [3, 4, 5, 6]]
    assert batch.action_ids.tolist() == [[0, 0, 2, 1], [1, 1, 1, 1]]
    assert batch.attention_mask[:, -1].all()
    assert batch.lengths.tolist() == [2, 4]
    assert batch.target_item_ids.tolist() == [3, 9]
    assert batch.future_item_ids[1, :2].tolist() == [9, 11]
    assert batch.future_mask.sum().item() == 2


def test_right_padding_mirror() -> None:
    batch = SequenceCollator(max_len=4, padding_side="right")(two_records())
    assert batch.item_ids.tolist() == [[0, 7, 0, 0], [3, 4, 5, 6]]
    assert batch.action_ids.tolist() == [[2, 1, 0, 0], [1, 1, 1, 1]]
    assert batch.lengths.tolist() == [2, 4]
    assert batch.attention_mask.sum(dim=1).tolist() == [2, 4]


def test_mask_derived_from_action_not_item() -> None:
    batch = SequenceCollator(max_len=4)([score_change_last()])
    assert batch.item_ids[0, -1].item() == 0
    assert batch.action_ids[0, -1].item() == ActionType.SCORE_CHANGE.value
    assert bool(batch.attention_mask[0, -1])
    assert batch.attention_mask.sum().item() == batch.lengths[0].item() == 2
    # a mask derived from item ids would be wrong:
    assert (batch.item_ids != 0).sum().item() == 1


def test_time_deltas_log1p_safe() -> None:
    batch = SequenceCollator(max_len=4)(two_records())
    assert torch.all(batch.time_deltas >= 0)
    assert batch.time_deltas[0].tolist() == [0.0, 0.0, 0.0, 1.5]
    assert batch.time_deltas[1].tolist() == [0.0, 2.0, 2.0, 2.0]
    assert torch.isfinite(torch.log1p(batch.time_deltas)).all()


def test_collator_rejects_bad_padding_side() -> None:
    with pytest.raises(ValueError):
        SequenceCollator(max_len=4, padding_side="middle")  # type: ignore[arg-type]


# --------------------------------------------------------------------- embedding


def test_score_change_embedding_not_zero_but_pad_is() -> None:
    batch = SequenceCollator(max_len=4)([score_change_last()])
    emb = MultiModalEmbedding(MultiModalEmbeddingConfig(num_items=20, d_model=8, max_len=4))
    out = emb(batch.item_ids, batch.action_ids, batch.time_deltas, batch.attention_mask)
    norms = out[0].norm(dim=-1)
    assert torch.all(norms[:2] == 0)  # PAD slots exactly zero
    assert norms[2] > 0 and norms[3] > 0  # VIEW and SCORE_CHANGE both live


# ----------------------------------------------------------------------- seq_rep


def test_seq_rep_equals_last_real_hidden_left() -> None:
    model = tiny_encoder()
    hidden, seq_rep = model(SequenceCollator(max_len=6)(two_records() + [score_change_last()]))
    assert torch.equal(seq_rep, hidden[:, -1, :])


def test_seq_rep_gather_right_padding() -> None:
    model = tiny_encoder()
    batch = SequenceCollator(max_len=6, padding_side="right")(two_records())
    hidden, seq_rep = model(batch)
    for b in range(2):
        last = int(batch.lengths[b]) - 1
        assert torch.equal(seq_rep[b], hidden[b, last])


def test_seq_rep_invariant_to_padding_side() -> None:
    model = tiny_encoder()
    recs = two_records() + [score_change_last()]
    _, rep_l = model(SequenceCollator(max_len=6, padding_side="left")(recs))
    _, rep_r = model(SequenceCollator(max_len=6, padding_side="right")(recs))
    assert torch.allclose(rep_l, rep_r, atol=1e-5)


def test_gather_last_real_rejects_unknown_padding_side() -> None:
    hidden = torch.zeros(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        gather_last_real(hidden, mask, "middle")  # type: ignore[arg-type]


# ------------------------------------------------------------ impression collator


def test_impression_collator_shapes_and_funnel() -> None:
    n_items, n_users = 6, 3
    catalog = np.random.rand(n_items + 1, PRODUCT_FEATURE_DIM).astype(np.float32)
    catalog[0] = 0
    users = np.random.rand(n_users, USER_FEATURE_DIM).astype(np.float32)
    family = np.array([-1, 0, 1, 2, 3, 4, 0], dtype=np.int64)
    slates = [
        ImpressionSlate(
            user_index=1,
            slate_id=0,
            candidate_item_ids=(1, 2, 3),
            y_click=(1, 1, 0),
            y_apply=(1, 0, 0),
            y_approve=(1, 0, 0),
            p_click=(0.5, 0.2, 0.1),
            p_apply=(0.4, 0.3, 0.2),
            p_approve=(0.9, 0.0, 0.5),
            payouts=(100.0, 200.0, 300.0),
            amounts=(5000.0, 0.0, 0.0),
            eligible=(True, False, True),
        ),
        ImpressionSlate(
            user_index=2,
            slate_id=1,
            candidate_item_ids=(4, 5),
            y_click=(0, 0),
            y_apply=(0, 0),
            y_approve=(0, 0),
            p_click=(0.1, 0.1),
            p_apply=(0.1, 0.1),
            p_approve=(0.1, 0.1),
            payouts=(1.0, 2.0),
            amounts=(0.0, 0.0),
            eligible=(True, True),
        ),
    ]
    batch = ImpressionCollator(catalog, users, family)(slates)
    assert batch.candidate_item_ids.shape == (2, 3)
    assert batch.candidate_mask.tolist() == [[True, True, True], [True, True, False]]
    assert batch.family_ids.tolist() == [[0, 1, 2], [3, 4, -1]]
    assert batch.tabular.shape == (2, 3, USER_FEATURE_DIM + PRODUCT_FEATURE_DIM)
    assert torch.allclose(batch.tabular[0, 1, :USER_FEATURE_DIM], torch.tensor(users[1]))
    assert torch.allclose(batch.tabular[0, 1, USER_FEATURE_DIM:], torch.tensor(catalog[2]))
    assert torch.all(batch.candidate_features[1, 2] == 0)
    assert batch.y_click.dtype == torch.float32
    assert batch.eligible.dtype == torch.bool
    assert batch.y_approve[0].tolist() == [1.0, 0.0, 0.0]
    assert batch.amounts[0, 0].item() == 5000.0
