"""Stage 1 compliance and retrieval-metric tests (TIGER, trie, eligibility, Recall@K)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from recsys import seed_everything
from recsys.data.schema import FinancialProduct, ProductFamily, UserProfile
from recsys.layers.prefix_trie import SemanticIdTrie
from recsys.layers.rq_vae import RQVAE, RQVAEConfig
from recsys.metrics.ranking_metrics import recall_at_k
from recsys.metrics.retrieval_baselines import eligible_popularity_top_k, eligible_random_top_k
from recsys.models.tiger.model import (
    NUM_CONTEXT_TOKENS,
    SCORE_CHANGE_TOK,
    STATE_TOK_OFFSET,
    TIER_TOK_OFFSET,
    TIGER,
    SemanticIdTokenizer,
    TIGERConfig,
)
from recsys.serving.eligibility_engine import (
    ComplianceViolation,
    EligibilityEngine,
    UserGateArrays,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


def product(item_id: int = 1, **overrides: object) -> FinancialProduct:
    base: dict[str, object] = dict(
        item_id=item_id,
        name=f"p{item_id}",
        family=ProductFamily.PERSONAL_LOAN,
        min_fico=660,
        max_dti=0.40,
        min_annual_income=40_000.0,
        licensed_states=frozenset({"CA", "TX"}),
        apr=0.12,
        annual_fee=0.0,
        reward_rate=0.0,
        term_months=36,
        partner_payout=250.0,
    )
    base.update(overrides)
    return FinancialProduct.model_validate(base)


def user(**overrides: object) -> UserProfile:
    base: dict[str, object] = dict(
        user_index=0, fico=700, dti=0.30, annual_income=60_000.0, state="CA"
    )
    base.update(overrides)
    return UserProfile.model_validate(base)


def _codes(num_items: int, base_sizes: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    codes = torch.stack(
        [torch.randint(0, s, (num_items + 1,), generator=gen) for s in base_sizes], 1
    )
    seen: dict[tuple[int, ...], int] = {}
    dedup = torch.zeros(num_items + 1, 1, dtype=torch.int64)
    for i in range(1, num_items + 1):
        key = tuple(codes[i].tolist())
        dedup[i, 0] = seen.get(key, 0)
        seen[key] = int(dedup[i, 0]) + 1
    codes = torch.cat([codes, dedup], dim=1)
    codes[0] = 0
    return codes


def _tiger_setup(
    num_items: int = 12,
) -> tuple[TIGER, SemanticIdTokenizer, torch.Tensor, TIGERConfig, SemanticIdTrie]:
    base_sizes = (4, 4, 4)
    codes = _codes(num_items, base_sizes)
    level_sizes: tuple[int, ...] = (*base_sizes, int(codes[1:, -1].max()) + 1)
    cfg = TIGERConfig(
        num_items=num_items, level_sizes=level_sizes, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, max_history_items=5,
    )  # fmt: skip
    tok = SemanticIdTokenizer(codes, cfg)
    trie = SemanticIdTrie.build(codes[1:], np.arange(1, num_items + 1), level_sizes)
    return TIGER(cfg).eval(), tok, codes, cfg, trie


def _history(
    num_items: int, b: int = 5, L: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    items = torch.randint(1, num_items + 1, (b, L), generator=gen)
    acts = torch.randint(1, 7, (b, L), generator=gen)
    mask = torch.ones(b, L, dtype=torch.bool)
    mask[0, :2] = False
    items[0, :2] = 0
    acts[0, :2] = 0
    items[:, 2] = 0
    acts[:, 2] = 2  # SCORE_CHANGE in every row
    tier = torch.randint(0, 5, (b,), generator=gen)
    state = torch.randint(0, 51, (b,), generator=gen)
    return items, acts, mask, tier, state


# ------------------------------------------------------------- eligibility engine


def _catalog() -> list[FinancialProduct]:
    return [
        product(1),
        product(2, min_fico=720),
        product(3, max_dti=0.25),
        product(4, min_annual_income=80_000.0),
        product(5, licensed_states=frozenset({"TX"})),
        product(6, family=ProductFamily.MORTGAGE, min_fico=640),
        product(7, family=ProductFamily.CREDIT_CARD),
        product(8, family=ProductFamily.MORTGAGE),
    ]


def test_eligibility_mask_matches_scalar_reference() -> None:
    products = _catalog()
    engine = EligibilityEngine.from_products(products)
    users = [
        user(user_index=0),
        user(user_index=1, fico=730, dti=0.2, annual_income=90_000.0, state="TX"),
        user(user_index=2, fico=650, state="NY"),
        user(
            user_index=3, held_product_ids=(1,), pending_product_ids=(7,), pending_family_ids=(0,)
        ),
        user(user_index=4, pending_product_ids=(6,), pending_family_ids=(4,)),
    ]
    m = engine.mask_for_users(users)
    assert m.shape == (5, 9) and not m[:, 0].any()
    for ui, u in enumerate(users):
        for p in products:
            gates_ok = (
                u.fico >= p.min_fico
                and u.dti <= p.max_dti
                and u.annual_income >= p.min_annual_income
                and u.state in p.licensed_states
            )
            ctx_ok = (
                p.item_id not in u.held_product_ids
                and p.item_id not in u.pending_product_ids
                and not (p.family is ProductFamily.MORTGAGE and 4 in u.pending_family_ids)
            )
            assert bool(m[ui, p.item_id]) == (gates_ok and ctx_ok), (ui, p.item_id)
        assert np.array_equal(engine.mask_for_user(u), m[ui])
    # pending mortgage masks the *whole* family (policy "mask"), pending card does not
    assert not m[4, 6] and not m[4, 8]
    assert not m[3, 1] and not m[3, 7]  # held + pending, item level
    assert m[3, 8]  # cards are "penalize": no family-level mask for user 3


@pytest.mark.parametrize("rule", ["min_fico", "max_dti", "min_annual_income", "licensed_state"])
def test_each_gate_individually(rule: str) -> None:
    overrides: dict[str, dict[str, object]] = {
        "min_fico": {"min_fico": 701},
        "max_dti": {"max_dti": 0.29},
        "min_annual_income": {"min_annual_income": 60_001.0},
        "licensed_state": {"licensed_states": frozenset({"TX"})},
    }
    override = overrides[rule]
    engine = EligibilityEngine.from_products([product(1), product(2, **override)])
    u = user()
    assert engine.mask_for_user(u).tolist() == [False, True, False]
    ex = engine.explain(u, 2)
    assert not ex[rule] and all(v for k, v in ex.items() if k != rule)
    with pytest.raises(ComplianceViolation):
        engine.assert_all_eligible(u, [1, 2])
    engine.assert_all_eligible(u, [1])


def test_context_rules_explain_filter_and_assert() -> None:
    engine = EligibilityEngine.from_products(_catalog())
    u = user(held_product_ids=(1,), pending_product_ids=(6,), pending_family_ids=(4,))
    ex = engine.explain(u, 1)
    assert not ex["not_held"] and ex["not_pending"] and ex["pending_family_policy"]
    assert not engine.explain(u, 6)["not_pending"]
    assert not engine.explain(u, 8)["pending_family_policy"]  # sibling mortgage blocked by policy
    assert engine.explain(u, 7)["pending_family_policy"]
    kept = engine.filter_candidates(u, np.array([1, 6, 8, 7, 0, 99, 3], dtype=np.int64))
    assert kept.tolist() == [7]
    with pytest.raises(ComplianceViolation):
        engine.assert_all_eligible(u, [7, 8])
    engine.assert_all_eligible(u, [7])
    # policy override: ignore the family rule
    engine2 = EligibilityEngine.from_products(_catalog(), {ProductFamily.MORTGAGE: "ignore"})
    assert engine2.mask_for_user(u)[8]
    arrays = UserGateArrays.from_users([u], engine.num_items)
    assert arrays.pending[0, 6] and arrays.pending_families[0, 4] and arrays.held[0, 1]


# ----------------------------------------------------------------------- trie


def test_trie_vectorized_mask_matches_brute_force() -> None:
    n = 40
    codes = _codes(n, (3, 3, 3), seed=5)
    level_sizes = (3, 3, 3, int(codes[1:, -1].max()) + 1)
    trie = SemanticIdTrie.build(codes[1:], np.arange(1, n + 1), level_sizes)
    gen = torch.Generator().manual_seed(2)
    allowed = torch.rand(6, n + 1, generator=gen) < 0.5
    allowed[:, 0] = False
    for level in range(4):
        prefixes = (
            torch.stack(
                [codes[int(torch.randint(1, n + 1, (1,), generator=gen))][:level] for _ in range(6)]
            )
            if level > 0
            else torch.zeros((6, 0), dtype=torch.int64)
        )
        got = trie.allowed_children_batch(prefixes, allowed)
        for i in range(6):
            expect = np.zeros(level_sizes[level], dtype=bool)
            for item in range(1, n + 1):
                if allowed[i, item] and torch.equal(codes[item, :level], prefixes[i]):
                    expect[int(codes[item, level])] = True
            assert got[i].numpy().tolist() == expect.tolist()
    # dict walk agrees with the vectorized mask when everything is allowed
    all_allowed = torch.ones((1, n + 1), dtype=torch.bool)
    all_allowed[0, 0] = False
    for item in range(1, n + 1):
        for level in range(4):
            pre = codes[item, :level]
            got_np = trie.allowed_children_batch(pre.unsqueeze(0), all_allowed)[0].numpy()
            assert got_np.tolist() == trie.allowed_next(pre.tolist()).tolist()
    # full-code item lookup
    items = trie.items_for_batch(codes[1:])
    assert items.tolist() == list(range(1, n + 1))
    assert int(trie.items_for_batch(torch.full((1, 4), 2))[0]) in (-1, *range(1, n + 1))
    assert trie.logit_mask(
        torch.zeros((1, 0), dtype=torch.int64), all_allowed, 100, (50, 60, 70, 80)
    ).shape == (1, 100)


def test_semantic_ids_unique_after_disambiguation() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 6)
    x[:32] = x[0]  # 32 identical rows collide on every code
    model = RQVAE(RQVAEConfig(input_dim=6, latent_dim=4, num_levels=2, codebook_size=4)).eval()
    sids = model.assign_semantic_ids(x)
    assert sids.shape == (64, 3)
    assert len({tuple(r) for r in sids.tolist()}) == 64
    assert int(sids[:32, -1].max()) == 31 and int(sids[:32, -1].min()) == 0


# ----------------------------------------------------------------------- TIGER


def test_tokenizer_layout_context_and_score_change() -> None:
    model, tok, codes, cfg, trie = _tiger_setup()
    items, acts, mask, tier, state = _history(12)
    ht, ha, hm = tok.encode_history(items, acts, mask, tier, state)
    t = cfg.num_levels
    assert ht.shape == (5, NUM_CONTEXT_TOKENS + 5 * t)
    assert ht[:, 0].tolist() == (TIER_TOK_OFFSET + tier).tolist()
    assert ht[:, 1].tolist() == (STATE_TOK_OFFSET + state).tolist()
    assert hm[:, :2].all() and (ha[:, :2] == 0).all()
    # left PAD preserved for row 0, SCORE_CHANGE slot expanded to the special token
    assert (ht[0, 2 : 2 + 2 * t] == 0).all() and not hm[0, 2 : 2 + 2 * t].any()
    assert (ht[:, 2 + 2 * t : 2 + 3 * t] == SCORE_CHANGE_TOK).all()
    assert (ha[:, 2 + 2 * t : 2 + 3 * t] == 2).all()
    last = ht[1, -t:]
    assert torch.equal(tok.tokens_to_codes(last), codes[items[1, -1]])


def test_tiger_beams_are_all_eligible_and_unique() -> None:
    n = 12
    model, tok, codes, cfg, trie = _tiger_setup(n)
    items, acts, mask, tier, state = _history(n)
    ht, ha, hm = tok.encode_history(items, acts, mask, tier, state)
    gen = torch.Generator().manual_seed(3)
    allowed = torch.rand(5, n + 1, generator=gen) < 0.6
    allowed[:, 0] = False
    allowed[4] = False
    allowed[4, 3] = True  # exactly one eligible item
    out = model.generate(ht, ha, hm, trie, tok, allowed, beam_size=8)
    assert out.item_ids.shape == (5, 8) and out.codes.shape == (5, 8, 4)
    for i in range(5):
        live = out.item_ids[i][out.item_ids[i] > 0].tolist()
        assert len(live) == len(set(live))
        assert set(live) <= set(np.flatnonzero(allowed[i].numpy()).tolist())
        assert len(live) == min(8, int(allowed[i].sum()))
        assert torch.isfinite(out.log_probs[i][out.item_ids[i] > 0]).all()
        assert (out.log_probs[i][out.item_ids[i] < 0] == float("-inf")).all()
        for k in range(len(live)):
            assert torch.equal(out.codes[i, k], codes[live[k]])
    assert out.item_ids[4].tolist()[:1] == [3]
    # beams are sorted by log-prob
    lp = out.log_probs[out.log_probs > float("-inf")]
    assert torch.all(out.log_probs[:, :-1] >= out.log_probs[:, 1:]) and lp.numel() > 0


def test_tiger_empty_eligibility_returns_no_items() -> None:
    n = 12
    model, tok, codes, cfg, trie = _tiger_setup(n)
    items, acts, mask, tier, state = _history(n, b=2)
    ht, ha, hm = tok.encode_history(items, acts, mask, tier, state)
    allowed = torch.zeros((2, n + 1), dtype=torch.bool)
    out = model.generate(ht, ha, hm, trie, tok, allowed, beam_size=4)
    assert (out.item_ids == -1).all() and (out.log_probs == float("-inf")).all()


def test_held_and_pending_items_never_generated_even_when_top_scored() -> None:
    """Overfit the model onto target items, then mark them held / pending: the beam must
    exclude them although they are the model's top choice."""
    n = 12
    model, tok, codes, cfg, trie = _tiger_setup(n)
    model.train()
    items, acts, mask, tier, state = _history(n, b=4)
    ht, ha, hm = tok.encode_history(items, acts, mask, tier, state)
    targets = torch.tensor([1, 5, 9, 12])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        loss = model.next_sid_loss(ht, ha, hm, codes[targets], tok).target
        opt.zero_grad()
        loss.backward()
        opt.step()
        if float(loss.detach()) < 0.05:
            break
    model.eval()
    everything = torch.ones((4, n + 1), dtype=torch.bool)
    everything[:, 0] = False
    top = model.generate(ht, ha, hm, trie, tok, everything, beam_size=3)
    assert top.item_ids[:, 0].tolist() == targets.tolist()
    # products 1, 5, 9, 12 are held / pending for the respective users
    users = [
        user(user_index=0, held_product_ids=(1,)),
        user(user_index=1, pending_product_ids=(5,), pending_family_ids=(2,)),
        user(user_index=2, held_product_ids=(9,)),
        user(user_index=3, pending_product_ids=(12,), pending_family_ids=(2,)),
    ]
    engine = EligibilityEngine.from_products(
        [product(i, min_fico=300, max_dti=1.0, min_annual_income=0.0,
                 licensed_states=frozenset({"CA"})) for i in range(1, n + 1)]
    )  # fmt: skip
    allowed = torch.as_tensor(engine.mask_for_users(users))
    out = model.generate(ht, ha, hm, trie, tok, allowed, beam_size=3)
    for i, tgt in enumerate(targets.tolist()):
        assert tgt not in out.item_ids[i].tolist()
        assert (out.item_ids[i] > 0).sum() == 3
    # the post-retrieval gate agrees with the trie mask
    for i, u in enumerate(users):
        ids = out.item_ids[i][out.item_ids[i] > 0].numpy()
        assert engine.filter_candidates(u, ids).tolist() == ids.tolist()
        engine.assert_all_eligible(u, ids.tolist())


def test_next_sid_loss_history_term_counts_only_valid_slots() -> None:
    n = 12
    model, tok, codes, cfg, trie = _tiger_setup(n)
    items, acts, mask, tier, state = _history(n, b=5)
    ht, ha, hm = tok.encode_history(items, acts, mask, tier, state)
    loss = model.next_sid_loss(ht, ha, hm, codes[[1, 2, 3, 4, 5]], tok)
    # slots: row 0 = [PAD, PAD, SC, x, x]; rows 1-4 = [x, x, SC, x, x]
    # predictable slot j+1: real(j) & product(j+1).  A SCORE_CHANGE slot is real, so it can
    # predict its successor but is never itself a target:
    #   row 0: (2->3), (3->4) = 2;  rows 1-4: (0->1), (2->3), (3->4) = 3 each
    assert loss.num_history_predictions == 2 + 4 * 3
    assert torch.isfinite(loss.total) and loss.history > 0 and loss.target > 0
    cfg_off = TIGERConfig(**{**cfg.__dict__, "train_all_positions": False})
    model_off = TIGER(cfg_off)
    model_off.load_state_dict(model.state_dict())
    loss_off = model_off.next_sid_loss(ht, ha, hm, codes[[1, 2, 3, 4, 5]], tok)
    assert loss_off.num_history_predictions == 0 and float(loss_off.history) == 0.0
    assert torch.allclose(loss_off.target, loss.target)


# -------------------------------------------------------------- recall@k (D7)


def test_recall_at_k_hand_computed_cases() -> None:
    cand = np.array(
        [
            [3, 7, 9, 2, -1],  # user 0: beam of 4
            [1, 4, 5, 6, 8],  # user 1
            [2, 3, -1, -1, -1],  # user 2: short beam
            [9, 8, 7, 6, 5],  # user 3: no eligible positive -> skipped
        ]
    )
    positives = np.array([[3, 9, 11], [4, 12, 0], [2, 3, 5], [1, 0, 0]])
    valid = np.array(
        [
            [True, True, False],  # 11 is ineligible -> excluded from the denominator
            [True, True, False],  # 0 = padding
            [True, True, True],
            [False, False, False],  # positive 1 ineligible
        ]
    )
    r = recall_at_k(cand, positives, valid, k=5)
    assert r.num_users == 3 and r.num_excluded_positives == 2
    assert np.isnan(r.per_user[3])
    assert r.per_user[:3].tolist() == [1.0, 0.5, 2 / 3]
    assert r.mean == pytest.approx((1.0 + 0.5 + 2 / 3) / 3)
    r2 = recall_at_k(cand, positives, valid, k=2)
    assert r2.per_user[:3].tolist() == [0.5, 0.5, 2 / 3]
    r_none = recall_at_k(cand[3:], positives[3:], valid[3:], k=5)
    assert r_none.num_users == 0 and np.isnan(r_none.mean)
    with pytest.raises(ValueError):
        recall_at_k(cand, positives[:2], valid[:2], k=5)


def test_retrieval_baselines_respect_mask() -> None:
    counts = np.array([0, 5, 1, 9, 2, 7], dtype=np.int64)
    allowed = np.array([[False, True, True, False, True, True], [False] * 6])
    pop = eligible_popularity_top_k(counts, allowed, 3)
    assert pop[0].tolist() == [5, 1, 4] and pop[1].tolist() == [-1, -1, -1]
    rnd = eligible_random_top_k(allowed, 3, np.random.default_rng(0))
    assert set(rnd[0].tolist()) <= {1, 2, 4, 5} and len(set(rnd[0].tolist())) == 3
    assert rnd[1].tolist() == [-1, -1, -1]
