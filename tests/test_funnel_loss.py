"""Unified funnel loss (D3), stable log-space helpers, negative down-sampling (D2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from recsys import seed_everything
from recsys.losses.funnel_loss import (
    FunnelLossConfig,
    UnifiedFunnelLoss,
    naive_entire_space_bce,
)
from recsys.losses.stable import bce_from_log_prob, log1mexp, logsigmoid, masked_mean
from recsys.training.downsampling import NegativeDownsampler


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


def _batch(b: int = 2, k: int = 6, scale: float = 2.0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    z = {
        n: (torch.randn(b, k, generator=gen) * scale).requires_grad_(True)
        for n in ("z1", "z2", "z3")
    }
    y_click = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    y_apply = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    y_approve_oracle = torch.tensor([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    observed = torch.ones(b, k, dtype=torch.bool)
    observed[1, 0] = False  # row (1,0) is PENDING
    y_approve = torch.where(observed, y_approve_oracle, torch.zeros_like(y_approve_oracle))
    weight = observed.to(torch.float32)  # "drop"
    mask = torch.ones(b, k, dtype=torch.bool)
    mask[1, 5] = False  # empty slot
    amounts = torch.tensor([[5000.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    amount_logits = torch.randn(b, k, 3, generator=gen).requires_grad_(True)
    return {
        **z,
        "y_click": y_click,
        "y_apply": y_apply,
        "y_approve": y_approve,
        "approve_weight": weight,
        "approve_observed": observed,
        "candidate_mask": mask,
        "amounts": amounts,
        "amount_logits": amount_logits,
    }


def _call(loss: UnifiedFunnelLoss, d: dict[str, torch.Tensor], **kw: object):  # type: ignore[no-untyped-def]
    return loss(
        d["z1"], d["z2"], d["z3"], d["y_click"], d["y_apply"], d["y_approve"],
        d["approve_weight"], d["approve_observed"], d["candidate_mask"],
        amount_logits=d["amount_logits"], amounts=d["amounts"], **kw,
    )  # fmt: skip


# ------------------------------------------------------------------ stable helpers


def test_log1mexp_matches_reference_and_is_finite_at_extremes() -> None:
    x = torch.tensor(
        [-1e-8, -1e-4, -0.1, -0.5, -0.69, -0.70, -2.0, -30.0, -100.0], dtype=torch.float64
    )
    ref = torch.log(1 - torch.exp(x[2:]))
    assert torch.allclose(log1mexp(x[2:]), ref, rtol=1e-9)
    out = log1mexp(x)
    assert torch.isfinite(out).all() and (out <= 0).all()
    x_grad = x.clone().requires_grad_(True)
    log1mexp(x_grad).sum().backward()
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()
    # bce_from_log_prob at log p = -1e-8 (p -> 1) with y = 0 is large but finite
    assert torch.isfinite(bce_from_log_prob(torch.tensor(-1e-8), torch.tensor(0.0)))


def test_masked_mean_normalizes_by_mask_sum_and_handles_empty() -> None:
    v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    m = torch.tensor([[True, False, True], [False, False, False]])
    assert float(masked_mean(v, m).detach()) == 2.0
    empty = masked_mean(v, torch.zeros_like(m))
    assert float(empty.detach()) == 0.0
    empty.backward()
    assert v.grad is not None


# ------------------------------------------------------------------- funnel loss


def test_log_space_product_equals_naive_product_at_moderate_logits() -> None:
    d = _batch(scale=1.0)
    loss = UnifiedFunnelLoss()
    t = _call(loss, d)
    p1, p2, p3 = torch.sigmoid(d["z1"]), torch.sigmoid(d["z2"]), torch.sigmoid(d["z3"])
    naive_ctcvr = masked_mean(naive_entire_space_bce(p1 * p2, d["y_apply"]), d["candidate_mask"])
    w_prime = torch.where(
        d["y_apply"] > 0.5,
        d["approve_weight"] * d["approve_observed"],
        torch.ones_like(d["approve_weight"]),
    )
    naive_ctcavr = masked_mean(
        w_prime * naive_entire_space_bce(p1 * p2 * p3, d["y_approve"]), d["candidate_mask"]
    )
    assert torch.allclose(t.ctcvr, naive_ctcvr, atol=1e-5)
    assert torch.allclose(t.ctcavr, naive_ctcavr, atol=1e-5)
    # conditional terms are plain BCEs on their sub-populations
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    m_c = d["candidate_mask"] & (d["y_click"] > 0.5)
    assert torch.allclose(t.apply, masked_mean(bce(d["z2"], d["y_apply"], reduction="none"), m_c))
    assert t.num_rows == 11 and t.num_clicked == 5 and t.num_resolved_applications == 2
    assert t.num_amount_rows == 1
    assert float(t.total) == pytest.approx(
        float(t.click + t.apply + t.approve + t.ctcvr + t.ctcavr + 0.1 * t.amount), abs=1e-6
    )


def test_finite_loss_and_gradients_at_extreme_logits() -> None:
    for sign in (1.0, -1.0):
        d = _batch(scale=0.0)
        for n in ("z1", "z2", "z3"):
            d[n] = (torch.full((2, 6), sign * 30.0)).requires_grad_(True)
        loss = UnifiedFunnelLoss()
        t = _call(loss, d)
        assert torch.isfinite(t.total)
        t.total.backward()
        for n in ("z1", "z2", "z3"):
            g = d[n].grad
            assert g is not None and torch.isfinite(g).all()


def test_pending_rows_give_zero_gradient_to_z3() -> None:
    d = _batch()
    t = _call(UnifiedFunnelLoss(), d)
    t.total.backward()
    g3 = d["z3"].grad
    assert g3 is not None
    assert g3[1, 0] == 0.0  # pending: masked from L_approve and L_ctcavr (w' = 0)
    assert g3[0, 0] != 0.0  # resolved approved application
    # non-applied rows do get gradient through the entire-space term
    assert g3[0, 3] != 0.0
    # the same row still trains click / apply and the click->apply product
    g1, g2 = d["z1"].grad, d["z2"].grad
    assert g1 is not None and g2 is not None and g1[1, 0] != 0.0 and g2[1, 0] != 0.0
    # under "negative" weighting the pending row would train z3 toward 0
    d2 = _batch()
    d2["approve_weight"] = torch.ones_like(d2["approve_weight"])
    d2["approve_observed"] = torch.ones_like(d2["approve_observed"])
    _call(UnifiedFunnelLoss(), d2).total.backward()
    assert d2["z3"].grad is not None and d2["z3"].grad[1, 0] != 0.0


def test_mask_sum_normalization_and_empty_sub_populations() -> None:
    d = _batch()
    t = _call(UnifiedFunnelLoss(), d)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    rows = d["candidate_mask"] & (d["y_apply"] > 0.5) & d["approve_observed"]
    per_row = bce(d["z3"], d["y_approve"], reduction="none")[rows]
    assert torch.allclose(t.approve, per_row.mean())  # mean over the 2 resolved rows, not B*K
    # a batch with no applications at all: approve terms are 0, total finite
    d0 = _batch()
    d0["y_apply"] = torch.zeros_like(d0["y_apply"])
    d0["y_approve"] = torch.zeros_like(d0["y_approve"])
    d0["amounts"] = torch.zeros_like(d0["amounts"])
    t0 = _call(UnifiedFunnelLoss(), d0)
    assert float(t0.approve) == 0.0 and float(t0.amount) == 0.0 and torch.isfinite(t0.total)
    t0.total.backward()
    assert d0["z3"].grad is not None and torch.isfinite(d0["z3"].grad).all()


def test_entire_space_terms_supervise_z2_on_unclicked_rows_and_detach_flag() -> None:
    d = _batch()
    t = _call(
        UnifiedFunnelLoss(
            FunnelLossConfig(
                lambda_apply=0.0, lambda_click=0.0, lambda_approve=0.0, lambda_amount=0.0
            )
        ),
        d,
    )
    t.total.backward()
    g2 = d["z2"].grad
    assert g2 is not None and g2[0, 4] != 0.0  # non-clicked row gets apply-tower gradient
    g1 = d["z1"].grad
    assert g1 is not None and g1.abs().sum() > 0  # and z1 is regularized by the product terms
    d2 = _batch()
    cfg = FunnelLossConfig(
        lambda_apply=0.0,
        lambda_click=0.0,
        lambda_approve=0.0,
        lambda_amount=0.0,
        detach_upstream=True,
    )
    _call(UnifiedFunnelLoss(cfg), d2).total.backward()
    assert d2["z1"].grad is not None and d2["z1"].grad.abs().sum() == 0.0


@pytest.mark.parametrize("ssb_mode", ["none", "ips", "dr"])
@pytest.mark.parametrize("balancing", ["fixed", "running_mean", "uncertainty"])
def test_ssb_and_balancing_variants_run(ssb_mode: str, balancing: str) -> None:
    d = _batch()
    cfg = FunnelLossConfig(ssb_mode=ssb_mode, loss_balancing=balancing)  # type: ignore[arg-type]
    loss = UnifiedFunnelLoss(cfg).train()
    imp = torch.randn(2, 6, requires_grad=True) if ssb_mode == "dr" else None
    t = _call(loss, d, imputation_logits=imp)
    assert torch.isfinite(t.total)
    t.total.backward()
    assert d["z2"].grad is not None and torch.isfinite(d["z2"].grad).all()
    if ssb_mode == "dr":
        assert imp is not None and imp.grad is not None and float(t.imputation) >= 0.0
    if balancing == "uncertainty":
        assert loss.log_var.grad is not None
    if balancing == "running_mean":
        assert (loss.ema >= 0).all()
        assert set(t.effective_weights) == {
            "click",
            "apply",
            "approve",
            "ctcvr",
            "ctcavr",
            "amount",
        }
    if ssb_mode == "dr":
        with pytest.raises(ValueError):
            _call(loss, d)


def test_train_mask_removes_rows_from_every_entire_space_term() -> None:
    d = _batch()
    keep = torch.ones(2, 6, dtype=torch.bool)
    keep[0, 3:] = False
    t = _call(UnifiedFunnelLoss(), d, train_mask=keep)
    assert t.num_rows == 8
    d_sub = _batch()
    d_sub["candidate_mask"] = d_sub["candidate_mask"] & keep
    t_sub = _call(UnifiedFunnelLoss(), d_sub)
    for name in ("click", "apply", "approve", "ctcvr", "ctcavr"):
        assert torch.allclose(getattr(t, name), getattr(t_sub, name))


# -------------------------------------------------------------- down-sampling (D2)


def test_negative_downsampler_keeps_all_positives() -> None:
    ds = NegativeDownsampler(rate=0.25, seed=0)
    gen = torch.Generator().manual_seed(1)  # must differ from the sampler's seed
    y = (torch.rand(200, 50, generator=gen) < 0.1).float()
    mask = torch.ones(200, 50, dtype=torch.bool)
    keep = ds.train_mask(y, mask)
    assert keep[y > 0.5].all()
    kept_neg = keep[y < 0.5].float().mean()
    assert abs(float(kept_neg) - 0.25) < 0.02
    assert ds.logit_correction == math.log(0.25)
    assert NegativeDownsampler(rate=1.0).train_mask(y, mask).all()
    with pytest.raises(ValueError):
        NegativeDownsampler(rate=0.0)


def test_logit_correction_recovers_true_click_probability() -> None:
    """Fit a one-feature logistic model on down-sampled Bernoulli data: ``sigmoid(z + log r)``
    matches the generator's ``p_click`` while the uncorrected model does not."""
    rng = np.random.default_rng(0)
    n = 60_000
    x = rng.normal(size=n)
    p_true = 1.0 / (1.0 + np.exp(-(-2.5 + 1.5 * x)))
    y = (rng.random(n) < p_true).astype(np.float64)
    r = 0.25
    keep = (y > 0.5) | (rng.random(n) < r)
    xt = torch.tensor(x[keep], dtype=torch.float32)
    yt = torch.tensor(y[keep], dtype=torch.float32)
    w = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(400):
        z = w[0] * xt + w[1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    grid = torch.linspace(-2.0, 2.0, 9)
    z_train = (w[0] * grid + w[1]).detach()
    p_grid = 1.0 / (1.0 + np.exp(-(-2.5 + 1.5 * grid.numpy())))
    corrected = torch.sigmoid(z_train + math.log(r)).numpy()
    uncorrected = torch.sigmoid(z_train).numpy()
    assert np.abs(corrected - p_grid).max() < 0.03
    assert np.abs(uncorrected - p_grid).max() > 0.1
    # the slope is unaffected by down-sampling, only the intercept shifts by -log r
    assert abs(float(w[0]) - 1.5) < 0.1 and abs(float(w[1]) - (-2.5 - math.log(r))) < 0.1


def test_logsigmoid_alias() -> None:
    z = torch.tensor([-50.0, 0.0, 50.0])
    assert torch.allclose(logsigmoid(z), torch.nn.functional.logsigmoid(z))
