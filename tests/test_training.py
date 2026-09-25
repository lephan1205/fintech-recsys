"""Training infrastructure (D8): optimizer groups, schedule, early stopping, splits,
decision register, and an end-to-end smoke of ``train_all`` on a tiny dataset."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from recsys import seed_everything
from recsys.data.synthetic_generator import GeneratorConfig, SyntheticFintechDataGenerator
from recsys.layers.rq_vae import RQVAE, RQVAEConfig
from recsys.serving.calibration import CalibrationConfig
from recsys.serving.pipeline import PipelineArtifacts, RecommendationPipeline
from recsys.training.config import (
    PRM_OPTIMIZER,
    RANKER_OPTIMIZER,
    RQVAE_OPTIMIZER,
    TIGER_OPTIMIZER,
    ModelConfig,
    OptimizerConfig,
    SplitConfig,
    TrainingConfig,
    decision_register,
    register_as_dict,
)
from recsys.training.early_stopping import EarlyStopping
from recsys.training.optim import WarmupCosine, build_optimizer, lr_at, split_decay_params
from recsys.training.splits import SPLIT_NAMES, split_by_user
from recsys.training.trainers import train_all
from recsys.valuation.utility import UtilityConfig


@pytest.fixture(autouse=True)
def _seed() -> None:
    seed_everything(0)
    torch.set_num_threads(2)


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(10, 4)
        self.lin = nn.Linear(4, 4)
        self.ln = nn.LayerNorm(4)
        self.vec = nn.Parameter(torch.zeros(4))


def test_build_optimizer_parameter_groups() -> None:
    model = Toy()
    decay, no_decay = split_decay_params(model)
    assert decay == ["lin.weight"]
    assert set(no_decay) == {"emb.weight", "lin.bias", "ln.weight", "ln.bias", "vec"}
    opt = build_optimizer(model, TIGER_OPTIMIZER)
    assert isinstance(opt, torch.optim.AdamW)
    wd = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    assert wd[id(model.lin.weight)] == 0.01
    for p in (model.emb.weight, model.lin.bias, model.ln.weight, model.ln.bias, model.vec):
        assert wd[id(p)] == 0.0
    assert opt.param_groups[0]["betas"] == (0.9, 0.98)
    adam = build_optimizer(model, RQVAE_OPTIMIZER)
    assert isinstance(adam, torch.optim.Adam) and all(
        g["weight_decay"] == 0.0 for g in adam.param_groups
    )


def test_rqvae_codebooks_are_not_optimizer_parameters() -> None:
    model = RQVAE(RQVAEConfig(input_dim=6, latent_dim=4, num_levels=2, codebook_size=4))
    opt = build_optimizer(model, RQVAE_OPTIMIZER)
    ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(model.codebooks) not in ids and id(model.ema_sum) not in ids
    assert all(n not in ("codebooks", "ema_count", "ema_sum") for n, _ in model.named_parameters())


def test_warmup_cosine_schedule_peak_and_floor() -> None:
    cfg = TIGER_OPTIMIZER
    assert lr_at(0, cfg) == pytest.approx(cfg.lr / cfg.warmup_steps)
    assert lr_at(cfg.warmup_steps, cfg) == pytest.approx(cfg.lr)
    assert lr_at(cfg.total_steps, cfg) == pytest.approx(0.1 * cfg.lr)
    mid = lr_at((cfg.warmup_steps + cfg.total_steps) // 2, cfg)
    assert 0.1 * cfg.lr < mid < cfg.lr
    assert lr_at(10**6, cfg) == pytest.approx(0.1 * cfg.lr)
    # constant schedule for the RQ-VAE
    assert lr_at(0, RQVAE_OPTIMIZER) == lr_at(1999, RQVAE_OPTIMIZER) == RQVAE_OPTIMIZER.lr
    model = Toy()
    opt = build_optimizer(model, cfg)
    sched = WarmupCosine(opt, cfg)
    assert sched.lr == pytest.approx(lr_at(0, cfg))
    for _ in range(cfg.warmup_steps):
        sched.step()
    assert sched.lr == pytest.approx(cfg.lr)
    with pytest.raises(ValueError):
        OptimizerConfig("adamw", 1e-3, 0.0, 10, 5, 0.1, None, 1, 1)


def test_early_stopping_triggers_after_patience_and_restores_best() -> None:
    model = Toy()
    stop = EarlyStopping(patience=3, mode="min")
    assert stop.update(1.0, 1, model)
    with torch.no_grad():
        model.vec.fill_(9.0)
    assert stop.update(0.5, 2, model)  # best, vec == 9
    with torch.no_grad():
        model.vec.fill_(1.0)
    for step, v in ((3, 0.6), (4, 0.7)):
        assert not stop.update(v, step)
        assert not stop.should_stop
    assert not stop.update(0.55, 5)
    assert stop.should_stop and stop.best_step == 2 and stop.best_value == 0.5
    stop.restore(model)
    assert torch.all(model.vec == 9.0)
    up = EarlyStopping(patience=1, mode="max")
    assert up.update(0.1, 1) and up.update(0.2, 2) and not up.update(0.15, 3)
    assert up.should_stop


def test_split_by_user_is_disjoint_and_covers_everyone() -> None:
    splits = split_by_user(1000, SplitConfig())
    sizes = splits.sizes()
    assert sum(sizes.values()) == 1000
    assert sizes == {"train": 700, "val": 100, "calib": 100, "test": 100}
    users = [set(splits.users(n).tolist()) for n in SPLIT_NAMES]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not users[i] & users[j]
    assert np.array_equal(split_by_user(1000).assignment, splits.assignment)
    with pytest.raises(ValueError):
        SplitConfig(train=0.9)


def test_register_values_equal_config_defaults() -> None:
    reg = register_as_dict()
    assert reg["utility.alpha"] == "0.5" and reg["utility.delta"] == "25"
    assert (
        reg["user_benefit.horizon_years"] == "2" and reg["user_benefit.hold_years_mortgage"] == "7"
    )
    assert reg["user_benefit.hard_pull_cost"] == "15"
    assert reg["downsample.rate"] == "0.25" and reg["pending.policy"] == "drop"
    assert reg["pending.w_floor"] == "0.05"
    assert reg["calibration.method"] == "isotonic"
    assert reg["calibration.min_positives_isotonic"] == "500"
    assert reg["models.num_candidates"] == "100" and reg["models.slate_size"] == "10"
    assert reg["models.d_model"] == "64" and reg["models.hstu_max_len"] == "64"
    assert reg["models.rq_codebook_size"] == "32" and reg["models.rq_levels"] == "3"
    assert reg["models.prm_cannibalization_weight"] == "0.5"
    assert reg["generator.snapshot_at_days"] == "365" and reg["generator.num_products"] == "2000"
    assert reg["delay.instant_delay_days"] == "0.01"
    assert reg["ranker_optimizer.batch_size"] == "64" and reg["tiger_optimizer.batch_size"] == "256"
    assert reg["prm_optimizer.lr"] == "0.0005" and reg["rqvae_optimizer.weight_decay"] == "0"
    assert reg["funnel_loss.lambda_amount"] == "0.1" and reg["serving.batch_size"] == "1"
    # the register is generated, so it tracks the dataclasses exactly
    assert reg["utility.alpha"] == f"{UtilityConfig().alpha:g}"
    assert (
        reg["calibration.min_positives_isotonic"]
        == f"{CalibrationConfig().min_positives_isotonic:g}"
    )
    assert reg["models.tiger_max_history"] == str(ModelConfig().tiger_max_history)
    assert reg["ranker_optimizer.warmup_steps"] == str(RANKER_OPTIMIZER.warmup_steps)
    assert reg["prm_optimizer.total_steps"] == str(PRM_OPTIMIZER.total_steps)
    names = [r.name for r in decision_register()]
    assert len(names) == len(set(names))


def test_train_all_smoke_and_round_trip(tmp_path: Path) -> None:
    ds = SyntheticFintechDataGenerator(
        GeneratorConfig(num_users=60, num_products=80, seed=9, slates_per_user=2, slate_size=12)
    ).generate()
    cfg = TrainingConfig.tiny(steps=6)
    result = train_all(ds, cfg)
    for name in ("rqvae", "tiger", "ranker", "prm"):
        h = result.histories[name]
        assert h.steps and all(math.isfinite(v) for v in h.train_loss)
        assert h.evaluations and h.best_step >= 1
    assert "click" in result.artifacts.calibrators.calibrators
    assert result.histories["ranker"].extra["positives_per_batch"]["clicks"] > 0
    assert result.histories["tiger"].extra["selection"] == "val_recall@12"
    result.artifacts.save(tmp_path)
    loaded = PipelineArtifacts.load(tmp_path, ds, cfg)
    pipe = RecommendationPipeline(loaded)
    res = pipe.run(int(result.splits.users("test")[0]))
    assert res.size <= cfg.models.slate_size
    loaded.engine.assert_all_eligible(ds.users[res.user_index], res.item_ids)
