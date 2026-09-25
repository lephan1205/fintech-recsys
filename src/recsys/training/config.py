"""Training configuration (D8) and the decision register.

One :class:`OptimizerConfig` per model, consumed by ``training/optim.py``; the model
sizes of §2 / D6; the user split; negative down-sampling and pending policy.  Every
default here is a row of the decision register (:func:`decision_register`) that the
write-up's Appendix A must reproduce exactly (``tests/test_docs.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Literal

from recsys.data.delayed_feedback import PendingPolicy
from recsys.data.schema import DelayConfig
from recsys.data.synthetic_generator import GeneratorConfig
from recsys.losses.funnel_loss import FunnelLossConfig
from recsys.serving.calibration import CalibrationConfig
from recsys.valuation.user_benefit import UserBenefitConfig
from recsys.valuation.utility import UtilityConfig

OptimizerName = Literal["adam", "adamw"]


@dataclass(frozen=True)
class OptimizerConfig:
    name: OptimizerName
    lr: float
    weight_decay: float
    warmup_steps: int
    total_steps: int
    final_lr_fraction: float  # cosine floor as a fraction of lr (1.0 = constant)
    grad_clip: float | None
    batch_size: int  # 0 = full batch
    eval_every: int
    patience: int = 3
    betas: tuple[float, float] = (0.9, 0.98)

    def __post_init__(self) -> None:
        if self.total_steps < 1 or self.eval_every < 1:
            raise ValueError("total_steps and eval_every must be >= 1")
        if not 0.0 < self.final_lr_fraction <= 1.0:
            raise ValueError("final_lr_fraction must be in (0, 1]")
        if self.warmup_steps < 0 or self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps]")


#: RQ-VAE: plain Adam (= AdamW with wd 0), constant lr, full-catalog batch (D8).
RQVAE_OPTIMIZER = OptimizerConfig(
    name="adam", lr=1e-3, weight_decay=0.0, warmup_steps=0, total_steps=2000,
    final_lr_fraction=1.0, grad_clip=None, batch_size=0, eval_every=100,
)  # fmt: skip
#: TIGER: AdamW, warmup 500 -> cosine to 10 %, clip 1.0, 256 sequences.
TIGER_OPTIMIZER = OptimizerConfig(
    name="adamw", lr=1e-3, weight_decay=0.01, warmup_steps=500, total_steps=3000,
    final_lr_fraction=0.1, grad_clip=1.0, batch_size=256, eval_every=200,
)  # fmt: skip
#: HSTU + PLE: AdamW, warmup 500 -> cosine to 10 %, clip 1.0, 64 slates x K = 100 rows.
RANKER_OPTIMIZER = OptimizerConfig(
    name="adamw", lr=1e-3, weight_decay=0.01, warmup_steps=500, total_steps=2000,
    final_lr_fraction=0.1, grad_clip=1.0, batch_size=64, eval_every=100,
)  # fmt: skip
#: PRM: AdamW, warmup 200 -> cosine to 10 %, clip 1.0, 128 slates.
PRM_OPTIMIZER = OptimizerConfig(
    name="adamw", lr=5e-4, weight_decay=0.01, warmup_steps=200, total_steps=600,
    final_lr_fraction=0.1, grad_clip=1.0, batch_size=128, eval_every=50,
)  # fmt: skip


@dataclass(frozen=True)
class SplitConfig:
    """User-level split; ``val`` drives early stopping, ``calib`` fits calibrators only."""

    train: float = 0.70
    val: float = 0.10
    calib: float = 0.10
    test: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        if abs(self.train + self.val + self.calib + self.test - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")


@dataclass(frozen=True)
class DownsampleConfig:
    rate: float = 0.25  # r: keep probability for non-clicked impressions
    seed: int = 0


@dataclass(frozen=True)
class PendingConfig:
    policy: PendingPolicy = "drop"
    w_floor: float = 0.05


@dataclass(frozen=True)
class ModelConfig:
    """Architecture sizes (D6 / §2).  ``tiny()`` shrinks everything for tests."""

    d_model: int = 64
    hstu_layers: int = 2
    hstu_heads: int = 2
    hstu_max_len: int = 64  # L
    num_candidates: int = 100  # K = beam width
    slate_size: int = 10
    num_time_buckets: int = 32
    rq_levels: int = 3
    rq_codebook_size: int = 32
    rq_latent_dim: int = 16
    rq_hidden: int = 32
    tiger_max_history: int = 20
    tiger_layers: int = 2
    tiger_heads: int = 2
    tiger_d_ff: int = 128
    ple_levels: int = 2
    ple_shared_experts: int = 2
    ple_task_experts: int = 1
    ple_expert_hidden: int = 64
    ple_expert_dim: int = 32
    ple_tower_hidden: int = 32
    prm_d_model: int = 32
    prm_heads: int = 2
    prm_layers: int = 1
    prm_d_ff: int = 64
    prm_cannibalization_weight: float = 0.5  # beta
    prm_target: Literal["click", "utility"] = "click"
    prm_utility_temperature: float = 25.0

    @classmethod
    def tiny(cls) -> ModelConfig:
        return cls(
            d_model=16, hstu_layers=1, hstu_heads=2, hstu_max_len=8, num_candidates=12,
            slate_size=5, num_time_buckets=8, rq_levels=3, rq_codebook_size=8,
            rq_latent_dim=8, rq_hidden=16, tiger_max_history=6, tiger_layers=1,
            tiger_heads=2, tiger_d_ff=32, ple_levels=1, ple_expert_hidden=16,
            ple_expert_dim=8, ple_tower_hidden=8, prm_d_model=16, prm_d_ff=32,
        )  # fmt: skip


@dataclass(frozen=True)
class TrainingConfig:
    """Everything ``scripts/train_all.py`` needs; all fields are register rows."""

    models: ModelConfig = ModelConfig()
    split: SplitConfig = SplitConfig()
    downsample: DownsampleConfig = DownsampleConfig()
    pending: PendingConfig = PendingConfig()
    funnel_loss: FunnelLossConfig = FunnelLossConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    user_benefit: UserBenefitConfig = UserBenefitConfig()
    utility: UtilityConfig = UtilityConfig()
    rqvae_optimizer: OptimizerConfig = RQVAE_OPTIMIZER
    tiger_optimizer: OptimizerConfig = TIGER_OPTIMIZER
    ranker_optimizer: OptimizerConfig = RANKER_OPTIMIZER
    prm_optimizer: OptimizerConfig = PRM_OPTIMIZER
    rqvae_min_utilization: float = 0.9
    seed: int = 0

    @classmethod
    def tiny(cls, steps: int = 6) -> TrainingConfig:
        def short(opt: OptimizerConfig, batch: int) -> OptimizerConfig:
            return OptimizerConfig(
                opt.name, opt.lr, opt.weight_decay, min(2, steps), steps, opt.final_lr_fraction,
                opt.grad_clip, batch, max(1, steps // 2), patience=2,
            )  # fmt: skip

        return cls(
            models=ModelConfig.tiny(),
            calibration=CalibrationConfig(min_positives_isotonic=5.0),
            rqvae_optimizer=short(RQVAE_OPTIMIZER, 0),
            tiger_optimizer=short(TIGER_OPTIMIZER, 8),
            ranker_optimizer=short(RANKER_OPTIMIZER, 4),
            prm_optimizer=short(PRM_OPTIMIZER, 4),
            rqvae_min_utilization=0.0,
        )


# ------------------------------------------------------------------- register


@dataclass(frozen=True)
class RegisterRow:
    name: str
    value: str
    location: str


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, tuple):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return str(v)


def _rows(prefix: str, obj: Any, location: str, skip: tuple[str, ...] = ()) -> list[RegisterRow]:
    out: list[RegisterRow] = []
    if not is_dataclass(obj):
        raise TypeError("register rows need a dataclass")
    for f in fields(obj):
        if f.name in skip:
            continue
        v = getattr(obj, f.name)
        if is_dataclass(v) and not isinstance(v, type):
            continue  # nested configs are listed on their own
        out.append(RegisterRow(f"{prefix}.{f.name}", _fmt(v), location))
    return out


def decision_register(config: TrainingConfig | None = None) -> list[RegisterRow]:
    """Every numeric / categorical default, generated from the config dataclasses."""
    cfg = config or TrainingConfig()
    gen = GeneratorConfig(num_products=2000, num_users=3000, slates_per_user=4,
                          slate_size=100, ineligible_per_slate=10)  # fmt: skip
    rows: list[RegisterRow] = []
    rows += _rows(
        "generator", gen, "scripts/generate_dataset.py -> GeneratorConfig", skip=("delay",)
    )
    rows += _rows("delay", DelayConfig(), "data/schema.py::DelayConfig")
    rows += _rows("models", cfg.models, "training/config.py::ModelConfig")
    rows += _rows("split", cfg.split, "training/config.py::SplitConfig")
    rows += _rows("downsample", cfg.downsample, "training/config.py::DownsampleConfig")
    rows += _rows("pending", cfg.pending, "training/config.py::PendingConfig")
    rows += _rows("funnel_loss", cfg.funnel_loss, "losses/funnel_loss.py::FunnelLossConfig")
    rows += _rows("calibration", cfg.calibration, "serving/calibration.py::CalibrationConfig")
    rows += _rows("user_benefit", cfg.user_benefit, "valuation/user_benefit.py::UserBenefitConfig")
    rows += _rows("utility", cfg.utility, "valuation/utility.py::UtilityConfig")
    for name, opt in (
        ("rqvae_optimizer", cfg.rqvae_optimizer),
        ("tiger_optimizer", cfg.tiger_optimizer),
        ("ranker_optimizer", cfg.ranker_optimizer),
        ("prm_optimizer", cfg.prm_optimizer),
    ):
        rows += _rows(name, opt, f"training/config.py::{name.upper()}")
    rows.append(
        RegisterRow(
            "rqvae_min_utilization",
            _fmt(cfg.rqvae_min_utilization),
            "training/config.py::TrainingConfig",
        )
    )
    rows.append(
        RegisterRow("serving.batch_size", "1", "serving/pipeline.py::RecommendationPipeline.run")
    )
    return rows


def register_as_dict(config: TrainingConfig | None = None) -> dict[str, str]:
    return {r.name: r.value for r in decision_register(config)}


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """JSON-serializable view of a (nested) frozen config dataclass."""
    return asdict(cfg)
