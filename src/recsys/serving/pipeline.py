"""End-to-end serving pipeline (§5 Orchestration).

``RecommendationPipeline.run(user_index)``::

    eligibility (gates + held + pending + family policy)      -> allowed (N+1,)
    TIGER beam over the trie with the allowed mask            -> top-100 candidates
    post-retrieval vectorized gate                            -> re-asserted candidates
    HSTU + PLE, one B x (L + K) pass, +log r on the click logit
    calibrate (isotonic / Platt per tower)                    -> p̂1, p̂2, p̂3
    EV / E[amount] / NB / U + suitability guardrails          -> survivors
    PRM over the top-10 by U, greedy family-cannibalization    -> slate order
    assert every served id is eligible, unique, not held / pending -> telemetry

Eligibility is enforced three times (trie logit mask, post-retrieval gate, output
assertion).  Calibrated probabilities are what gets logged; PRM only outputs an ordering.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from recsys import seed_everything
from recsys.data.collator import SequenceCollator
from recsys.data.impression_collator import ImpressionCollator
from recsys.data.schema import (
    NUM_STATES,
    NUM_TIERS,
    PRODUCT_FEATURE_DIM,
    USER_FEATURE_DIM,
    InteractionRecord,
    SyntheticDataset,
    UserProfile,
)
from recsys.layers.hstu import HSTUConfig
from recsys.layers.prefix_trie import SemanticIdTrie
from recsys.layers.rq_vae import RQVAE, RQVAEConfig
from recsys.models.ple.model import PLEConfig
from recsys.models.prm.model import PRM, PRMConfig, build_prm_features, rerank
from recsys.models.ranker import HSTUPLERanker
from recsys.models.tiger.model import TIGER, SemanticIdTokenizer, TIGERConfig
from recsys.serving.calibration import CalibratorSet
from recsys.serving.eligibility_engine import (
    ComplianceViolation,
    EligibilityEngine,
    UserGateArrays,
    policy_vector,
)
from recsys.serving.stages import ValuationStage, ValuedCandidates
from recsys.training.config import TrainingConfig, config_to_dict
from recsys.training.splits import records_by_user
from recsys.valuation.user_benefit import ProductEconomics, UserFinancialState

STAGES: tuple[str, ...] = (
    "eligibility",
    "retrieval",
    "post_filter",
    "ranking",
    "calibration",
    "valuation",
    "rerank",
    "assert",
)


@dataclass
class StageTiming:
    name: str
    ms: float


@dataclass
class PipelineTelemetry:
    stages: list[StageTiming] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return float(sum(s.ms for s in self.stages))

    def as_dict(self) -> dict[str, float]:
        out = {s.name: s.ms for s in self.stages}
        out["total"] = self.total_ms
        return out


@dataclass
class SlateResult:
    user_index: int
    item_ids: list[int]  # served order
    p_click: list[float]
    p_apply: list[float]
    p_approve: list[float]
    p_funded: list[float]
    expected_value: list[float]
    expected_amount: list[float]
    net_user_benefit: list[float]
    utility: list[float]
    num_eligible: int
    num_retrieved: int
    num_after_post_filter: int
    num_after_guardrails: int
    telemetry: PipelineTelemetry

    @property
    def size(self) -> int:
        return len(self.item_ids)


# ------------------------------------------------------------------ artifacts


def standardize_catalog(ds: SyntheticDataset) -> torch.Tensor:
    """``(N+1, F_p)`` standardized catalog features for the RQ-VAE (row 0 zero)."""
    x = (ds.catalog_features - ds.catalog_mean) / ds.catalog_std
    x[0] = 0.0
    return torch.as_tensor(x, dtype=torch.float32)


@dataclass
class PipelineArtifacts:
    """Everything the pipeline needs; models may be untrained (tests) or loaded."""

    config: TrainingConfig
    products: list[Any]
    users: list[UserProfile]
    records: dict[int, InteractionRecord]
    dataset: SyntheticDataset
    engine: EligibilityEngine
    rqvae: RQVAE
    item_codes: torch.Tensor  # (N+1, T)
    level_sizes: tuple[int, ...]
    trie: SemanticIdTrie
    tiger: TIGER
    tokenizer: SemanticIdTokenizer
    ranker: HSTUPLERanker
    calibrators: CalibratorSet
    prm: PRM
    seq_collator: SequenceCollator  # for the ranker (L = hstu_max_len)
    tiger_collator: SequenceCollator  # for TIGER (L = tiger_max_history)
    imp_collator: ImpressionCollator
    valuation: ValuationStage
    user_gates: UserGateArrays

    def eval(self) -> None:
        for m in (self.rqvae, self.tiger, self.ranker, self.prm):
            m.eval()

    # ---------------------------------------------------------------- save / load
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.rqvae.state_dict(), out / "rqvae.pt")
        torch.save(self.tiger.state_dict(), out / "tiger.pt")
        torch.save(self.ranker.state_dict(), out / "ranker.pt")
        torch.save(self.prm.state_dict(), out / "prm.pt")
        np.save(out / "item_codes.npy", self.item_codes.cpu().numpy())
        self.calibrators.save(out / "calibrators.json")
        meta = {"config": config_to_dict(self.config), "level_sizes": list(self.level_sizes)}
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, in_dir: str | Path, dataset: SyntheticDataset, config: TrainingConfig
    ) -> PipelineArtifacts:
        src = Path(in_dir)
        codes = torch.as_tensor(np.load(src / "item_codes.npy"), dtype=torch.int64)
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        art = build_artifacts(
            dataset, config, item_codes=codes, level_sizes=tuple(meta["level_sizes"])
        )
        art.rqvae.load_state_dict(torch.load(src / "rqvae.pt", weights_only=True))
        art.tiger.load_state_dict(torch.load(src / "tiger.pt", weights_only=True))
        art.ranker.load_state_dict(torch.load(src / "ranker.pt", weights_only=True))
        art.prm.load_state_dict(torch.load(src / "prm.pt", weights_only=True))
        art.calibrators = CalibratorSet.load(src / "calibrators.json")
        art.eval()
        return art


def make_model_configs(
    ds: SyntheticDataset, config: TrainingConfig, level_sizes: tuple[int, ...]
) -> tuple[RQVAEConfig, TIGERConfig, HSTUConfig, PLEConfig, PRMConfig]:
    m = config.models
    n = ds.num_items
    rq = RQVAEConfig(
        input_dim=PRODUCT_FEATURE_DIM, latent_dim=m.rq_latent_dim, num_levels=m.rq_levels,
        codebook_size=m.rq_codebook_size, encoder_hidden=(m.rq_hidden,),
    )  # fmt: skip
    tiger = TIGERConfig(
        num_items=n, level_sizes=level_sizes, d_model=m.d_model, n_heads=m.tiger_heads,
        n_layers=m.tiger_layers, d_ff=m.tiger_d_ff, max_history_items=m.tiger_max_history,
        beam_size=m.num_candidates,
    )  # fmt: skip
    hstu = HSTUConfig(
        num_items=n, d_model=m.d_model, n_heads=m.hstu_heads, n_layers=m.hstu_layers,
        max_len=m.hstu_max_len, max_candidates=m.num_candidates,
        num_time_buckets=m.num_time_buckets, tabular_dim=USER_FEATURE_DIM + PRODUCT_FEATURE_DIM,
    )  # fmt: skip
    task_names: tuple[str, ...] = ("click", "apply", "approve", "amount")
    dims: tuple[int, ...] = (1, 1, 1, 3)
    if config.funnel_loss.ssb_mode == "dr":
        task_names, dims = (*task_names, "imputation"), (*dims, 1)
    ple = PLEConfig(
        input_dim=hstu.fusion_dim, task_names=task_names, task_output_dims=dims,
        num_shared_experts=m.ple_shared_experts, num_task_experts=m.ple_task_experts,
        expert_hidden=(m.ple_expert_hidden,), expert_dim=m.ple_expert_dim,
        num_levels=m.ple_levels, tower_hidden=(m.ple_tower_hidden,),
    )  # fmt: skip
    prm = PRMConfig(
        cand_dim=m.d_model, user_dim=m.d_model, d_model=m.prm_d_model, n_heads=m.prm_heads,
        n_layers=m.prm_layers, d_ff=m.prm_d_ff, slate_size=m.slate_size,
        cannibalization_weight=m.prm_cannibalization_weight, prm_target=m.prm_target,
        utility_temperature=m.prm_utility_temperature,
    )  # fmt: skip
    return rq, tiger, hstu, ple, prm


def build_artifacts(
    ds: SyntheticDataset,
    config: TrainingConfig | None = None,
    seed: int = 0,
    item_codes: torch.Tensor | None = None,
    level_sizes: tuple[int, ...] | None = None,
) -> PipelineArtifacts:
    """Untrained artifacts (random weights).  Semantic IDs come from a k-means-initialized
    RQ-VAE unless ``item_codes`` are given (loading)."""
    cfg = config or TrainingConfig()
    seed_everything(seed)
    m = cfg.models
    rq_cfg = RQVAEConfig(
        input_dim=PRODUCT_FEATURE_DIM, latent_dim=m.rq_latent_dim, num_levels=m.rq_levels,
        codebook_size=m.rq_codebook_size, encoder_hidden=(m.rq_hidden,),
    )  # fmt: skip
    rqvae = RQVAE(rq_cfg)
    x = standardize_catalog(ds)
    if item_codes is None:
        rqvae.train()
        rqvae.init_codebooks_kmeans(x[1:])
        rqvae.eval()
        item_codes = assign_item_codes(rqvae, x)
    if level_sizes is None:
        level_sizes = (*([m.rq_codebook_size] * m.rq_levels), int(item_codes[1:, -1].max()) + 1)
    _, tiger_cfg, hstu_cfg, ple_cfg, prm_cfg = make_model_configs(ds, cfg, level_sizes)
    trie = SemanticIdTrie.build(item_codes[1:], np.arange(1, ds.num_items + 1), level_sizes)
    tokenizer = SemanticIdTokenizer(item_codes, tiger_cfg)
    engine = EligibilityEngine.from_products(ds.products)
    user_gates = UserGateArrays.from_users(ds.users, ds.num_items)
    econ = ProductEconomics.from_products(ds.products)
    state = UserFinancialState.from_users(ds.users)
    imp_collator = ImpressionCollator(
        ds.catalog_features, ds.user_features, ds.family_by_item(),
        snapshot_at_days=ds.snapshot_at_days, delay_config=DelayFromConfig.get(),
        pending_policy=cfg.pending.policy, w_floor=cfg.pending.w_floor,
        user_pending_families=user_gates.pending_families, payout_by_item=econ.partner_payout,
        slate_size=None,
    )  # fmt: skip
    return PipelineArtifacts(
        config=cfg,
        products=list(ds.products),
        users=list(ds.users),
        records=records_by_user(ds.interactions),
        dataset=ds,
        engine=engine,
        rqvae=rqvae,
        item_codes=item_codes,
        level_sizes=level_sizes,
        trie=trie,
        tiger=TIGER(tiger_cfg),
        tokenizer=tokenizer,
        ranker=HSTUPLERanker(hstu_cfg, ple_cfg, cfg.downsample.rate),
        calibrators=CalibratorSet(),
        prm=PRM(prm_cfg),
        seq_collator=SequenceCollator(max_len=m.hstu_max_len),
        tiger_collator=SequenceCollator(max_len=m.tiger_max_history),
        imp_collator=imp_collator,
        valuation=ValuationStage(
            econ, state, policy_vector(engine.policy, "penalize"), cfg.user_benefit, cfg.utility
        ),
        user_gates=user_gates,
    )


class DelayFromConfig:
    """Indirection so the pipeline never hard-codes the delay law (the generator owns it)."""

    @staticmethod
    def get() -> Any:
        from recsys.data.schema import DelayConfig

        return DelayConfig()


@torch.no_grad()
def assign_item_codes(rqvae: RQVAE, x: torch.Tensor) -> torch.Tensor:
    """``(N+1, T)`` semantic ids with the dedup level; row 0 (PAD) is all zeros."""
    codes = rqvae.assign_semantic_ids(x[1:])
    out = torch.zeros((x.shape[0], codes.shape[1]), dtype=torch.int64)
    out[1:] = codes
    return out


# ------------------------------------------------------------------- pipeline


class RecommendationPipeline:
    def __init__(self, artifacts: PipelineArtifacts) -> None:
        self.a = artifacts
        self.cfg = artifacts.config
        artifacts.eval()

    @contextmanager
    def _timed(self, telemetry: PipelineTelemetry, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            telemetry.stages.append(StageTiming(name, (time.perf_counter() - t0) * 1000.0))

    # ---------------------------------------------------------------- stages
    def retrieve(self, user: UserProfile, allowed: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
        a = self.a
        rec = a.records[user.user_index]
        seq = a.tiger_collator([rec])
        ht, ha, hm = a.tokenizer.encode_history(
            seq.item_ids, seq.action_ids, seq.attention_mask,
            torch.tensor([user.tier_index]), torch.tensor([user.state_index]),
        )  # fmt: skip
        gen = a.tiger.generate(
            ht, ha, hm, a.trie, a.tokenizer, torch.as_tensor(allowed).unsqueeze(0),
            beam_size=self.cfg.models.num_candidates,
        )  # fmt: skip
        ids = gen.item_ids[0].numpy()
        out: npt.NDArray[np.int64] = ids[ids > 0].astype(np.int64)
        return out

    def score(self, user: UserProfile, candidate_ids: npt.NDArray[np.int64]) -> ValuedCandidates:
        a = self.a
        batch = a.imp_collator.collate_candidates(
            np.array([user.user_index]), candidate_ids[None, :]
        )
        seq = a.seq_collator([a.records[user.user_index]])
        out = a.ranker.predict(batch.with_sequence(seq))
        return a.valuation.value(batch, out, a.calibrators)

    @torch.no_grad()
    def run(self, user_index: int) -> SlateResult:
        a = self.a
        user = a.users[user_index]
        tel = PipelineTelemetry()
        with self._timed(tel, "eligibility"):
            allowed = a.engine.mask_for_user(user)
            num_eligible = int(allowed.sum())
        with self._timed(tel, "retrieval"):
            cand = self.retrieve(user, allowed) if num_eligible > 0 else np.zeros(0, np.int64)
        with self._timed(tel, "post_filter"):
            kept = a.engine.filter_candidates(user, cand)
        if kept.size == 0:
            for name in STAGES[3:]:
                tel.stages.append(StageTiming(name, 0.0))
            return SlateResult(
                user_index,
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                num_eligible,
                int(cand.size),
                0,
                0,
                tel,
            )
        with self._timed(tel, "ranking"):
            batch = a.imp_collator.collate_candidates(np.array([user.user_index]), kept[None, :])
            seq = a.seq_collator([a.records[user.user_index]])
            raw = a.ranker.predict(batch.with_sequence(seq))
        with self._timed(tel, "calibration"):
            z = (t.detach().numpy().reshape(-1) for t in (raw.z1, raw.z2, raw.z3))
            a.calibrators.calibrate(*z)  # measured separately; recomputed inside valuation
        with self._timed(tel, "valuation"):
            val = a.valuation.value(batch, raw, a.calibrators)
        with self._timed(tel, "rerank"):
            order = self._rerank(val, batch.family_ids, raw)
        with self._timed(tel, "assert"):
            items = [int(kept[j]) for j in order]
            if len(set(items)) != len(items):
                raise ComplianceViolation(f"duplicate items in slate for user {user_index}")
            a.engine.assert_all_eligible(user, items)
            bad = [i for i in items if i in user.held_product_ids or i in user.pending_product_ids]
            if bad:
                raise ComplianceViolation(f"held / pending products re-recommended: {bad}")
        sel = np.array(order, dtype=np.int64)
        return SlateResult(
            user_index=user_index,
            item_ids=items,
            p_click=val.p1[0, sel].tolist(),
            p_apply=val.p2[0, sel].tolist(),
            p_approve=val.p3[0, sel].tolist(),
            p_funded=val.p_funded[0, sel].tolist(),
            expected_value=val.ev[0, sel].tolist(),
            expected_amount=val.expected_amount[0, sel].tolist(),
            net_user_benefit=val.nb[0, sel].tolist(),
            utility=val.utility[0, sel].tolist(),
            num_eligible=num_eligible,
            num_retrieved=int(cand.size),
            num_after_post_filter=int(kept.size),
            num_after_guardrails=int(val.keep.sum()),
            telemetry=tel,
        )

    def _rerank(self, val: ValuedCandidates, family_ids: torch.Tensor, raw: Any) -> list[int]:
        """Top-``slate_size`` survivors by utility -> PRM scores -> greedy family rerank."""
        a, m = self.a, self.cfg.models
        u = val.utility[0]
        survivors = np.flatnonzero(val.keep[0])
        if survivors.size == 0:
            return []
        top = survivors[np.argsort(-u[survivors], kind="stable")][: m.slate_size]
        idx = torch.as_tensor(top, dtype=torch.int64)

        def t(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(arr[:, top], dtype=torch.float32)

        feats = build_prm_features(
            raw.h_cand[:, idx], t(val.p1), t(val.p2), t(val.p3), t(val.ev), t(val.nb),
            torch.as_tensor(u[top][None, :], dtype=torch.float32), family_ids[:, idx],
        )  # fmt: skip
        mask = torch.ones((1, top.size), dtype=torch.bool)
        scores = a.prm(feats, raw.h_user, mask).scores
        order = rerank(scores, family_ids[:, idx], mask, m.prm_cannibalization_weight)[0]
        return [int(top[j]) for j in order.tolist()]

    def run_many(self, user_indices: Sequence[int]) -> list[SlateResult]:
        return [self.run(int(u)) for u in user_indices]


def latency_summary(results: Sequence[SlateResult]) -> dict[str, dict[str, float]]:
    """Per-stage p50 / p99 / max in ms across results (plus total)."""
    out: dict[str, dict[str, float]] = {}
    names = [*STAGES, "total"]
    for name in names:
        vals = np.array([r.telemetry.as_dict().get(name, 0.0) for r in results])
        if vals.size == 0:
            continue
        out[name] = {
            "p50": float(np.percentile(vals, 50)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(vals.max()),
        }
    return out


__all__ = [
    "NUM_STATES",
    "NUM_TIERS",
    "STAGES",
    "PipelineArtifacts",
    "PipelineTelemetry",
    "RecommendationPipeline",
    "SlateResult",
    "StageTiming",
    "assign_item_codes",
    "build_artifacts",
    "latency_summary",
    "make_model_configs",
    "standardize_catalog",
]
