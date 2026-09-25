"""Training loops (D8) — plain loops over the configs in ``training/config.py``.

``scripts/train_all.py`` uses only these.  Every loop: ``build_optimizer`` (parameter
groups), ``WarmupCosine`` schedule, gradient clipping, an evaluation every
``eval_every`` steps with the model's own criterion, early stopping with patience,
and restoration of the best checkpoint.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from recsys.data.collator import SequenceCollator
from recsys.data.impression_collator import ImpressionBatch, ImpressionCollator
from recsys.data.schema import (
    POSITIVE_ACTIONS,
    ImpressionSlate,
    InteractionRecord,
    SyntheticDataset,
    UserProfile,
)
from recsys.losses.funnel_loss import UnifiedFunnelLoss
from recsys.losses.listwise_loss import prm_listwise_loss
from recsys.metrics.ranking_metrics import recall_at_k
from recsys.models.prm.model import build_prm_features
from recsys.serving.calibration import CalibratorSet, fit_funnel_calibrators
from recsys.serving.pipeline import (
    PipelineArtifacts,
    assign_item_codes,
    build_artifacts,
    standardize_catalog,
)
from recsys.training.config import TrainingConfig
from recsys.training.downsampling import NegativeDownsampler
from recsys.training.early_stopping import EarlyStopping
from recsys.training.optim import WarmupCosine, build_optimizer, clip_gradients
from recsys.training.splits import Splits, split_by_user


@dataclass
class TrainHistory:
    name: str
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    evaluations: list[dict[str, float]] = field(default_factory=list)
    best_step: int = -1
    best_value: float = float("nan")
    stopped_early: bool = False
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "train_loss": self.train_loss,
            "evaluations": self.evaluations,
            "best_step": self.best_step,
            "best_value": self.best_value,
            "stopped_early": self.stopped_early,
            "seconds": self.seconds,
            "extra": self.extra,
        }


def minibatches(
    n: int, batch_size: int, steps: int, rng: np.random.Generator
) -> Iterator[npt.NDArray[np.int64]]:
    """Yield ``steps`` index arrays, reshuffling every epoch (``batch_size = 0`` = full)."""
    if n == 0:
        raise ValueError("no training examples")
    size = n if batch_size <= 0 else min(batch_size, n)
    perm = rng.permutation(n)
    pos = 0
    for _ in range(steps):
        if pos + size > n:
            perm = rng.permutation(n)
            pos = 0
        yield perm[pos : pos + size]
        pos += size


def _log(log: Any, msg: str) -> None:
    if log is not None:
        log(msg)


# ------------------------------------------------------------------- RQ-VAE


def train_rqvae(art: PipelineArtifacts, cfg: TrainingConfig, log: Any = None) -> TrainHistory:
    """Full-catalog Adam; criterion: reconstruction MSE subject to utilization >= threshold."""
    opt_cfg = cfg.rqvae_optimizer
    x = standardize_catalog(art.dataset)[1:]
    model = art.rqvae
    model.train()
    if not bool(model.initialized):
        model.init_codebooks_kmeans(x)
    opt = build_optimizer(model, opt_cfg)
    sched = WarmupCosine(opt, opt_cfg)
    stop = EarlyStopping(patience=opt_cfg.patience, mode="min")
    hist = TrainHistory("rqvae")
    t0 = time.perf_counter()
    for step in range(1, opt_cfg.total_steps + 1):
        out = model(x)
        opt.zero_grad()
        out.loss_total.backward()
        clip_gradients(model, opt_cfg)
        opt.step()
        sched.step()
        hist.steps.append(step)
        hist.train_loss.append(float(out.loss_total.detach()))
        if step % opt_cfg.eval_every == 0 or step == opt_cfg.total_steps:
            model.eval()
            with torch.no_grad():
                recon = float(torch.nn.functional.mse_loss(model(x).recon, x))
                util = model.codebook_utilization(x)
            model.train()
            ok = bool(float(util.min()) >= cfg.rqvae_min_utilization)
            ev = {
                "step": step,
                "recon_mse": recon,
                "min_utilization": float(util.min()),
                "feasible": float(ok),
            }
            ev.update({f"utilization_{i}": float(u) for i, u in enumerate(util.tolist())})
            hist.evaluations.append(ev)
            # infeasible checkpoints (utilization below threshold) never count as improvements
            value = recon if ok else float("inf")
            improved = stop.update(value, step, model)
            _log(
                log,
                f"[rqvae] step {step} recon {recon:.4f} util {util.tolist()}"
                f"{' *' if improved else ''}",
            )
            if stop.should_stop:
                hist.stopped_early = True
                break
    if stop.best_state is None or stop.best_value == float("inf"):
        # never feasible: keep the last weights and record it
        hist.extra["utilization_constraint_met"] = False
    else:
        stop.restore(model)
        hist.extra["utilization_constraint_met"] = True
    hist.best_step, hist.best_value = stop.best_step, float(stop.best_value or float("nan"))
    model.eval()
    hist.seconds = time.perf_counter() - t0
    return hist


# -------------------------------------------------------------------- TIGER


@dataclass
class TigerBatch:
    tokens: torch.Tensor
    actions: torch.Tensor
    mask: torch.Tensor
    target_codes: torch.Tensor
    user_indices: torch.Tensor


class TigerBatcher:
    """Records with a positive target -> token batches (context tokens included)."""

    def __init__(
        self,
        records: Sequence[InteractionRecord],
        users: Sequence[UserProfile],
        art: PipelineArtifacts,
        positive_only: bool = True,
    ) -> None:
        self.records = [
            r for r in records if not positive_only or r.target.action in POSITIVE_ACTIONS
        ]
        self.users = users
        self.art = art

    def __len__(self) -> int:
        return len(self.records)

    def batch(self, idx: npt.NDArray[np.int64]) -> TigerBatch:
        recs = [self.records[int(i)] for i in idx]
        seq = self.art.tiger_collator(recs)
        tier = torch.tensor([self.users[r.user_index].tier_index for r in recs])
        state = torch.tensor([self.users[r.user_index].state_index for r in recs])
        ht, ha, hm = self.art.tokenizer.encode_history(
            seq.item_ids, seq.action_ids, seq.attention_mask, tier, state
        )
        return TigerBatch(ht, ha, hm, self.art.item_codes[seq.target_item_ids], seq.user_indices)


@torch.no_grad()
def tiger_val_loss(art: PipelineArtifacts, batcher: TigerBatcher, batch_size: int = 256) -> float:
    if len(batcher) == 0:
        return float("nan")
    art.tiger.eval()
    total, n = 0.0, 0
    for start in range(0, len(batcher), batch_size):
        idx = np.arange(start, min(start + batch_size, len(batcher)))
        b = batcher.batch(idx)
        loss = art.tiger.next_sid_loss(b.tokens, b.actions, b.mask, b.target_codes, art.tokenizer)
        total += float(loss.target) * len(idx)
        n += len(idx)
    return total / max(n, 1)


@dataclass
class RetrievalEval:
    recall: dict[int, float]  # k -> Recall@k
    per_user: npt.NDArray[np.float64]  # Recall@max(k) per user (nan = no eligible positive)
    user_indices: npt.NDArray[np.int64]
    beam_yield: float
    num_users: int
    num_excluded_positives: int
    candidates: npt.NDArray[np.int64]  # (U, beam)
    allowed: npt.NDArray[np.bool_]  # (U, N+1)


@torch.no_grad()
def retrieval_eval(
    art: PipelineArtifacts,
    records: Sequence[InteractionRecord],
    positives: npt.NDArray[np.int64],
    ks: Sequence[int] = (10, 50, 100),
    batch_size: int = 32,
) -> RetrievalEval:
    """Beam-search ``records`` (one per user) and score ``positives (U, P)`` (0 = pad) with
    the D7 recall: only *eligible* positives count, ineligible ones are reported."""
    art.tiger.eval()
    users = [art.users[r.user_index] for r in records]
    allowed = art.engine.mask_for_users(users)
    beam = art.config.models.num_candidates
    cands = np.full((len(records), beam), -1, dtype=np.int64)
    for start in range(0, len(records), batch_size):
        recs = records[start : start + batch_size]
        seq = art.tiger_collator(recs)
        tier = torch.tensor([art.users[r.user_index].tier_index for r in recs])
        state = torch.tensor([art.users[r.user_index].state_index for r in recs])
        ht, ha, hm = art.tokenizer.encode_history(
            seq.item_ids, seq.action_ids, seq.attention_mask, tier, state
        )
        gen = art.tiger.generate(
            ht, ha, hm, art.trie, art.tokenizer,
            torch.as_tensor(allowed[start : start + len(recs)]), beam_size=beam,
        )  # fmt: skip
        cands[start : start + len(recs)] = gen.item_ids.numpy()
    pos = np.asarray(positives, dtype=np.int64)
    valid = (pos > 0) & np.take_along_axis(allowed, np.clip(pos, 0, None), axis=1)
    out = {k: recall_at_k(cands, pos, valid, k) for k in ks}
    top = out[max(ks)]
    return RetrievalEval(
        recall={k: r.mean for k, r in out.items()},
        per_user=top.per_user,
        user_indices=np.array([r.user_index for r in records], dtype=np.int64),
        beam_yield=float((cands > 0).sum(axis=1).mean() / beam),
        num_users=top.num_users,
        num_excluded_positives=top.num_excluded_positives,
        candidates=cands,
        allowed=allowed,
    )


def next_item_positives(records: Sequence[InteractionRecord]) -> npt.NDArray[np.int64]:
    """``(U, 1)`` target item per record (0 when the target action is not positive)."""
    return np.array(
        [[r.target.item_id if r.target.action in POSITIVE_ACTIONS else 0] for r in records],
        dtype=np.int64,
    )


def train_tiger(
    art: PipelineArtifacts, splits: Splits, cfg: TrainingConfig, log: Any = None
) -> TrainHistory:
    """Next-SID CE; early stopping on val loss; checkpoint selected by val Recall@100."""
    opt_cfg = cfg.tiger_optimizer
    ds = art.dataset
    train = TigerBatcher(splits.records(ds, "train"), art.users, art)
    val_records = [r for r in splits.records(ds, "val") if r.target.action in POSITIVE_ACTIONS]
    val = TigerBatcher(val_records, art.users, art)
    model = art.tiger
    model.train()
    opt = build_optimizer(model, opt_cfg)
    sched = WarmupCosine(opt, opt_cfg)
    stop = EarlyStopping(patience=opt_cfg.patience, mode="min")
    best_recall = EarlyStopping(patience=10**9, mode="max")
    hist = TrainHistory("tiger")
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()
    k_sel = cfg.models.num_candidates
    for step, idx in enumerate(
        minibatches(len(train), opt_cfg.batch_size, opt_cfg.total_steps, rng), 1
    ):
        b = train.batch(idx)
        loss = model.next_sid_loss(b.tokens, b.actions, b.mask, b.target_codes, art.tokenizer)
        opt.zero_grad()
        loss.total.backward()
        clip_gradients(model, opt_cfg)
        opt.step()
        sched.step()
        hist.steps.append(step)
        hist.train_loss.append(float(loss.total.detach()))
        if step % opt_cfg.eval_every == 0 or step == opt_cfg.total_steps:
            model.eval()
            vloss = tiger_val_loss(art, val)
            rec = (
                retrieval_eval(art, val_records, next_item_positives(val_records), ks=(k_sel,))
                if val_records
                else None
            )
            model.train()
            recall = rec.recall[k_sel] if rec is not None else float("nan")
            ev = {"step": step, "val_next_sid_loss": vloss, f"val_recall@{k_sel}": recall}
            hist.evaluations.append(ev)
            stop.update(vloss, step)
            improved = best_recall.update(recall if recall == recall else -1.0, step, model)
            _log(
                log,
                f"[tiger] step {step} train {hist.train_loss[-1]:.4f} val {vloss:.4f} "
                f"recall@{k_sel} {recall:.4f} {'*' if improved else ''}",
            )
            if stop.should_stop:
                hist.stopped_early = True
                break
    best_recall.restore(model)
    hist.best_step = best_recall.best_step
    hist.best_value = float(
        best_recall.best_value if best_recall.best_value is not None else float("nan")
    )
    hist.extra["selection"] = f"val_recall@{k_sel}"
    model.eval()
    hist.seconds = time.perf_counter() - t0
    return hist


# ------------------------------------------------------------------- ranker


class RankerBatcher:
    """Slates -> ``ImpressionBatch`` with the user's history truncated at ``served_at_days``."""

    def __init__(self, slates: Sequence[ImpressionSlate], art: PipelineArtifacts) -> None:
        self.slates = list(slates)
        self.art = art

    def __len__(self) -> int:
        return len(self.slates)

    def batch(self, idx: npt.NDArray[np.int64]) -> ImpressionBatch:
        slates = [self.slates[int(i)] for i in idx]
        batch = self.art.imp_collator(slates)
        recs = [self.art.records[s.user_index] for s in slates]
        seq = self.art.seq_collator(recs, cutoff_days=[s.served_at_days for s in slates])
        return batch.with_sequence(seq)


def funnel_loss_on(
    model: torch.nn.Module,
    loss: UnifiedFunnelLoss,
    batch: ImpressionBatch,
    train_mask: torch.Tensor | None = None,
) -> Any:
    out = model(batch)
    return loss(
        out.z1, out.z2, out.z3, batch.y_click, batch.y_apply, batch.y_approve,
        batch.approve_weight, batch.approve_observed, batch.candidate_mask,
        train_mask=train_mask, amount_logits=out.amount_logits, amounts=batch.amounts,
        imputation_logits=out.imputation_logits,
    )  # fmt: skip


@torch.no_grad()
def ranker_val_loss(
    art: PipelineArtifacts, loss: UnifiedFunnelLoss, batcher: RankerBatcher, batch_size: int = 64
) -> dict[str, float]:
    art.ranker.eval()
    loss.eval()
    acc: dict[str, float] = {}
    n = 0
    for start in range(0, len(batcher), batch_size):
        idx = np.arange(start, min(start + batch_size, len(batcher)))
        terms = funnel_loss_on(art.ranker, loss, batcher.batch(idx)).as_dict()
        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + v * len(idx)
        n += len(idx)
    loss.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def train_ranker(
    art: PipelineArtifacts, splits: Splits, cfg: TrainingConfig, log: Any = None
) -> TrainHistory:
    """HSTU + PLE with the unified funnel loss and negative down-sampling; early stopping on
    the val unified loss (all terms)."""
    opt_cfg = cfg.ranker_optimizer
    ds = art.dataset
    train = RankerBatcher(splits.slates(ds, "train"), art)
    val = RankerBatcher(splits.slates(ds, "val"), art)
    model = art.ranker
    loss = UnifiedFunnelLoss(cfg.funnel_loss)
    rng = np.random.default_rng(cfg.seed)
    sample = train.batch(rng.permutation(len(train))[: min(len(train), 256)])
    model.fit_tabular_stats(sample.tabular, sample.candidate_mask)
    model.train()
    params = list(model.parameters()) + [p for p in loss.parameters() if p.requires_grad]
    opt = build_optimizer(torch.nn.ModuleList([model, loss]), opt_cfg)
    sched = WarmupCosine(opt, opt_cfg)
    downsampler = NegativeDownsampler(cfg.downsample.rate, cfg.downsample.seed)
    stop = EarlyStopping(patience=opt_cfg.patience, mode="min")
    hist = TrainHistory("ranker")
    hist.extra["num_trainable_parameters"] = int(sum(p.numel() for p in params))
    positives: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step, idx in enumerate(
        minibatches(len(train), opt_cfg.batch_size, opt_cfg.total_steps, rng), 1
    ):
        batch = train.batch(idx)
        keep = downsampler.train_mask(batch.y_click, batch.candidate_mask)
        terms = funnel_loss_on(model, loss, batch, keep)
        opt.zero_grad()
        terms.total.backward()
        clip_gradients(model, opt_cfg)
        opt.step()
        sched.step()
        hist.steps.append(step)
        hist.train_loss.append(float(terms.total.detach()))
        if len(positives) < 20:
            positives.append(
                {
                    "rows_before_downsampling": float(batch.candidate_mask.sum()),
                    "rows_after_downsampling": float(terms.num_rows),
                    "clicks": float(terms.num_clicked),
                    "applies": float((batch.y_apply * batch.candidate_mask).sum()),
                    "resolved_approvals": float(
                        (batch.y_approve * batch.approve_observed * batch.candidate_mask).sum()
                    ),
                    "resolved_applications": float(terms.num_resolved_applications),
                }
            )
        if step % opt_cfg.eval_every == 0 or step == opt_cfg.total_steps:
            ev = ranker_val_loss(art, loss, val) if len(val) else {"total": float("nan")}
            ev["step"] = float(step)
            hist.evaluations.append(ev)
            model.train()
            improved = stop.update(ev["total"], step, model)
            _log(
                log,
                f"[ranker] step {step} train {hist.train_loss[-1]:.4f} val {ev['total']:.4f} "
                f"{'*' if improved else ''}",
            )
            if stop.should_stop:
                hist.stopped_early = True
                break
    stop.restore(model)
    hist.best_step = stop.best_step
    hist.best_value = float(stop.best_value if stop.best_value is not None else float("nan"))
    hist.extra["positives_per_batch"] = (
        {k: float(np.mean([p[k] for p in positives])) for k in positives[0]} if positives else {}
    )
    hist.extra["downsample_rate"] = cfg.downsample.rate
    model.eval()
    hist.seconds = time.perf_counter() - t0
    return hist


# --------------------------------------------------------------- calibrators


@dataclass
class ScoredSlates:
    """Serving logits / labels for a set of slates, flattened over real candidate rows."""

    z1: npt.NDArray[np.float64]
    z2: npt.NDArray[np.float64]
    z3: npt.NDArray[np.float64]
    y_click: npt.NDArray[np.float64]
    y_apply: npt.NDArray[np.float64]
    y_approve: npt.NDArray[np.float64]
    y_approve_oracle: npt.NDArray[np.float64]
    p_click_true: npt.NDArray[np.float64]
    p_apply_true: npt.NDArray[np.float64]
    p_approve_true: npt.NDArray[np.float64]
    approve_observed: npt.NDArray[np.bool_]
    approve_weight: npt.NDArray[np.float64]
    candidate_mask: npt.NDArray[np.bool_]
    user_indices: npt.NDArray[np.int64]  # per row
    family_ids: npt.NDArray[np.int64]
    payouts: npt.NDArray[np.float64]
    amounts: npt.NDArray[np.float64]
    expected_amount: npt.NDArray[np.float64]
    slate_ids: npt.NDArray[np.int64]  # per row


@torch.no_grad()
def score_slates(
    art: PipelineArtifacts, slates: Sequence[ImpressionSlate], batch_size: int = 64
) -> ScoredSlates:
    from recsys.valuation.expected_value import expected_amount as _ea

    art.ranker.eval()
    batcher = RankerBatcher(slates, art)
    cols: dict[str, list[np.ndarray]] = {}

    def add(name: str, arr: np.ndarray) -> None:
        cols.setdefault(name, []).append(arr.reshape(-1))

    for start in range(0, len(batcher), batch_size):
        idx = np.arange(start, min(start + batch_size, len(batcher)))
        b = batcher.batch(idx)
        out = art.ranker.predict(b)
        k = b.slate_size
        for name, t in (("z1", out.z1), ("z2", out.z2), ("z3", out.z3)):
            add(name, t.numpy().astype(np.float64))
        for name in (
            "y_click", "y_apply", "y_approve", "y_approve_oracle", "approve_weight",
            "payouts", "amounts",
        ):  # fmt: skip
            add(name, getattr(b, name).numpy().astype(np.float64))
        add("p_click_true", b.p_click.numpy().astype(np.float64))
        add("p_apply_true", b.p_apply.numpy().astype(np.float64))
        add("p_approve_true", b.p_approve.numpy().astype(np.float64))
        add("approve_observed", b.approve_observed.numpy())
        add("candidate_mask", b.candidate_mask.numpy())
        add("user_indices", np.repeat(b.user_indices.numpy(), k))
        add("family_ids", b.family_ids.numpy())
        add("slate_ids", np.repeat(np.array([batcher.slates[int(i)].slate_id for i in idx]), k))
        amt = _ea(out.amount_logits) if out.amount_logits is not None else np.zeros((len(idx), k))
        add("expected_amount", amt)
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    return ScoredSlates(**cat)


def fit_calibrators(art: PipelineArtifacts, splits: Splits, cfg: TrainingConfig) -> CalibratorSet:
    """Fit on the calibration split with the down-sampling-corrected logits."""
    scored = score_slates(art, splits.slates(art.dataset, "calib"))
    cs = fit_funnel_calibrators(
        scored.z1, scored.z2, scored.z3, scored.y_click, scored.y_apply, scored.y_approve,
        scored.approve_observed, scored.approve_weight, scored.candidate_mask, cfg.calibration,
    )  # fmt: skip
    art.calibrators = cs
    return cs


# ----------------------------------------------------------------------- PRM


@dataclass
class PRMExamples:
    features: torch.Tensor  # (N, S, F)
    user_vec: torch.Tensor  # (N, d)
    mask: torch.Tensor  # (N, S)
    labels: torch.Tensor  # (N, S) y_click
    family_ids: torch.Tensor  # (N, S)
    utility: torch.Tensor  # (N, S)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def subset(self, idx: npt.NDArray[np.int64]) -> PRMExamples:
        t = torch.as_tensor(idx, dtype=torch.int64)
        return PRMExamples(
            self.features[t], self.user_vec[t], self.mask[t], self.labels[t],
            self.family_ids[t], self.utility[t],
        )  # fmt: skip


def _row(arr: np.ndarray, i: int, cols: npt.NDArray[np.int64]) -> torch.Tensor:
    """``(1, len(cols))`` float32 slice of row ``i``."""
    return torch.as_tensor(arr[i : i + 1, cols], dtype=torch.float32)


@torch.no_grad()
def build_prm_examples(
    art: PipelineArtifacts, slates: Sequence[ImpressionSlate], batch_size: int = 64
) -> PRMExamples:
    """Score, calibrate and value each slate; keep the top-``slate_size`` survivors by
    utility as PRM slots with their click labels (slates with no survivor are dropped)."""
    art.ranker.eval()
    s = art.config.models.slate_size
    batcher = RankerBatcher(slates, art)
    feats, users, masks, labels, fams, utils = [], [], [], [], [], []
    for start in range(0, len(batcher), batch_size):
        idx = np.arange(start, min(start + batch_size, len(batcher)))
        b = batcher.batch(idx)
        out = art.ranker.predict(b)
        val = art.valuation.value(b, out, art.calibrators)
        for i in range(len(idx)):
            surv = np.flatnonzero(val.keep[i])
            if surv.size == 0:
                continue
            top = surv[np.argsort(-val.utility[i, surv], kind="stable")][:s]
            n = top.size
            pad = np.concatenate([top, np.zeros(s - n, dtype=np.int64)])
            ti = torch.as_tensor(pad)

            f = build_prm_features(
                out.h_cand[i : i + 1, ti],
                _row(val.p1, i, pad),
                _row(val.p2, i, pad),
                _row(val.p3, i, pad),
                _row(val.ev, i, pad),
                _row(val.nb, i, pad),
                _row(val.utility, i, pad),
                b.family_ids[i : i + 1, ti],
            )
            m = torch.zeros(s, dtype=torch.bool)
            m[:n] = True
            feats.append(f[0])
            users.append(out.h_user[i])
            masks.append(m)
            labels.append(b.y_click[i, ti] * m)
            fams.append(b.family_ids[i, ti])
            utils.append(torch.as_tensor(val.utility[i, pad], dtype=torch.float32))
    if not feats:
        raise ValueError("no PRM examples: every slate lost all candidates to the guardrails")
    return PRMExamples(
        torch.stack(feats), torch.stack(users), torch.stack(masks), torch.stack(labels),
        torch.stack(fams), torch.stack(utils),
    )  # fmt: skip


def train_prm(
    art: PipelineArtifacts, splits: Splits, cfg: TrainingConfig, log: Any = None
) -> TrainHistory:
    opt_cfg = cfg.prm_optimizer
    prm_cfg = art.prm.config
    ds = art.dataset
    train = build_prm_examples(art, splits.slates(ds, "train"))
    val_slates = splits.slates(ds, "val")
    val = build_prm_examples(art, val_slates) if val_slates else None
    model = art.prm
    model.train()
    opt = build_optimizer(model, opt_cfg)
    sched = WarmupCosine(opt, opt_cfg)
    stop = EarlyStopping(patience=opt_cfg.patience, mode="min")
    hist = TrainHistory("prm")
    hist.extra["num_train_slates"] = len(train)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()

    def loss_on(ex: PRMExamples) -> torch.Tensor:
        scores = model(ex.features, ex.user_vec, ex.mask).scores
        return prm_listwise_loss(
            scores, ex.labels, ex.mask, ex.family_ids, prm_cfg.loss_cannibalization_weight,
            prm_cfg.prm_target, ex.utility, prm_cfg.utility_temperature,
        )  # fmt: skip

    for step, idx in enumerate(
        minibatches(len(train), opt_cfg.batch_size, opt_cfg.total_steps, rng), 1
    ):
        loss = loss_on(train.subset(idx))
        opt.zero_grad()
        loss.backward()
        clip_gradients(model, opt_cfg)
        opt.step()
        sched.step()
        hist.steps.append(step)
        hist.train_loss.append(float(loss.detach()))
        if step % opt_cfg.eval_every == 0 or step == opt_cfg.total_steps:
            model.eval()
            with torch.no_grad():
                vloss = float(loss_on(val)) if val is not None else float("nan")
            model.train()
            hist.evaluations.append({"step": step, "val_listwise_loss": vloss})
            improved = stop.update(vloss, step, model)
            _log(
                log,
                f"[prm] step {step} train {hist.train_loss[-1]:.4f} val {vloss:.4f}"
                f"{' *' if improved else ''}",
            )
            if stop.should_stop:
                hist.stopped_early = True
                break
    stop.restore(model)
    hist.best_step = stop.best_step
    hist.best_value = float(stop.best_value if stop.best_value is not None else float("nan"))
    model.eval()
    hist.seconds = time.perf_counter() - t0
    return hist


# ------------------------------------------------------------------ train all


@dataclass
class TrainAllResult:
    artifacts: PipelineArtifacts
    splits: Splits
    histories: dict[str, TrainHistory]

    def histories_as_dict(self) -> dict[str, Any]:
        return {k: v.as_dict() for k, v in self.histories.items()}


def train_all(
    ds: SyntheticDataset, cfg: TrainingConfig | None = None, log: Any = None
) -> TrainAllResult:
    """RQ-VAE -> semantic ids -> TIGER -> HSTU + PLE -> calibrators -> PRM."""
    cfg = cfg or TrainingConfig()
    splits = split_by_user(ds.num_users, cfg.split)
    art = build_artifacts(ds, cfg, seed=cfg.seed)
    histories: dict[str, TrainHistory] = {}
    histories["rqvae"] = train_rqvae(art, cfg, log)
    codes = assign_item_codes(art.rqvae, standardize_catalog(ds))
    rqvae_state = art.rqvae.state_dict()
    art = build_artifacts(ds, cfg, seed=cfg.seed, item_codes=codes)
    art.rqvae.load_state_dict(rqvae_state)
    histories["tiger"] = train_tiger(art, splits, cfg, log)
    histories["ranker"] = train_ranker(art, splits, cfg, log)
    cs = fit_calibrators(art, splits, cfg)
    _log(log, f"[calibration] {cs.reports}")
    histories["prm"] = train_prm(art, splits, cfg, log)
    art.eval()
    return TrainAllResult(art, splits, histories)


__all__ = [
    "PRMExamples",
    "RankerBatcher",
    "RetrievalEval",
    "ScoredSlates",
    "TigerBatcher",
    "TrainAllResult",
    "TrainHistory",
    "build_prm_examples",
    "fit_calibrators",
    "funnel_loss_on",
    "minibatches",
    "next_item_positives",
    "ranker_val_loss",
    "retrieval_eval",
    "score_slates",
    "train_all",
    "train_prm",
    "train_ranker",
    "train_rqvae",
    "train_tiger",
]


def _touch(_: ImpressionCollator | SequenceCollator | None = None) -> None:
    """Imports kept for type references in docstrings."""
