"""Offline evaluation that produces the ``docs/results/*.md`` tables.

Everything the write-up cites is generated here from trained artifacts, never typed by
hand: retrieval (D7), funnel quality (D2.5), calibration comparison (D5), the
pending-policy ablation (D1), the alpha Pareto sweep (D5), latency (D6) and the
measured positives-per-batch (D8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from recsys.data.schema import (
    FAMILY_ORDER,
    NUM_FAMILIES,
    TIER_ORDER,
    ApplicationStatus,
    ImpressionSlate,
    InteractionRecord,
    SyntheticDataset,
)
from recsys.data.synthetic_generator import (
    GeneratorConfig,
    SyntheticFintechDataGenerator,
    dataset_summary,
)
from recsys.metrics.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    max_calibration_error,
)
from recsys.metrics.ranking_metrics import auc, gauc, normalized_cross_entropy, pr_auc, recall_at_k
from recsys.metrics.retrieval_baselines import (
    eligible_popularity_top_k,
    eligible_random_top_k,
    positive_item_counts,
)
from recsys.serving.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    _sigmoid,
    apply_calibrator,
)
from recsys.serving.pipeline import (
    PipelineArtifacts,
    RecommendationPipeline,
    build_artifacts,
    latency_summary,
)
from recsys.training.config import PendingConfig, TrainingConfig
from recsys.training.splits import Splits
from recsys.training.trainers import (
    RankerBatcher,
    ScoredSlates,
    next_item_positives,
    retrieval_eval,
    score_slates,
    train_ranker,
)
from recsys.valuation.pareto import ParetoRow, pareto_sweep, pareto_table_markdown

FloatArray = npt.NDArray[np.float64]
TIER_NAMES = {i: t.value for i, t in enumerate(TIER_ORDER)}
FAMILY_NAMES = {i: f.value for i, f in enumerate(FAMILY_ORDER)}


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(v: Any) -> str:
        if isinstance(v, float):
            return "nan" if v != v else f"{v:.4f}"
        return str(v)

    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _nanmean(x: FloatArray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ------------------------------------------------------------------- dataset


def dataset_summary_markdown(ds: SyntheticDataset, splits: Splits) -> str:
    rows = [(k, v) for k, v in dataset_summary(ds).items()]
    sizes = splits.sizes()
    body = "# Dataset summary\n\n" + md_table(("statistic", "value"), rows)
    body += "\n## Split sizes (users)\n\n" + md_table(("split", "users"), list(sizes.items()))
    return body


# ----------------------------------------------------------------- retrieval


@dataclass
class RetrievalTables:
    markdown: str
    recall_next_item: dict[int, float]
    baselines: dict[str, dict[int, float]]


def _padded_positives(
    records: Sequence[InteractionRecord], per_user: dict[int, set[int]]
) -> npt.NDArray[np.int64]:
    width = max((len(v) for v in per_user.values()), default=1)
    out = np.zeros((len(records), max(width, 1)), dtype=np.int64)
    for i, r in enumerate(records):
        items = sorted(per_user.get(r.user_index, set()))
        out[i, : len(items)] = items
    return out


def retrieval_tables(
    art: PipelineArtifacts, ds: SyntheticDataset, splits: Splits, ks: Sequence[int] = (10, 50, 100)
) -> RetrievalTables:
    ks = tuple(k for k in ks if k <= art.config.models.num_candidates) or (
        art.config.models.num_candidates,
    )
    k_max = max(ks)
    test_records = [r for r in splits.records(ds, "test") if r.target.item_id > 0]
    positives = next_item_positives(test_records)
    ev = retrieval_eval(art, test_records, positives, ks)
    counts = positive_item_counts(splits.records(ds, "train"), ds.num_items)
    pop = eligible_popularity_top_k(counts, ev.allowed, k_max)
    rnd = eligible_random_top_k(ev.allowed, k_max, np.random.default_rng(0))
    valid = (positives > 0) & np.take_along_axis(ev.allowed, positives, axis=1)
    baselines = {
        "eligible-popularity": {k: recall_at_k(pop, positives, valid, k).mean for k in ks},
        "eligible-random": {k: recall_at_k(rnd, positives, valid, k).mean for k in ks},
    }
    rows = [("TIGER (beam)", *[ev.recall[k] for k in ks])]
    rows += [(name, *[b[k] for k in ks]) for name, b in baselines.items()]
    md = "# Retrieval metrics (test split, next-item positives)\n\n"
    md += md_table(("retriever", *[f"Recall@{k}" for k in ks]), rows)
    md += (
        f"\nUsers counted: {ev.num_users}; positives excluded as ineligible: "
        f"{ev.num_excluded_positives}; beam yield |C_u| / {art.config.models.num_candidates}: "
        f"{ev.beam_yield:.4f}\n"
    )
    # slices
    tiers = np.array([ds.users[r.user_index].tier_index for r in test_records])
    fam = ds.family_by_item()
    target_family = np.array([fam[r.target.item_id] for r in test_records])
    md += f"\n## Recall@{k_max} by credit tier\n\n" + md_table(
        ("tier", "users", f"Recall@{k_max}"),
        [
            (
                TIER_NAMES[t],
                int(np.isfinite(ev.per_user[tiers == t]).sum()),
                _nanmean(ev.per_user[tiers == t]),
            )
            for t in range(len(TIER_ORDER))
        ],
    )
    md += f"\n## Recall@{k_max} by target family\n\n" + md_table(
        ("family", "users", f"Recall@{k_max}"),
        [
            (
                FAMILY_NAMES[f],
                int(np.isfinite(ev.per_user[target_family == f]).sum()),
                _nanmean(ev.per_user[target_family == f]),
            )
            for f in range(NUM_FAMILIES)
        ],
    )
    # slate-level positives (click / apply / resolved approve) for the same users
    test_slates = splits.slates(ds, "test")
    snap = ds.snapshot_at_days
    per_kind: dict[str, dict[int, set[int]]] = {"click": {}, "apply": {}, "approve": {}}
    for s in test_slates:
        for item, yc, ya, yp, d in zip(
            s.candidate_item_ids,
            s.y_click,
            s.y_apply,
            s.y_approve,
            s.decision_delay_days,
            strict=True,
        ):
            if yc:
                per_kind["click"].setdefault(s.user_index, set()).add(item)
            if ya:
                per_kind["apply"].setdefault(s.user_index, set()).add(item)
            if ya and s.served_at_days + d <= snap and yp:
                per_kind["approve"].setdefault(s.user_index, set()).add(item)
    rows = []
    for kind, per_user in per_kind.items():
        pos = _padded_positives(test_records, per_user)
        val = (pos > 0) & np.take_along_axis(ev.allowed, pos, axis=1)
        r = {k: recall_at_k(ev.candidates, pos, val, k) for k in ks}
        rows.append((kind, r[k_max].num_users, *[r[k].mean for k in ks]))
    md += "\n## Slate-level positives (same beams)\n\n" + md_table(
        ("positive definition", "users", *[f"Recall@{k}" for k in ks]), rows
    )
    return RetrievalTables(md, ev.recall, baselines)


# -------------------------------------------------------------------- funnel


@dataclass
class FunnelTables:
    markdown: str
    metrics: dict[str, dict[str, float]]


def funnel_tables(art: PipelineArtifacts, scored: ScoredSlates) -> FunnelTables:
    m = scored.candidate_mask
    p1, p2, p3 = art.calibrators.calibrate(scored.z1, scored.z2, scored.z3)
    groups = scored.user_indices
    clicked = m & (scored.y_click > 0.5)
    resolved = m & (scored.y_apply > 0.5) & scored.approve_observed
    w = scored.approve_weight
    metrics: dict[str, dict[str, float]] = {}
    for name, y, p, sel, wt in (
        ("click", scored.y_click, p1, m, None),
        ("apply | click", scored.y_apply, p2, clicked, None),
        ("approve | apply (resolved, weighted)", scored.y_approve, p3, resolved, w),
    ):
        ws = None if wt is None else wt[sel]
        metrics[name] = {
            "rows": float(sel.sum()),
            "positives": float(y[sel].sum()),
            "AUC": auc(y[sel], p[sel], ws),
            "PR-AUC": pr_auc(y[sel], p[sel], ws),
            "GAUC": gauc(y[sel], p[sel], groups[sel], ws),
            "NCE": normalized_cross_entropy(y[sel], p[sel], ws),
            "ECE": expected_calibration_error(p[sel], y[sel], weights=ws),
        }
    ceiling: dict[str, dict[str, float]] = {}
    for name, y, p, sel, wt in (
        ("click", scored.y_click, scored.p_click_true, m, None),
        ("apply | click", scored.y_apply, scored.p_apply_true, clicked, None),
        ("approve | apply (resolved, weighted)", scored.y_approve, scored.p_approve_true,
         resolved, w),
    ):  # fmt: skip
        ws = None if wt is None else wt[sel]
        ceiling[name] = {
            "AUC": auc(y[sel], p[sel], ws),
            "PR-AUC": pr_auc(y[sel], p[sel], ws),
            "GAUC": gauc(y[sel], p[sel], groups[sel], ws),
            "NCE": normalized_cross_entropy(y[sel], p[sel], ws),
        }
    pf = p1 * p2 * p3
    realized = float((scored.y_approve * scored.payouts * m * scored.approve_observed).sum())
    expected = float((pf * scored.payouts * m).sum())
    md = "# Funnel metrics (test split, calibrated probabilities)\n\n"
    md += md_table(
        ("task", "rows", "positives", "AUC", "PR-AUC", "GAUC", "NCE", "ECE"),
        [
            (
                k,
                int(v["rows"]),
                int(v["positives"]),
                v["AUC"],
                v["PR-AUC"],
                v["GAUC"],
                v["NCE"],
                v["ECE"],
            )
            for k, v in metrics.items()
        ],
    )
    md += (
        f"\nRealized partner revenue on resolved rows: ${realized:,.0f} (a **lower bound**: "
        f"pending applications are excluded, D4); expected revenue Σ P_funded · payout: "
        f"${expected:,.0f}.\n"
    )
    md += (
        "\n## Oracle ceiling (the generator's own probabilities scored on the same rows)\n\n"
        "Labels are Bernoulli draws from these probabilities, so no model can beat this "
        "ordering in expectation.\n\n"
    )
    md += md_table(
        ("task", "AUC", "PR-AUC", "GAUC", "NCE"),
        [(k, v["AUC"], v["PR-AUC"], v["GAUC"], v["NCE"]) for k, v in ceiling.items()],
    )
    is_pending = scored.status == int(ApplicationStatus.PENDING)
    pending_rows = int((m & (scored.y_apply > 0.5) & is_pending).sum())
    md += (
        "Pending applications in the test split (excluded from approval metrics): "
        f"{pending_rows}.\n"
    )
    return FunnelTables(md, metrics)


def training_summary_markdown(histories: dict[str, Any], cfg: TrainingConfig) -> str:
    """One row per model from ``artifacts/histories.json``: steps run, selection criterion,
    best step / value, early stop, wall time; plus RQ-VAE utilization per level."""
    criteria = {
        "rqvae": f"recon MSE s.t. min utilization >= {cfg.rqvae_min_utilization:g}",
        "tiger": "val Recall@100 (checkpoint), val next-SID loss (stop)",
        "ranker": "val unified funnel loss (total)",
        "prm": "val listwise loss",
    }
    budgets = {
        "rqvae": cfg.rqvae_optimizer.total_steps,
        "tiger": cfg.tiger_optimizer.total_steps,
        "ranker": cfg.ranker_optimizer.total_steps,
        "prm": cfg.prm_optimizer.total_steps,
    }
    rows = []
    for name in ("rqvae", "tiger", "ranker", "prm"):
        h = histories.get(name)
        if not h:
            continue
        rows.append(
            (
                name,
                int(len(h.get("train_loss", []))),
                int(budgets[name]),
                criteria[name],
                int(h.get("best_step", -1)),
                float(h.get("best_value", float("nan"))),
                "yes" if h.get("stopped_early") else "no",
                float(h.get("seconds", 0.0)),
            )
        )
    md = "# Training summary (from artifacts/histories.json)\n\n"
    md += md_table(
        (
            "model", "steps run", "step budget", "selection criterion", "best step", "best value",
            "early stop", "seconds",
        ),
        rows,
    )  # fmt: skip
    rq = histories.get("rqvae", {})
    evals = rq.get("evaluations", [])
    if evals:
        last = evals[-1]
        util = [(k, float(v)) for k, v in last.items() if k.startswith("utilization_")]
        md += "\n## RQ-VAE codebook utilization at the last evaluation\n\n"
        md += md_table(("level", "utilization"), [(k.split("_")[1], v) for k, v in util])
        md += (
            f"\nUtilization constraint met: "
            f"{'yes' if rq.get('extra', {}).get('utilization_constraint_met') else 'no'}.\n"
        )
    extra = histories.get("ranker", {}).get("extra", {})
    if "num_trainable_parameters" in extra:
        md += f"\nRanker trainable parameters: {int(extra['num_trainable_parameters'])}.\n"
    prm_extra = histories.get("prm", {}).get("extra", {})
    if "num_train_slates" in prm_extra:
        md += f"PRM training slates (>= 1 positive): {int(prm_extra['num_train_slates'])}.\n"
    return md


def positives_per_batch_markdown(histories: dict[str, Any], cfg: TrainingConfig) -> str:
    extra = histories.get("ranker", {}).get("extra", {})
    ppb = extra.get("positives_per_batch", {})
    rows = [(k, v) for k, v in ppb.items()]
    md = "# Measured positives per ranker batch (first 20 training batches)\n\n"
    md += (
        f"Batch = {cfg.ranker_optimizer.batch_size} slates; negative down-sampling rate r = "
        f"{cfg.downsample.rate}.\n\n"
    )
    return md + md_table(("quantity", "mean per batch"), rows)


# --------------------------------------------------------------- calibration


def calibration_tables(art: PipelineArtifacts, calib: ScoredSlates, test: ScoredSlates) -> str:
    """Fit isotonic and Platt on every tower (calibration split) and score both on test."""
    tiers_test = np.array([art.users[int(u)].tier_index for u in test.user_indices])
    rows: list[tuple[Any, ...]] = []
    tier_rows: list[tuple[Any, ...]] = []
    towers = {
        "click": (
            calib.z1,
            calib.y_click,
            calib.candidate_mask,
            None,
            test.z1,
            test.y_click,
            test.candidate_mask,
            None,
        ),
        "apply": (
            calib.z2,
            calib.y_apply,
            calib.candidate_mask & (calib.y_click > 0.5),
            None,
            test.z2,
            test.y_apply,
            test.candidate_mask & (test.y_click > 0.5),
            None,
        ),
        "approve": (
            calib.z3,
            calib.y_approve,
            calib.candidate_mask & (calib.y_apply > 0.5) & calib.approve_observed,
            calib.approve_weight,
            test.z3,
            test.y_approve,
            test.candidate_mask & (test.y_apply > 0.5) & test.approve_observed,
            test.approve_weight,
        ),
    }
    for tower, (zc, yc, mc, wc, zt, yt, mt, wt) in towers.items():
        fits: dict[str, Any] = {}
        if mc.sum() >= 2 and yc[mc].min() != yc[mc].max():
            iso = IsotonicCalibrator()
            iso.fit(_sigmoid(zc[mc]), yc[mc], None if wc is None else wc[mc])
            platt = PlattCalibrator()
            platt.fit(zc[mc], yc[mc], None if wc is None else wc[mc])
            fits = {"isotonic": iso, "platt": platt}
        served = art.calibrators.reports.get(tower, "n/a")
        wtt = None if wt is None else wt[mt]
        variants: list[tuple[str, FloatArray]] = [("uncalibrated (sigmoid)", _sigmoid(zt[mt]))]
        variants += [(name, apply_calibrator(cal, zt[mt])) for name, cal in fits.items()]
        for name, p in variants:
            rows.append(
                (
                    tower,
                    name,
                    int(mt.sum()),
                    expected_calibration_error(p, yt[mt], weights=wtt),
                    max_calibration_error(p, yt[mt], weights=wtt),
                    brier_score(p, yt[mt], weights=wtt),
                )
            )
            for t in range(len(TIER_ORDER)):
                sel = tiers_test[mt] == t
                if sel.sum() < 20:
                    continue
                ws = None if wtt is None else wtt[sel]
                tier_rows.append(
                    (
                        tower,
                        name,
                        TIER_NAMES[t],
                        int(sel.sum()),
                        expected_calibration_error(p[sel], yt[mt][sel], weights=ws),
                        brier_score(p[sel], yt[mt][sel], weights=ws),
                    )
                )
        rows.append((tower, f"served: {served}", "", "", "", ""))
    md = "# Calibration (fitted on the calibration split, scored on the test split)\n\n"
    md += md_table(("tower", "method", "rows", "ECE", "MCE", "Brier"), rows)
    md += "\n## By credit tier (ECE / Brier)\n\n"
    md += md_table(("tower", "method", "tier", "rows", "ECE", "Brier"), tier_rows)
    return md


# ------------------------------------------------------------------ ablation


MORTGAGE_YOUNG_ABLATION = GeneratorConfig(
    num_users=800, num_products=150, seed=21, slates_per_user=3, slate_size=20,
    ineligible_per_slate=2, snapshot_at_days=40.0, recency_window_days=10.0,
    served_window_days=30.0, family_mix=(0.05, 0.05, 0.10, 0.10, 0.70), min_history=4,
    max_history=12, mean_gap_days=1.0,
)  # fmt: skip


def pending_policy_ablation(
    base_cfg: TrainingConfig,
    generator: GeneratorConfig = MORTGAGE_YOUNG_ABLATION,
    steps: int = 300,
    log: Any = None,
) -> str:
    """Train the ranker three times (one per ``pending_policy``) on a mortgage-heavy,
    young-cut-off dataset and report the approval-probability bias on the mortgage slice
    against the generator's truth."""
    from recsys.training.splits import split_by_user

    ds = SyntheticFintechDataGenerator(generator).generate()
    splits = split_by_user(ds.num_users, base_cfg.split)
    rows = []
    for policy in ("drop", "ipw", "negative"):
        opt = replace(base_cfg.ranker_optimizer, total_steps=steps, warmup_steps=min(50, steps),
                      eval_every=max(steps // 4, 1))  # fmt: skip
        cfg = replace(base_cfg, pending=PendingConfig(policy=policy), ranker_optimizer=opt)
        art = build_artifacts(ds, cfg, seed=cfg.seed)
        train_ranker(art, splits, cfg, log)
        scored = score_slates(art, splits.slates(ds, "test"))
        m = scored.candidate_mask & (scored.y_apply > 0.5)
        mort = m & (scored.family_ids == 4)  # mortgage family index
        p3 = _sigmoid(scored.z3)
        truth = scored.p_approve_true
        bias = float(p3[mort].mean() - truth[mort].mean())
        a = auc(scored.y_approve_oracle[mort], p3[mort])
        ece = expected_calibration_error(p3[mort], scored.y_approve_oracle[mort])
        pending_share = float((scored.status[mort] == int(ApplicationStatus.PENDING)).mean())
        rows.append(
            (
                policy,
                int(mort.sum()),
                pending_share,
                float(truth[mort].mean()),
                float(p3[mort].mean()),
                bias,
                a,
                ece,
            )
        )
    md = "# Pending-policy ablation (mortgage slice, young cut-off)\n\n"
    md += (
        f"Dataset: {generator.num_users} users, {generator.num_products} products, family mix "
        f"{generator.family_mix}, snapshot {generator.snapshot_at_days} d, served window "
        f"{generator.served_window_days} d; ranker trained {steps} steps per policy.  Bias = "
        "mean(p̂3) − mean(generator p_approve) on applied mortgage rows of the test split; "
        "AUC / ECE against the *oracle* approval label.\n\n"
    )
    md += md_table(
        (
            "pending_policy",
            "applied mortgage rows",
            "pending share",
            "mean p_approve (truth)",
            "mean p̂3",
            "bias",
            "AUC vs oracle",
            "ECE vs oracle",
        ),
        rows,
    )
    return md


# ------------------------------------------------------------------- latency


def latency_markdown(
    art: PipelineArtifacts, user_indices: Sequence[int]
) -> tuple[str, dict[str, dict[str, float]]]:
    pipe = RecommendationPipeline(art)
    pipe.run(int(user_indices[0]))  # warm-up
    results = pipe.run_many(user_indices)
    summary = latency_summary(results)
    rows = [(name, s["p50"], s["p99"], s["max"]) for name, s in summary.items()]
    md = f"# Serving latency (CPU, B = 1, {len(results)} test users)\n\n"
    md += md_table(("stage", "p50 ms", "p99 ms", "max ms"), rows)
    total = summary["total"]["p50"]
    verdict = "within" if total <= 10.0 else "ABOVE"
    md += f"\nTotal p50 = {total:.2f} ms, {verdict} the 10 ms budget.\n"
    md += (
        f"Served slates: mean size {np.mean([r.size for r in results]):.2f}, mean eligible "
        f"{np.mean([r.num_eligible for r in results]):.1f}, mean retrieved "
        f"{np.mean([r.num_retrieved for r in results]):.1f}.\n"
    )
    return md, summary


# --------------------------------------------------------------------- pareto


def pareto_rows_for_slates(
    art: PipelineArtifacts, ds: SyntheticDataset, slates: Sequence[ImpressionSlate], k: int
) -> list[ParetoRow]:
    """Score + calibrate + value the slates once, then sweep alpha on the valuation."""
    batcher = RankerBatcher(slates, art)
    cols: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(batcher), 64):
            idx = np.arange(start, min(start + 64, len(batcher)))
            b = batcher.batch(idx)
            val = art.valuation.value(b, art.ranker.predict(b), art.calibrators)
            fam = b.family_ids.numpy()
            pend = (
                b.pending_family_mask.numpy()
                & art.valuation.penalize_families[np.clip(fam, 0, NUM_FAMILIES - 1)]
            )
            nb_raw = val.nb + art.valuation.utility.pending_family_penalty * pend
            for name, arr in (
                ("p_funded", val.p_funded), ("payout", b.payouts.numpy()), ("nb", nb_raw),
                ("family", fam), ("mask", b.candidate_mask.numpy()),
                ("refinance_ok", val.refinance_ok), ("pending", pend),
                ("tier", np.array([ds.users[int(u)].tier_index for u in b.user_indices])),
            ):  # fmt: skip
                cols.setdefault(name, []).append(np.asarray(arr))
    c = {name: np.concatenate(v) for name, v in cols.items()}
    return pareto_sweep(
        c["p_funded"], c["payout"], c["nb"], c["family"], c["mask"], c["refinance_ok"],
        c["pending"], c["tier"], k=k, config=art.valuation.utility,
    )  # fmt: skip


def pareto_markdown(
    art: PipelineArtifacts, ds: SyntheticDataset, slates: Sequence[ImpressionSlate], k: int
) -> str:
    rows = pareto_rows_for_slates(art, ds, slates, k)
    return "# Pareto sweep over alpha (test slates)\n\n" + pareto_table_markdown(rows, k)
