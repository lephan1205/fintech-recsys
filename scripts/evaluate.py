"""Evaluate trained artifacts and write every ``docs/results/*.md`` table.

Usage::

    python scripts/evaluate.py --data data/generated --artifacts artifacts --out docs/results
    python scripts/evaluate.py ... --skip-ablation          # skip the 3x ranker retraining

Outputs: dataset_summary.md, retrieval_metrics.md, funnel_metrics.md,
positives_per_batch.md, calibration.md, pending_policy_ablation.md, pareto_sweep.md,
latency.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data.synthetic_generator import load_dataset  # noqa: E402
from recsys.serving.evaluation import (  # noqa: E402
    calibration_tables,
    dataset_summary_markdown,
    funnel_tables,
    latency_markdown,
    pareto_markdown,
    pending_policy_ablation,
    positives_per_batch_markdown,
    retrieval_tables,
)
from recsys.serving.pipeline import PipelineArtifacts  # noqa: E402
from recsys.training.config import TrainingConfig  # noqa: E402
from recsys.training.splits import split_by_user  # noqa: E402
from recsys.training.trainers import score_slates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("docs/results"))
    ap.add_argument("--quick", action="store_true", help="artifacts were trained with --quick")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--ablation-steps", type=int, default=300)
    ap.add_argument("--latency-users", type=int, default=50)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.data)
    cfg = TrainingConfig.tiny() if args.quick else TrainingConfig()
    art = PipelineArtifacts.load(args.artifacts, ds, cfg)
    splits = split_by_user(ds.num_users, cfg.split)
    histories = {}
    hist_path = args.artifacts / "histories.json"
    if hist_path.exists():
        histories = json.loads(hist_path.read_text(encoding="utf-8"))

    def write(name: str, text: str) -> None:
        (out / name).write_text(text, encoding="utf-8")
        print(f"wrote {out / name}")

    t0 = time.perf_counter()
    write("dataset_summary.md", dataset_summary_markdown(ds, splits))
    write("retrieval_metrics.md", retrieval_tables(art, ds, splits).markdown)
    test_scored = score_slates(art, splits.slates(ds, "test"))
    calib_scored = score_slates(art, splits.slates(ds, "calib"))
    write("funnel_metrics.md", funnel_tables(art, test_scored).markdown)
    write("positives_per_batch.md", positives_per_batch_markdown(histories, cfg))
    write("calibration.md", calibration_tables(art, calib_scored, test_scored))
    write(
        "pareto_sweep.md",
        pareto_markdown(art, ds, splits.slates(ds, "test"), cfg.models.slate_size),
    )
    rng = np.random.default_rng(0)
    test_users = splits.users("test")
    pick = rng.choice(test_users, size=min(args.latency_users, test_users.size), replace=False)
    md, _ = latency_markdown(art, [int(u) for u in pick])
    write("latency.md", md)
    if not args.skip_ablation:
        write(
            "pending_policy_ablation.md",
            pending_policy_ablation(cfg, steps=args.ablation_steps, log=print),
        )
    print(f"evaluation finished in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
