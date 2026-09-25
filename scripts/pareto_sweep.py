"""Offline Pareto sweep over the platform-vs-user weight alpha (D5).

Usage::

    python scripts/pareto_sweep.py --data data/generated --artifacts artifacts \
        --out docs/results/pareto_sweep.md

Re-values the test slates under every alpha in {0, 0.1, ..., 1} and reports
ExpectedRevenue@10, ExpectedUserBenefit@10 and HarmRate@10, overall and by credit tier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data.synthetic_generator import load_dataset  # noqa: E402
from recsys.serving.evaluation import pareto_markdown  # noqa: E402
from recsys.serving.pipeline import PipelineArtifacts  # noqa: E402
from recsys.training.config import TrainingConfig  # noqa: E402
from recsys.training.splits import split_by_user  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/pareto_sweep.md"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ds = load_dataset(args.data)
    cfg = TrainingConfig.tiny() if args.quick else TrainingConfig()
    art = PipelineArtifacts.load(args.artifacts, ds, cfg)
    splits = split_by_user(ds.num_users, cfg.split)
    md = pareto_markdown(art, ds, splits.slates(ds, "test"), cfg.models.slate_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
