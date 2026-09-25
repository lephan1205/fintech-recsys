"""Serve a few users through the full cascade and print slates + per-stage latency.

Usage::

    python scripts/run_e2e_pipeline.py --data data/generated --artifacts artifacts --serve-users 5
    python scripts/run_e2e_pipeline.py --untrained --num-users 200 --num-products 300   # demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data.schema import FAMILY_ORDER  # noqa: E402
from recsys.data.synthetic_generator import (  # noqa: E402
    GeneratorConfig,
    SyntheticFintechDataGenerator,
    load_dataset,
)
from recsys.serving.pipeline import (  # noqa: E402
    PipelineArtifacts,
    RecommendationPipeline,
    SlateResult,
    build_artifacts,
    latency_summary,
)
from recsys.training.config import TrainingConfig  # noqa: E402


def print_slate(res: SlateResult, family_by_item: np.ndarray, products: list) -> None:  # type: ignore[type-arg]
    print(
        f"user {res.user_index}: eligible {res.num_eligible}, retrieved {res.num_retrieved}, "
        f"post-filter {res.num_after_post_filter}, survivors {res.num_after_guardrails}, "
        f"total {res.telemetry.total_ms:.1f} ms"
    )
    print(
        f"{'#':>2} {'item':>5} {'family':<22} {'p_click':>8} {'p_apply':>8} {'p_appr':>7} "
        f"{'EV $':>8} {'E[amt] $':>10} {'NB $':>9} {'U $':>8}"
    )
    for i, item in enumerate(res.item_ids):
        fam = FAMILY_ORDER[int(family_by_item[item])].value
        print(
            f"{i + 1:>2} {item:>5} {fam:<22} {res.p_click[i]:>8.3f} {res.p_apply[i]:>8.3f} "
            f"{res.p_approve[i]:>7.3f} {res.expected_value[i]:>8.2f} "
            f"{res.expected_amount[i]:>10.0f} {res.net_user_benefit[i]:>9.1f} "
            f"{res.utility[i]:>8.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--untrained", action="store_true", help="random weights on a fresh dataset")
    ap.add_argument("--num-users", type=int, default=200)
    ap.add_argument("--num-products", type=int, default=300)
    ap.add_argument("--serve-users", type=int, default=5)
    ap.add_argument("--latency-users", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="artifacts were trained with --quick")
    args = ap.parse_args()

    if args.untrained:
        ds = SyntheticFintechDataGenerator(
            GeneratorConfig(
                num_users=args.num_users, num_products=args.num_products, seed=args.seed
            )
        ).generate()
        art = build_artifacts(ds, TrainingConfig(), seed=args.seed)
    else:
        ds = load_dataset(args.data)
        cfg = TrainingConfig.tiny() if args.quick else TrainingConfig()
        art = PipelineArtifacts.load(args.artifacts, ds, cfg)
    pipe = RecommendationPipeline(art)
    rng = np.random.default_rng(args.seed)
    users = rng.choice(ds.num_users, size=min(args.latency_users, ds.num_users), replace=False)
    results = pipe.run_many(users.tolist())
    fam = ds.family_by_item()
    for res in results[: args.serve_users]:
        print_slate(res, fam, ds.products)
        print()
    print("per-stage latency (ms), B = 1, CPU")
    print(f"{'stage':<14} {'p50':>8} {'p99':>8} {'max':>8}")
    for name, s in latency_summary(results).items():
        print(f"{name:<14} {s['p50']:>8.2f} {s['p99']:>8.2f} {s['max']:>8.2f}")
    print(f"compliance: 0 violations over {len(results)} users (asserted in the pipeline)")


if __name__ == "__main__":
    main()
