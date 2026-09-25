"""Generate a synthetic credit-marketplace dataset (format v2) and print its summary.

Usage::

    python scripts/generate_dataset.py --out data/generated --num-users 3000 \
        --num-products 2000 --slates-per-user 4 --slate-size 100 --ineligible-per-slate 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data.synthetic_generator import (  # noqa: E402
    GeneratorConfig,
    SyntheticFintechDataGenerator,
    dataset_summary,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=Path("data/generated"))
    ap.add_argument("--num-users", type=int, default=3000)
    ap.add_argument("--num-products", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--slates-per-user", type=int, default=4)
    ap.add_argument("--slate-size", type=int, default=100)
    ap.add_argument("--ineligible-per-slate", type=int, default=10)
    ap.add_argument("--snapshot-at-days", type=float, default=365.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config = GeneratorConfig(
        num_users=args.num_users,
        num_products=args.num_products,
        seed=args.seed,
        slates_per_user=args.slates_per_user,
        slate_size=args.slate_size,
        ineligible_per_slate=args.ineligible_per_slate,
        snapshot_at_days=args.snapshot_at_days,
    )
    t0 = time.perf_counter()
    ds = SyntheticFintechDataGenerator(config).generate()
    t1 = time.perf_counter()
    save_dataset(ds, args.out, config)
    t2 = time.perf_counter()
    print(f"generated in {t1 - t0:.1f}s, saved to {args.out} in {t2 - t1:.1f}s")
    print("-" * 48)
    for k, v in dataset_summary(ds).items():
        fmt = f"{v:>14.4f}" if isinstance(v, float) and not v.is_integer() else f"{v:>14.0f}"
        print(f"{k:<32} {fmt}")


if __name__ == "__main__":
    main()
