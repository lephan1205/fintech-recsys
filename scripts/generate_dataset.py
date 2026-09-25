#!/usr/bin/env python
"""Generate a synthetic credit-marketplace dataset and write it to disk.

Example::

    python scripts/generate_dataset.py --out data/ --num-users 500 --num-products 2000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from recsys.data.synthetic_generator import (
    GeneratorConfig,
    SyntheticFintechDataGenerator,
    dataset_summary,
    save_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--num-users", type=int, default=2000)
    parser.add_argument("--num-products", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slates-per-user", type=int, default=3)
    args = parser.parse_args()

    config = GeneratorConfig(
        num_users=args.num_users,
        num_products=args.num_products,
        seed=args.seed,
        slates_per_user=args.slates_per_user,
    )
    t0 = time.perf_counter()
    ds = SyntheticFintechDataGenerator(config).generate()
    t1 = time.perf_counter()
    save_dataset(ds, args.out, config)
    t2 = time.perf_counter()

    print(f"generated in {t1 - t0:.1f}s, saved to {args.out} in {t2 - t1:.1f}s")
    print("-" * 48)
    for k, v in dataset_summary(ds).items():
        fmt = f"{v:>14.4f}" if v != int(v) else f"{v:>14.0f}"
        print(f"{k:<32} {fmt}")


if __name__ == "__main__":
    main()
