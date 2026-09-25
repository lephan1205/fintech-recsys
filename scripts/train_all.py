"""Train every stage on a generated dataset and save the serving artifacts.

Usage::

    python scripts/train_all.py --data data/generated --out artifacts [--quick]

``--quick`` uses the tiny configuration (smoke test).  Training histories are written
to ``<out>/histories.json`` and the configuration to ``<out>/meta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys.data.synthetic_generator import load_dataset  # noqa: E402
from recsys.training.config import TrainingConfig  # noqa: E402
from recsys.training.trainers import train_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data", type=Path, default=Path("data/generated"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--quick", action="store_true", help="tiny config for a smoke run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.data)
    cfg = TrainingConfig.tiny() if args.quick else TrainingConfig()
    t0 = time.perf_counter()
    result = train_all(ds, cfg, log=print)
    result.artifacts.save(args.out)
    (args.out / "histories.json").write_text(
        json.dumps(result.histories_as_dict(), indent=1), encoding="utf-8"
    )
    (args.out / "splits.json").write_text(json.dumps(result.splits.sizes()), encoding="utf-8")
    print(f"trained and saved to {args.out} in {time.perf_counter() - t0:.0f}s")
    for name, h in result.histories.items():
        print(
            f"{name:<8} best step {h.best_step:>5}  best {h.best_value:.4f}  "
            f"early stop {h.stopped_early}  {h.seconds:.0f}s"
        )
    print("calibrators:", result.artifacts.calibrators.reports)


if __name__ == "__main__":
    main()
