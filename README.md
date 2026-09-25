# fintech-recsys

A four-stage credit-marketplace recommender (credit cards, balance-transfer cards, personal
loans, auto refinance, mortgages) built for hard underwriting compliance, honest
probabilities and a sub-10 ms CPU scoring budget:

1. **TIGER generative retrieval** over RQ-VAE semantic IDs with a prefix trie that masks
   ineligible products at decoding time.
2. **HSTU scoring backbone** that scores 100 candidates in one batched `(L + K)` pass.
3. **PLE multi-task funnel** (`p(click)`, `p(apply | click)`, `p(approve | apply)`, ZILN
   amount) trained with a unified funnel loss that corrects sample-selection bias in log
   space and handles pending (delayed) approval decisions as a labeling concern.
4. **Calibration, valuation and PRM re-ranking**: isotonic / Platt per tower, expected
   value plus a deterministic net-user-benefit, a serving-time trade-off weight `α`,
   suitability guardrails, and a Pre-LN transformer re-ranker with a family
   cannibalization penalty.

Diagrams and the full design-review write-up: `docs/ARCHITECTURE.md` (written after the
measured results exist; see `docs/results/`).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Generate a format-v2 dataset (delayed feedback, extended economics)
.venv/bin/python scripts/generate_dataset.py --out data/generated
# 2. Train every stage and save the serving artifacts
.venv/bin/python scripts/train_all.py --data data/generated --out artifacts
# 3. Evaluate: writes docs/results/*.md (retrieval, funnel, calibration, ablation, latency)
.venv/bin/python scripts/evaluate.py --data data/generated --artifacts artifacts
# 4. Serve a few users end to end with per-stage latency
.venv/bin/python scripts/run_e2e_pipeline.py --data data/generated --artifacts artifacts
# Quality gate
.venv/bin/python -m pytest -q && .venv/bin/python -m mypy && .venv/bin/python -m ruff check .
```

## Results

Filled in from `docs/results/` once the training run has completed.
