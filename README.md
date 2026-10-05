# fintech-recsys

> **Personal project** by [Le Phan](https://github.com/lephan1205), built independently in PyTorch.
> **All data is synthetic**: it comes from the generator in `src/recsys/data/`. No real member,
> product or company data is used, and the project is not affiliated with any employer or
> credit marketplace.

A four-stage credit-marketplace recommender (credit cards, balance-transfer cards, personal
loans, auto refinance, mortgages) with underwriting eligibility enforced three times: a
prefix-trie logit mask at generation, a post-retrieval gate, and an output assertion. Every learned probability is a proper-scoring-rule estimate that is calibrated before 
it is turned into dollars; business policy (guardrails, product family diversity) lives at valuation and
re-ranking time.

1. **TIGER generative retrieval** over RQ-VAE Semantic IDs, beam width 100, with the trie
   masking every code that cannot end in an eligible product.
2. **HSTU scoring backbone** that scores the 100 candidates in one batched `B × (L + K)`
   pass with an attention mask that keeps candidates independent of each other.
3. **PLE multi-task funnel** (`p(click)`, `p(apply | click)`, `p(approve | apply)`, ZILN
   amount) trained with a unified funnel loss whose entire-space terms are written in log
   space on the same towers; pending applications are a labeling concern (`pending_policy`).
4. **Calibration → valuation → guardrails → PRM**: isotonic / Platt per tower, expected value
   and a deterministic net user benefit combined under a serving-time `α`, suitability
   guardrails, and a Pre-LN transformer re-ranker that outputs an ordering only.

For a detailed discussion of the design, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**. Full measured results are in [docs/RESULTS.md](docs/RESULTS.md) and every tuned value with its rationale in [docs/DECISION_REGISTER.md](docs/DECISION_REGISTER.md).

## Design highlights

This project shows how to design an end-to-end recommender for a regulated marketplace. The main deliverable is the design reasoning: [ARCHITECTURE.md](docs/ARCHITECTURE.md) explains where each component sits and why, and the [decision register](docs/DECISION_REGISTER.md) records every tuned value with its rationale.

- **Compliance by construction, not by filtering.** Eligibility is a hard constraint at three points: a prefix-trie logit mask means the retriever can't *generate* an ineligible product, a post-retrieval gate re-checks the candidates, and an assertion checks the final slate ([§4.2](docs/ARCHITECTURE.md#42-the-prefix-trie-as-a-compliance-mechanism)).
- **A cascade sized to the serving budget.** Generative retrieval narrows the catalog to 100 candidates; HSTU then scores all 100 in one batched pass, with an attention mask that keeps candidates independent ([§5.2](docs/ARCHITECTURE.md#52-m-falcon-scoring-all-candidates-in-one-pass)).
- **The conversion funnel modeled as one multi-task problem.** PLE towers predict click → apply → approve (plus a ZILN loan-amount head) under one funnel loss. The loss trains on every impression, not just clicked ones, to avoid sample-selection bias, and treats delayed or pending approvals as a labeling policy ([§6.3](docs/ARCHITECTURE.md#63-the-unified-funnel-loss)).
- **Calibrate first, then value.** Probabilities are calibrated per tower before they're converted into expected revenue and net member benefit ([§7.1](docs/ARCHITECTURE.md#71-calibration-before-valuation-never-after-prm), [§7.2](docs/ARCHITECTURE.md#72-valuation-what-an-offer-is-worth-in-dollars)).
- **Business policy lives at serving time, not in the loss.** The revenue vs. member-benefit weight `α`, suitability guardrails and product-family diversity are all applied after training, so changing policy never requires a retrain. A transformer re-ranker (PRM) only reorders the final slate.
- **Failure modes stated up front.** The [appendix](docs/ARCHITECTURE.md#appendix--failure-modes-and-what-catches-them) maps each failure mode to the mechanism or test that catches it, and the 127 tests check the invariants (eligibility, causality, calibration, loss stability).

![Serving pipeline](docs/diagrams/serving_pipeline.svg)

![Training pipeline](docs/diagrams/training_pipeline.svg)

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
# 3. Evaluate: writes docs/results/*.md (retrieval, funnel, calibration, ablation, Pareto, latency)
.venv/bin/python scripts/evaluate.py --data data/generated --artifacts artifacts
# 4. Serve a few users end to end with per-stage latency
.venv/bin/python scripts/run_e2e_pipeline.py --data data/generated --artifacts artifacts --serve-users 5
# Quality gate
.venv/bin/python -m pytest -q && .venv/bin/python -m mypy && .venv/bin/python -m ruff check . && .venv/bin/python scripts/check_diagrams.py
```

## Results on the synthetic benchmark

These numbers show that the pipeline runs end to end and is measured honestly. They are not the point of the project: the synthetic labels are noisy by design (see the oracle ceilings) and the models are small. [docs/RESULTS.md](docs/RESULTS.md) has the full analysis.

Synthetic benchmark: 2 000 products, 3 000 members, 1.2 M impression rows, members split
2100 / 300 / 300 / 300 (train / val / calib / test). Sources: `docs/results/*.md`.

<!-- source: docs/results/retrieval_metrics.md, funnel_metrics.md, pareto_sweep.md, latency.md -->
| metric | value | baseline / ceiling |
|---|---|---|
| Retrieval `Recall@100` (TIGER beam, eligible next-item positives) | 0.3379 | eligible-popularity 0.3836, eligible-random 0.1598 |
| Retrieval `Recall@10` | 0.0868 | eligible-popularity 0.0548 |
| Click AUC (calibrated) / ECE | 0.5353 / 0.0019 | oracle ceiling 0.6316 |
| Apply-given-click AUC / ECE | 0.5754 / 0.0117 | oracle ceiling 0.6143 |
| Approve-given-apply AUC / ECE | 0.7843 / 0.0280 | oracle ceiling 0.9251 |
| Pareto at `α = 0.5`: revenue / user benefit / harm per slate ($) | 35.97 / 437.31 / 0.000 | `α = 1`: 47.41 / 308.46 / 0.160 |
| Pending-policy bias on the mortgage slice (`drop` / `ipw` / `negative`) | 0.0659 / 0.1721 / -0.5909 | truth 0.6255 |
| Serving latency p50, CPU, `B = 1` | 19.84 ms | target 10 ms; retrieval alone 16.5931 ms |
| Eligibility violations in served slates | 0 | asserted by the pipeline |

The latency target is missed by the TIGER beam (the decoder re-runs the history at each of
four levels for 100 beams); the write-up analyzes the gap and the KV-cache mitigation rather
than hiding it.

## Layout

```
src/recsys/data        schema (single source of truth), generator, collators, delayed feedback
src/recsys/layers      RQ-VAE, prefix trie, HSTU, transformer blocks
src/recsys/models      tiger/, hstu/, ple/, prm/, ranker.py
src/recsys/losses      unified funnel loss, ZILN, listwise, stable log-space helpers
src/recsys/serving     eligibility engine, calibration, valuation stage, pipeline, evaluation
src/recsys/valuation   expected value, net user benefit, utility + guardrails, Pareto sweep
src/recsys/training    configs + decision register, optimizers, splits, trainers
scripts/               generate_dataset, train_all, evaluate, pareto_sweep, run_e2e_pipeline, check_diagrams
docs/                  ARCHITECTURE.md, RESULTS.md, DECISION_REGISTER.md, diagrams/, results/
tests/                 127 tests; tests/test_docs.py ties the write-up to the code and results
```
