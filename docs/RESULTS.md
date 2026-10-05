# Results

Measured results for the recommender described in [ARCHITECTURE.md](ARCHITECTURE.md). Every number is copied from a table under `docs/results/`, produced by `scripts/evaluate.py`; `tests/test_docs.py` checks that each numeric table cell here exists in a results file.

## 1. Dataset

The benchmark is a synthetic dataset built on the schema of [ARCHITECTURE.md](ARCHITECTURE.md) (Section 3): 2 000 products, 3 000 members, 85 312 history events and 1 200 000 impression rows (3 000 members × 4 slates × 100 candidates), generated at a cut-off of 365 days with slates served in the 90 days before it. Of the applications in the histories, 0.0458 are still pending at the cut-off (`docs/results/dataset_summary.md`); in the impression slates 94 test-split applications are pending (`docs/results/funnel_metrics.md`). Members are split 2100 / 300 / 300 / 300.

**What the generator controls.** Credit attributes are drawn jointly (FICO and DTI have a measured correlation of −0.7438 on the benchmark), tiers follow FICO bands, products have family-specific gate floors so that eligibility varies by tier, and the click, apply and approval probabilities are explicit functions of product economics, gate margins and the member's affinity, with the approval probability exactly zero when any hard gate fails (`tests/test_schema_and_generator.py::test_approve_probability_hard_gates`). Decision delays are log-normal per family and outcome: cards resolve instantly with probability 0.8, personal loans in about two days, auto refinance about five, mortgages around 35 days when approved and 20 when declined. Impression slates deliberately include ineligible products (ten per hundred) so that the retrieval evaluation can measure how many positives the eligibility rules remove and so that the compliance gates have something to catch. Members carry held products, open applications, recent hard pulls and a financial state (revolving balance, other debt, an existing loan and its rate) that the benefit formulas read. What the generator does not model is listed in [ARCHITECTURE.md](ARCHITECTURE.md) (Section 10).

## 2. Training protocol

Each model has its own optimizer configuration (`training/config.py`), a warmup-then-cosine schedule (`training/optim.py::WarmupCosine`, `tests/test_training.py::test_warmup_cosine_schedule_peak_and_floor`), decoupled weight decay on weight matrices only (never on embeddings, biases or normalization gains; `test_build_optimizer_parameter_groups`), global-norm clipping, and early stopping with patience counted in evaluations on a per-model criterion with restoration of the best checkpoint (`training/early_stopping.py`, `test_early_stopping_triggers_after_patience_and_restores_best`). Calibrators are fitted afterwards on the calibration split, and the PRM is trained last on valued training slates. `tests/test_training.py::test_train_all_smoke_and_round_trip` runs the whole procedure on a tiny dataset and reloads the artifacts.

<!-- source: docs/results/training_summary.md -->
| model | steps run | step budget | selection criterion | best step | best value | early stop | seconds |
|---|---|---|---|---|---|---|---|
| rqvae | 2000 | 2000 | recon MSE s.t. min utilization >= 0.9 | 1900 | 0.0190 | no | 3.2733 |
| tiger | 800 | 3000 | val Recall@100 (checkpoint), val next-SID loss (stop) | 200 | 0.3105 | yes | 66.1745 |
| ranker | 700 | 2000 | val unified funnel loss (total) | 400 | 2.3747 | yes | 183.2982 |
| prm | 500 | 600 | val listwise loss | 350 | 2.3033 | yes | 0.9670 |

The RQ-VAE reaches utilization 1.0000 on all three levels. TIGER's best validation `Recall@100` is at its first evaluation and the validation loss rises afterwards, so it stops at step 800 of 3000; the ranker stops at step 700 of 2000 with its best total loss at step 400. Both stopped early on a rising validation loss, which is the overfitting signature of a small dataset rather than an optimization failure (the training losses keep falling in `artifacts/run.log`); the [decision register](DECISION_REGISTER.md) lists the dataset size and the step budgets as the parameters that would change this. The ranker has 312032 trainable parameters; the PRM trains on 8400 slates with at least one positive.

## 3. Funnel metrics

<!-- source: docs/results/funnel_metrics.md -->
| task | rows | positives | AUC | PR-AUC | GAUC | NCE | ECE |
|---|---|---|---|---|---|---|---|
| click | 120000 | 9162 | 0.5353 | 0.0852 | 0.5268 | 1.0098 | 0.0019 |
| apply \| click | 9162 | 1886 | 0.5754 | 0.2554 | 0.5354 | 0.9990 | 0.0117 |
| approve \| apply (resolved, weighted) | 1792 | 1460 | 0.7843 | 0.9338 | 0.7813 | 0.9419 | 0.0280 |

<!-- source: docs/results/funnel_metrics.md -->
| task | oracle AUC | oracle PR-AUC | oracle GAUC | oracle NCE |
|---|---|---|---|---|
| click | 0.6316 | 0.1294 | 0.6199 | 0.9697 |
| apply \| click | 0.6143 | 0.2826 | 0.5901 | 0.9733 |
| approve \| apply (resolved, weighted) | 0.9251 | 0.9811 | 0.9109 | 0.4800 |

The second table is the ceiling: it scores the generator's own probabilities, from which the labels were drawn, on the same rows. The click and apply tasks are intrinsically noisy on this generator (a perfect model reaches AUC 0.6316 and 0.6143), and the towers recover 0.5353 and 0.5754 of that; the approval task is much more learnable (ceiling 0.9251) and the tower reaches 0.7843. Normalized cross-entropy above 1 on click (1.0098) says the calibrated click tower is no better than the base rate in log-loss terms, which is consistent with a ranking signal that is weak but real (GAUC 0.5268 > 0.5). The apply and approve towers are evaluated on their conditional populations (clicked rows, resolved applications with the training weights). These are the numbers the design must be judged on, and they say the same thing as the retrieval table: on this benchmark the labels are noisy and the models are small and early-stopped, so the *machinery* (compliance, calibration, valuation, latency accounting) is the deliverable and the absolute metrics are a floor, not a ceiling.

Predicted against realized revenue on the test split: the model's expected revenue Σ P_funded · payout is $397,259 and realized partner revenue on resolved rows is $363,467 (`docs/results/funnel_metrics.md`). The realized figure is a lower bound because the 94 pending applications in the test split are excluded from it, so part of the gap is unresolved outcomes rather than over-prediction.

<!-- source: docs/results/positives_per_batch.md -->
| quantity | mean per batch |
|---|---|
| rows_before_downsampling | 6400.0000 |
| rows_after_downsampling | 1954.8500 |
| clicks | 490.0000 |
| applies | 102.3000 |
| resolved_approvals | 81.3500 |
| resolved_applications | 97.3000 |

## 4. Calibration

Calibrators are fitted on the calibration split and scored on the test split (`docs/results/calibration.md`); the served calibrators were isotonic on all three towers (weighted positives 8984.0, 1793.0 and 1435.0, all above the threshold). ECE is the bin-weighted mean gap between predicted probability and observed rate, MCE the worst bin's gap, and Brier the mean squared error of the probability.

<!-- source: docs/results/calibration.md -->
| tower | method | rows | ECE | MCE | Brier |
|---|---|---|---|---|---|
| click | uncalibrated (sigmoid) | 120000 | 0.0085 | 0.0221 | 0.0705 |
| click | isotonic | 120000 | 0.0019 | 0.1213 | 0.0705 |
| click | platt | 120000 | 0.0022 | 0.0174 | 0.0705 |
| apply | uncalibrated (sigmoid) | 9162 | 0.0545 | 0.0675 | 0.1644 |
| apply | isotonic | 9162 | 0.0117 | 0.4882 | 0.1616 |
| apply | platt | 9162 | 0.0085 | 0.0236 | 0.1615 |
| approve | uncalibrated (sigmoid) | 1792 | 0.0193 | 0.5000 | 0.1244 |
| approve | isotonic | 1792 | 0.0280 | 0.1714 | 0.1253 |
| approve | platt | 1792 | 0.0251 | 0.2018 | 0.1250 |

Isotonic cuts click ECE from 0.0085 to 0.0019 and apply ECE from 0.0545 to 0.0117, at the price of a large MCE (0.1213 and 0.4882): a step function's worst bin is a sparse bin. Platt is the better calibrator on the apply tower here (ECE 0.0085, MCE 0.0236) and is close on click. On the approve tower, scored on 1792 test rows, neither method beats the raw sigmoid on ECE (0.0193 raw against 0.0280 isotonic and 0.0251 platt), and the raw MCE of 0.5000 is a single empty-bin artefact; with 1435.0 weighted positives the tower is barely above the isotonic threshold and the calibration split is small. The per-tier table below shows the same pattern: on the approve tower Platt is better calibrated than isotonic in four of five tiers (for example 0.0178 against 0.0323 ECE on PRIME), while on click isotonic and Platt are within 0.004 of each other in every tier.

Sliced by credit tier (ECE / Brier on the test split):

<!-- source: docs/results/calibration.md -->
| tower | method | tier | rows | ECE | Brier |
|---|---|---|---|---|---|
| click | uncalibrated (sigmoid) | DEEP_SUBPRIME | 9200 | 0.0060 | 0.0607 |
| click | uncalibrated (sigmoid) | SUBPRIME | 11200 | 0.0054 | 0.0627 |
| click | uncalibrated (sigmoid) | NEAR_PRIME | 39200 | 0.0079 | 0.0673 |
| click | uncalibrated (sigmoid) | PRIME | 35200 | 0.0091 | 0.0734 |
| click | uncalibrated (sigmoid) | SUPER_PRIME | 25200 | 0.0107 | 0.0786 |
| click | isotonic | DEEP_SUBPRIME | 9200 | 0.0038 | 0.0607 |
| click | isotonic | SUBPRIME | 11200 | 0.0012 | 0.0627 |
| click | isotonic | NEAR_PRIME | 39200 | 0.0020 | 0.0672 |
| click | isotonic | PRIME | 35200 | 0.0017 | 0.0734 |
| click | isotonic | SUPER_PRIME | 25200 | 0.0018 | 0.0785 |
| click | platt | DEEP_SUBPRIME | 9200 | 0.0003 | 0.0607 |
| click | platt | SUBPRIME | 11200 | 0.0010 | 0.0627 |
| click | platt | NEAR_PRIME | 39200 | 0.0017 | 0.0672 |
| click | platt | PRIME | 35200 | 0.0030 | 0.0734 |
| click | platt | SUPER_PRIME | 25200 | 0.0041 | 0.0785 |
| apply | uncalibrated (sigmoid) | DEEP_SUBPRIME | 597 | 0.0610 | 0.1288 |
| apply | uncalibrated (sigmoid) | SUBPRIME | 752 | 0.1033 | 0.1296 |
| apply | uncalibrated (sigmoid) | NEAR_PRIME | 2841 | 0.0483 | 0.1575 |
| apply | uncalibrated (sigmoid) | PRIME | 2806 | 0.0603 | 0.1669 |
| apply | uncalibrated (sigmoid) | SUPER_PRIME | 2166 | 0.0388 | 0.1923 |
| apply | isotonic | DEEP_SUBPRIME | 597 | 0.0185 | 0.1249 |
| apply | isotonic | SUBPRIME | 752 | 0.0315 | 0.1205 |
| apply | isotonic | NEAR_PRIME | 2841 | 0.0222 | 0.1557 |
| apply | isotonic | PRIME | 2806 | 0.0201 | 0.1637 |
| apply | isotonic | SUPER_PRIME | 2166 | 0.0158 | 0.1908 |
| apply | platt | DEEP_SUBPRIME | 597 | 0.0130 | 0.1252 |
| apply | platt | SUBPRIME | 752 | 0.0274 | 0.1209 |
| apply | platt | NEAR_PRIME | 2841 | 0.0213 | 0.1557 |
| apply | platt | PRIME | 2806 | 0.0129 | 0.1635 |
| apply | platt | SUPER_PRIME | 2166 | 0.0108 | 0.1907 |
| approve | uncalibrated (sigmoid) | DEEP_SUBPRIME | 87 | 0.0979 | 0.0767 |
| approve | uncalibrated (sigmoid) | SUBPRIME | 105 | 0.0998 | 0.1384 |
| approve | uncalibrated (sigmoid) | NEAR_PRIME | 529 | 0.0475 | 0.1445 |
| approve | uncalibrated (sigmoid) | PRIME | 551 | 0.0352 | 0.1236 |
| approve | uncalibrated (sigmoid) | SUPER_PRIME | 520 | 0.0355 | 0.1102 |
| approve | isotonic | DEEP_SUBPRIME | 87 | 0.0532 | 0.0738 |
| approve | isotonic | SUBPRIME | 105 | 0.0974 | 0.1447 |
| approve | isotonic | NEAR_PRIME | 529 | 0.0450 | 0.1473 |
| approve | isotonic | PRIME | 551 | 0.0323 | 0.1231 |
| approve | isotonic | SUPER_PRIME | 520 | 0.0386 | 0.1098 |
| approve | platt | DEEP_SUBPRIME | 87 | 0.0707 | 0.0788 |
| approve | platt | SUBPRIME | 105 | 0.0893 | 0.1390 |
| approve | platt | NEAR_PRIME | 529 | 0.0384 | 0.1459 |
| approve | platt | PRIME | 551 | 0.0178 | 0.1231 |
| approve | platt | SUPER_PRIME | 520 | 0.0316 | 0.1106 |

Calibration is not uniform across tiers. On the approval tower the two thin tiers (87 and 105 rows) carry ECE around 0.05–0.10 under every method, while the three populous tiers sit at 0.02–0.05; this is a sample-size effect, not a fairness finding, but it is exactly the slice a fairness audit would start from, and it is why `HarmRate@10` is also reported by tier.

## 5. Pareto frontier

The overall `α` sweep is shown in [ARCHITECTURE.md](ARCHITECTURE.md) (Section 7.3; `valuation/pareto.py::pareto_sweep`, `docs/results/pareto_sweep.md`). Up to `α = 0.7`, revenue rises from $34.63 to $37.81 while member benefit falls by less than four dollars and the harm rate stays at 0.000. Beyond that the trade-off turns sharply: at `α = 1` revenue is $47.41 but benefit collapses to $308.46 and 16% of shown products have `NB < 0` (harm rate 0.160). The default `α = 0.5` gives $35.97 and $437.31 with harm 0.000.

By tier at the two endpoints and the default:

<!-- source: docs/results/pareto_sweep.md -->
| α | tier | Revenue@10 | UserBenefit@10 | HarmRate@10 |
|---|---|---:|---:|---:|
| 0.0 | DEEP_SUBPRIME | 13.91 | 86.03 | 0.000 |
| 0.0 | SUBPRIME | 17.67 | 150.64 | 0.000 |
| 0.0 | NEAR_PRIME | 25.88 | 290.84 | 0.000 |
| 0.0 | PRIME | 40.38 | 560.28 | 0.000 |
| 0.0 | SUPER_PRIME | 55.29 | 752.33 | 0.000 |
| 0.5 | DEEP_SUBPRIME | 14.92 | 85.59 | 0.000 |
| 0.5 | SUBPRIME | 18.63 | 150.20 | 0.000 |
| 0.5 | NEAR_PRIME | 26.86 | 290.37 | 0.000 |
| 0.5 | PRIME | 42.08 | 559.31 | 0.000 |
| 0.5 | SUPER_PRIME | 57.03 | 751.49 | 0.000 |
| 1.0 | DEEP_SUBPRIME | 21.50 | 41.87 | 0.070 |
| 1.0 | SUBPRIME | 26.59 | 81.06 | 0.130 |
| 1.0 | NEAR_PRIME | 35.97 | 197.92 | 0.193 |
| 1.0 | PRIME | 54.09 | 425.03 | 0.143 |
| 1.0 | SUPER_PRIME | 74.59 | 515.97 | 0.176 |

Benefit scales with tier because prime members have larger balances to refinance and qualify for cheaper products; the `α = 1` policy halves the benefit of deep-subprime members (86.03 → 41.87) for less than eight dollars of extra revenue per slate, which is the concrete case for keeping `α` in the flat region. The harm at `α = 1` is spread across tiers (0.070 deep-subprime to 0.193 near-prime) rather than concentrated on the vulnerable end, which is the pattern one expects when payout, not creditworthiness, drives the ranking.

## 6. Retrieval quality

Retrieval is measured on the test split with the member's next item as the positive. TIGER is compared with two baselines that use no history. *Eligible-random* picks 100 of the member's eligible products uniformly at random; it is the floor, the recall that eligibility alone buys. *Eligible-popularity* ranks the member's eligible products by how often each appeared in a positive event in the training split and keeps the top 100; it is the bar a sequence model must clear to justify its cost. Both use the same eligibility mask and fill the same number of slots as the beam, so the difference in recall is what TIGER learns from the history rather than the effect of the compliance filter. Test positives the member was not eligible for are left out, since no compliant retriever could return them.

<!-- source: docs/results/retrieval_metrics.md -->
| retriever | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|
| TIGER (beam) | 0.0868 | 0.2283 | 0.3379 |
| eligible-popularity | 0.0548 | 0.2877 | 0.3836 |
| eligible-random | 0.0274 | 0.0959 | 0.1598 |

Users counted: 219; positives excluded as ineligible: 70; beam yield 0.9807.

<!-- source: docs/results/retrieval_metrics.md -->
| slice | users | Recall@100 |
|---|---|---|
| DEEP_SUBPRIME | 8 | 0.8750 |
| SUBPRIME | 14 | 0.7857 |
| NEAR_PRIME | 72 | 0.4167 |
| PRIME | 73 | 0.2740 |
| SUPER_PRIME | 52 | 0.1154 |
| target CREDIT_CARD | 52 | 0.1154 |
| target BALANCE_TRANSFER_CARD | 47 | 0.4468 |
| target PERSONAL_LOAN | 41 | 0.3171 |
| target AUTO_REFINANCE | 48 | 0.4583 |
| target MORTGAGE | 31 | 0.3871 |

**Reading the numbers.** TIGER beats both baselines at `Recall@10` and beats random everywhere, but eligible-popularity is ahead at `Recall@50` and `Recall@100`. Three things explain this. First, eligibility makes popularity a strong baseline: the generator's next-item process has a strong popularity component, and once eligibility has removed most of the catalog, the most popular remaining products are a good guess. The tier slice shows the effect at its extreme: deep-subprime members are eligible for only a few dozen products, so any eligible list reaches recall 0.8750. Second, the beam is a generative top-100: it commits to a few prefixes at level one and spends its width on their continuations, so it is more precise at the head and covers less of the tail than a scored list of the same length. This is the expected shape of a generative retriever and the reason it is followed by a scorer rather than used alone. The tail loss is worst for credit cards (0.1154), the largest and most popular family. Third, the retriever is under-trained rather than mis-designed: its best validation `Recall@100` (0.3105) came at the first evaluation and had begun to drift by step 400 (Section 2), and the [decision register](DECISION_REGISTER.md) lists the training budget as the parameter that would change this.

On the served slates, with click, apply and approve as the positive, recall at 100 falls down the funnel, as expected:

<!-- source: docs/results/retrieval_metrics.md -->
| positive definition | users | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|
| click | 300 | 0.0349 | 0.1457 | 0.2532 |
| apply | 293 | 0.0313 | 0.1372 | 0.2376 |
| approve | 292 | 0.0307 | 0.1286 | 0.2211 |

## 7. Serving latency

Measured over 50 test members after one warm-up call (`serving/evaluation.py::latency_markdown`):

<!-- source: docs/results/latency.md -->
| stage | p50 ms | p99 ms | max ms |
|---|---|---|---|
| eligibility | 0.0236 | 0.0288 | 0.0289 |
| retrieval | 16.5931 | 17.2971 | 17.4457 |
| post_filter | 0.0622 | 0.0717 | 0.0718 |
| ranking | 2.3920 | 2.4681 | 2.4708 |
| calibration | 0.0558 | 0.0672 | 0.0698 |
| valuation | 0.2681 | 0.2997 | 0.3140 |
| rerank | 0.4297 | 0.4668 | 0.4688 |
| assert | 0.0322 | 0.0396 | 0.0409 |
| total | 19.8375 | 20.5171 | 20.7005 |

The total p50 of 19.84 ms is above the 10 ms budget, and the table says exactly why: retrieval is 16.5931 ms and everything else together is under 3.3 ms. The beam re-runs the full history sequence at every one of the four levels for 100 beams, so the decoder processes `4 × 100` sequences of about 80 tokens per request; the compliance mask itself is not the cost (the eligibility stage is 0.0236 ms and the vectorized trie query is inside the retrieval figure but is a gather over `(P, 32)`). Three mitigations are available in order of engineering cost: a KV cache so that only the appended code tokens are recomputed per level (the largest win, expected to bring retrieval near the ranking stage's cost), a smaller beam for members whose eligible set is small (the beam yield already reports how many slots were fillable), and batching requests. None changes the design; all are recorded as the retrieval failure mode in the appendix of [ARCHITECTURE.md](ARCHITECTURE.md). The scorer at 2.3920 ms for 100 candidates, the valuation at 0.2681 ms and the PRM at 0.4297 ms are within budget with room to spare, which supports the cascade argument of [ARCHITECTURE.md](ARCHITECTURE.md) (Section 2): the expensive component is the one that must run before candidate count is known.

## 8. The test suite as evidence

| test file | invariant it proves |
|---|---|
| `tests/test_schema_and_generator.py` | schema contracts; `item_id 0` only for score changes; funnel monotonicity `y_click ≥ y_apply ≥ y_approve`; hard-gate approval probability exactly zero; generator determinism; pending events and pending context; format-v2 round trip |
| `tests/test_collator.py` | left padding puts the latest event last; masks come from actions; padding-side invariance; the observed view at a young cut-off never leaks the oracle label; pending-family mask |
| `tests/test_delayed_feedback.py` | observed status vs oracle; delay CDF and sampler match the law; policy weights; the `drop < ipw` / `negative` bias ordering on a mortgage slice |
| `tests/test_tiger.py` | eligibility engine equals the scalar reference gate by gate; trie mask equals brute force; unique Semantic IDs; every beam is eligible and unique; held and pending items are never generated even when top-scored; recall metric and baselines respect the mask |
| `tests/test_hstu.py` | M-FALCON mask structure; causality under left padding; no PAD leakage; zero gradient from future to past; candidate independence under permutation and removal; batched equals per-candidate; right-padding parity |
| `tests/test_funnel_loss.py` | `log1mexp` matches the reference and is finite at extremes; mask-sum normalization; log-space equals naïve product at moderate logits; finite loss and gradients at `|z| = 30`; pending rows give zero gradient to `z₃`; entire-space terms supervise `z₂` on unclicked rows; down-sampling keeps all positives and the logit correction recovers the true probability |
| `tests/test_calibration_and_valuation.py` | ECE / MCE / Brier on known cases; weighted metrics equal duplication; weighted PAVA and Platt recover known solutions; fallback and auto selection; `p₃` never fitted on unobserved rows; per-family benefit formulas by hand; utility endpoints; guardrails; weakly monotone `α` sweep; slate metrics; PRM features and rerank |
| `tests/test_overfit.py` | every model drives its loss toward zero on a fixed batch: RQ-VAE, TIGER (full recall on memorized examples), HSTU + PLE under the unified loss, ZILN, PRM |
| `tests/test_training.py` | optimizer parameter groups; codebooks are not parameters; schedule peak and floor; early stopping restores the best checkpoint; user split is disjoint and complete; the register equals the config defaults; `train_all` smoke and round trip |
| `tests/test_pipeline.py` | served slates are eligible, unique, guarded and at most ten; empty eligibility is handled; the output assertion catches an injected violation; artifacts round-trip to identical slates |
| `tests/test_docs.py` | the write-up's tables come from the results files, the decision register equals the config defaults, the diagrams pass their checklist |
