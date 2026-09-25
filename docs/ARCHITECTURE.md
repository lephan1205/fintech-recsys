# A four-stage, compliance-constrained recommender for a credit marketplace

*Generative retrieval → HSTU scoring → multi-task funnel → calibrated valuation and slate re-ranking, with underwriting eligibility enforced three times and honest probabilities as the audited output.*

Every number in this document is copied from a table under `docs/results/` (produced by `scripts/evaluate.py`), and every design claim names the file and symbol that implements it. `tests/test_docs.py` asserts that each numeric cell in the tables below exists in a results file, that Appendix A equals the config defaults, and that the two diagrams pass `scripts/check_diagrams.py`.

![Serving pipeline](diagrams/serving_pipeline.svg)

![Training pipeline](diagrams/training_pipeline.svg)

---

## 1. Abstract

A credit marketplace recommends financial products (credit cards, balance-transfer cards, personal loans, auto refinance, mortgages) to members whose eligibility is decided by hard underwriting rules, whose funnel (click → apply → approve) is steeply imbalanced, and whose most valuable outcomes arrive days to weeks after the impression. The recommender described here is a four-stage cascade. Stage 1 is generative retrieval: an RQ-VAE turns the 18-dimensional product record into a hierarchical Semantic ID, a TIGER-style decoder generates the IDs of the top-100 candidates by beam search, and a prefix trie masks every code whose continuation cannot end in an eligible product, so ineligible items are never generated. Stage 2 scores the 100 candidates in a single batched pass of an HSTU backbone whose attention mask keeps every candidate independent of the others. Stage 3 is a PLE multi-task funnel producing raw logits for `p(click)`, `p(apply | click)`, `p(approve | apply)` and a zero-inflated log-normal amount, trained with a unified funnel loss whose entire-space terms are written in log space on the same towers, with pending applications handled as a labeling concern. Stage 4 calibrates the logits per tower, converts them to expected partner value and a deterministic net user benefit, combines both under a serving-time weight `α` with suitability guardrails, and re-ranks the survivors with a small transformer whose output is only an ordering.

Headline results on the synthetic benchmark (`docs/results/`): TIGER reaches `Recall@100` of 0.3379 against 0.3836 for an eligible-popularity baseline and 0.1598 for eligible-random, while beating popularity at `Recall@10` (0.0868 vs 0.0548); the calibrated towers reach AUC 0.5353 / 0.5754 / 0.7843 for click / apply / approve against an oracle ceiling of 0.6316 / 0.6143 / 0.9251, with expected calibration error 0.0019 / 0.0117 / 0.0280; the `α` sweep traces a clean Pareto frontier from $34.63 expected revenue and $438.00 user benefit per slate at `α = 0` to $47.41 and $308.46 at `α = 1`, with `HarmRate@10` staying at 0.000 through `α = 0.7`; the serving path costs 19.84 ms p50 on one CPU thread group, of which 16.5931 ms is the TIGER beam, so the design misses its 10 ms target and the gap is analyzed rather than hidden. Zero eligibility violations were observed in any served slate, and the tests that guarantee this are named in each section.

## 2. Problem setting

**The marketplace.** A member sees a slate of up to ten products. Clicking opens an offer; applying triggers a hard credit pull and a partner decision; an approval funds a loan or opens a line and pays the platform a partner payout. The platform therefore has two stakeholders whose interests only partly overlap: partners pay for approvals, members want the product that improves their finances. A recommender that maximizes payout alone will push high-payout, high-APR products at members who would be better off with a cheaper one, which is both a trust problem and, for a regulated lender, a suitability problem.

**Hard underwriting.** Every product declares a minimum FICO, a maximum debt-to-income ratio, a minimum income and a set of licensed states. A member outside these bounds is not "unlikely to be approved"; they are ineligible, and showing them the product is a compliance defect, not a ranking error. The same holds for products the member already holds and products with an open application. This is why eligibility is enforced as a hard mask at three points of the pipeline rather than learned as a feature (Section 4.2 and Section 8).

**Delayed, censored outcomes.** Card decisions are mostly instant; personal loans take one to three days; auto refinance three to seven; mortgages weeks. At any training cut-off a fraction of applications is still pending, and the fraction is largest exactly for the family with the largest amounts. Treating a pending application as a decline biases the approval tower downward for mortgages; dropping the row removes signal; inverse-propensity weighting needs the delay law. Section 6.4 measures all three.

**Funnel imbalance.** On the benchmark dataset the click-through rate is 0.0762, the apply rate given a click is 0.2016 and the approval rate given an application is 0.8232, so approvals occur on 0.0126 of impressions (`docs/results/dataset_summary.md`). A batch of impressions carries hundreds of clicks, tens of applications and a few dozen resolved approvals; the loss design must keep every term well defined at those counts (Section 6.3).

**Latency budget.** The target is a sub-10 ms CPU scoring budget per member at batch size one. That budget is why the design is a cascade: no scorer that reads the full interaction history can score a 2 000-item catalog at ranking fidelity in that time, and no retriever that runs in that time can score candidates against the full history. Section 5.4 makes the argument quantitatively with the measured per-stage latencies.

**Why a cascade at all.** The four stages are not four models bolted together; each stage exists because a different constraint binds there. Retrieval is where compliance is cheapest to enforce (a mask on a logit vector) and where catalog size is the cost driver. Scoring is where the history matters and where candidate count is the cost driver. The funnel stage is where the labels live and where imbalance and censoring must be handled. Valuation is where business policy lives, and it is kept out of every learned component so that a policy change is a config change, not a retraining. A cascade lets each of these concerns be tested in isolation: `tests/test_tiger.py` proves compliance, `tests/test_hstu.py` proves candidate independence, `tests/test_funnel_loss.py` proves the loss is finite and unbiased where it claims to be, and `tests/test_calibration_and_valuation.py` proves the dollar arithmetic.

## 3. Data contract

**Schema as the single source of truth.** `src/recsys/data/schema.py` defines the pydantic records every stage consumes: `FinancialProduct` (gates, economics, family), `UserProfile` (credit attributes, state, held products, pending products and families, the financial state used by the benefit formulas), `InteractionRecord` (one timeline per user) and `ImpressionSlate` (served candidates with the funnel labels, the generator's true probabilities, the serve time and the partner's decision delay). `SyntheticDataset` bundles them with dense `catalog_features (N+1, 18)` and `user_features (U, 26)` matrices. The generator, the collators, the models and the evaluation all import these types; nothing re-derives a feature from raw fields.

**The reserved-index rule.** `item_id 0` is reserved: it is the padding index of every embedding table, and it is the item id of `SCORE_CHANGE` events, which are member-level (a credit score moved) rather than product-level. Consequently masks are derived from actions, never from items: `attention_mask = action_ids != PAD` in `data/collator.py::SequenceCollator`. A mask derived from `item_ids` would silently drop every score-change event, and `tests/test_collator.py::test_mask_derived_from_action_not_item` and `test_score_change_embedding_not_zero_but_pad_is` pin both halves of the rule. Row 0 of the catalog matrix is zero and is never eligible (`serving/eligibility_engine.py`), so a model can never generate or score the padding item.

**Left padding and the user state.** Sequences are left-padded by default. With left padding the most recent real event is always at position `L-1`, so every sequential model reads its user state as `hidden[:, -1, :]` with no per-row gather, and causal recency indexing is aligned across the batch (`data/collator.py`, `layers/transformer_blocks.py::gather_last_real` for the right-padded alternative). `tests/test_collator.py::test_seq_rep_invariant_to_padding_side` shows the two conventions give the same representation, which is what lets HSTU place its candidate tokens at "now" without knowing where each row's history starts (Section 5.2).

**Oracle versus observed labels.** The generator draws every funnel label from a true probability and, for applications, draws a decision delay from a per-family, per-outcome law (`data/schema.py::DelayConfig`: cards resolve instantly with probability 0.8, personal loans in about two days, auto refinance about five, mortgages around 35 days when approved and 20 when declined, all log-normal). The slate stores the *oracle* approval and the delay. Everything the models see is the *observed* view at the training cut-off `snapshot_at_days`: `data/delayed_feedback.py::observed_status` maps `(y_apply, served_at, delay, snapshot, y_approve)` to `NOT_APPLIED / APPROVED / DECLINED / PENDING`, and `data/impression_collator.py::ImpressionCollator` zeroes the approval label and amount on pending rows while keeping the oracle values in separate fields that only the evaluation reads. `tests/test_collator.py::test_impression_collator_observed_view_at_young_snapshot` asserts the oracle label of a pending row never leaks into the training tensors. Histories carry the same distinction through the `APPLY_PENDING` action, so a member's sequence at the cut-off contains a pending application as an event without its outcome.

**What the generator controls.** Credit attributes are drawn jointly (FICO and DTI have a measured correlation of −0.7438 on the benchmark), tiers follow FICO bands, products have family-specific gate floors so that eligibility varies by tier, and the click, apply and approval probabilities are explicit functions of product economics, gate margins and the member's affinity, with the approval probability exactly zero when any hard gate fails (`tests/test_schema_and_generator.py::test_approve_probability_hard_gates`). Impression slates deliberately include ineligible products (ten per hundred) so that the retrieval evaluation can measure how many positives the eligibility rules remove and so that the compliance gates have something to catch. Members carry held products, open applications, recent hard pulls and a financial state (revolving balance, other debt, an existing loan and its rate) that the benefit formulas read.

**What it does not model.** There is no position bias, no seasonality, no competition between concurrent slates, no partner-side policy drift, and the delay laws are stationary and known to the `ipw` weighting. Section 10 returns to each of these.

**Benchmark size.** Two thousand products and three thousand members, split by member into 2100 / 300 / 300 / 300 for train / validation / calibration / test (`docs/results/dataset_summary.md`). Members, not events, are split so that no member's behaviour appears on both sides of any boundary; the calibration split is used only to fit calibrators and is therefore disjoint from everything the models were trained or selected on (`training/splits.py::split_by_user`, `tests/test_training.py::test_split_by_user_is_disjoint_and_covers_everyone`).

<!-- source: docs/results/dataset_summary.md -->
| statistic | value |
|---|---|
| num_products | 2000 |
| num_users | 3000 |
| num_history_events | 85312 |
| impressions | 1200000 |
| ctr | 0.0762 |
| apply_rate_given_click | 0.2016 |
| approval_rate_given_apply | 0.8232 |
| approvals_per_impression | 0.0126 |
| pending_rate_given_apply | 0.0458 |
| corr_fico_dti | -0.7438 |
| products (credit card / balance transfer / personal loan / auto refinance / mortgage) | 789 / 305 / 382 / 211 / 313 |
| users train / val / calib / test | 2100 / 300 / 300 / 300 |

## 4. Stage 1 — Generative retrieval under a compliance mask

### 4.1 Semantic IDs from an RQ-VAE

**Choice.** Each product's standardized 18-dimensional feature vector (gate thresholds, APR, rewards, fees, family one-hot, economics) is encoded by `layers/rq_vae.py::RQVAE` into a 16-dimensional latent and residually quantized at three levels with a codebook of 32 codes per level; a fourth disambiguation level makes every ID unique (`RQVAE.assign_semantic_ids`, `tests/test_tiger.py::test_semantic_ids_unique_after_disambiguation`). The result is a coarse-to-fine tuple in which products with similar underwriting and economics share prefixes.

**Alternatives.** Random item IDs with a softmax over the catalog (SASRec + MIPS) and a two-tower model with approximate nearest neighbours were both considered. Both make the retriever's output an unconstrained set of item ids that must be filtered afterwards; neither gives a structure on which a per-request eligibility constraint can be applied *during* generation. Hashing-based IDs share the problem. Discrete hierarchical IDs are chosen because they are the only representation on which the prefix-trie mask of Section 4.2 is meaningful.

**Why depth 3 and codebook 32.** The code space has `32³ = 32 768` prefixes for a catalog of 2 000 items, so collisions are rare before disambiguation and the fourth level is short; a shallower tree would put many items under one prefix and weaken the trie's ability to say "no eligible leaf here", a deeper one would lengthen the beam (each level is one forward pass over `B·W` rows; Section 8). The utilization criterion in `training/trainers.py::train_rqvae` refuses any checkpoint whose minimum per-level utilization is below 0.9; on the benchmark all three levels reach utilization 1.0000 (`docs/results/training_summary.md`), so the hierarchy is fully used. The register (Appendix A) records the sensitivity argument.

**Training details that matter.** Codebooks are not optimizer parameters. They are buffers updated by exponential moving averages (decay 0.99) of the residuals assigned to each code, with dead-code re-seeding every 100 steps; only `recon + 0.25·commit` carries a gradient. Plain Adam with no weight decay is used because decay would shrink the encoder outputs toward the origin while the EMA codebooks follow them, contracting the residuals at depths 2–3 and collapsing utilization. `tests/test_training.py::test_rqvae_codebooks_are_not_optimizer_parameters` guards this, and `tests/test_overfit.py::test_rqvae_reconstructs_with_ema_codebooks` shows the reconstruction actually converges.

**Collision handling.** Two products that quantize to the same three codes receive different fourth-level tokens; the trie's leaves are therefore unique and the `item_for` round trip is exact. This still leaves a subtler failure, discussed next: a prefix can be allowed because *some* leaf under it is eligible even though the leaf finally decoded is not.

### 4.2 The prefix trie as a compliance mechanism

**Choice.** `layers/prefix_trie.py::SemanticIdTrie` is built once from the full catalog. Per request the caller supplies the member's eligible-item mask, and `allowed_children_batch(prefix, allowed)` returns, for a batch of beam prefixes, exactly the codes whose subtree contains at least one eligible leaf. `models/tiger/model.py::TIGER.generate` sets every other logit to `−∞` before the top-k, so an ineligible product cannot be generated. `tests/test_tiger.py::test_trie_vectorized_mask_matches_brute_force` compares the vectorized mask with a dictionary walk; `test_tiger_beams_are_all_eligible_and_unique` and `test_held_and_pending_items_never_generated_even_when_top_scored` are the zero-violation tests: a model deliberately overfitted to prefer a held or pending item still never emits it.

**Why masking at generation beats post-filtering.** A post-filter on a top-100 list yields fewer than 100 candidates for heavily gated members (a heavily gated member may be eligible for only a few dozen products) and, worse, spends the beam's capacity on items that will be discarded. Masking during decoding spends every beam slot on an eligible item, which is why the measured beam yield is 0.9807 of 100 slots even though the mean member is eligible for only 786.5 of 2 000 products (`docs/results/retrieval_metrics.md`, `docs/results/latency.md`).

**Why the trie alone is insufficient: the three enforcement points.** The pipeline enforces eligibility three times (`serving/pipeline.py::RecommendationPipeline.run`), and each point catches something the previous one cannot.

1. *The trie logit mask* (gate ①) is the constraint at generation time. It depends on the item → Semantic ID → item round trip being exact and current. It is not, in three ways: the IDs are a lossy hash of the features, so a stale trie after RQ-VAE retraining maps codes to different items; the fourth level is assigned once and a catalog update can shift it; and a prefix is allowed as soon as *one* eligible leaf lies under it, so an early beam that survives on the strength of an eligible sibling can, after a later level is masked, end on a code sequence whose only decodable leaf is ineligible unless the last level is masked as strictly as the first. The implementation masks every level, but the guarantee is a property of the trie *and* the codes *and* the catalog version, which is too many moving parts to be the only defence.
2. *The post-retrieval vectorized gate* (gate ②, `serving/eligibility_engine.py::EligibilityEngine.filter_candidates`) re-asserts `allowed[ids]` on the returned item ids. It knows nothing about Semantic IDs, so it survives swapping TIGER for two-tower + ANN, adding a fallback retriever, or serving from a cache. On the benchmark it never removes anything (`tests/test_pipeline.py` asserts `num_after_post_filter == num_retrieved`), which is exactly the evidence that gate ① works; the gate exists for the day that stops being true.
3. *The output assertion* (gate ③, `EligibilityEngine.assert_all_eligible`) runs on the served slate and raises `ComplianceViolation`. It covers every stage after retrieval: a bug in valuation, guardrails or PRM that resurrects a masked candidate, or a held product re-entering through the PRM's greedy fill, cannot reach the member. `tests/test_pipeline.py::test_output_assertion_catches_violations` injects such a slate and checks the exception.

**What each gate masks.** The eligibility engine (`EligibilityEngine.mask`) compiles the catalog's declarative gates into dense arrays and evaluates one `(users × products)` boolean matrix with four broadcast comparisons (FICO floor, DTI ceiling, income floor, licensed state), then removes held products, products with an open application, and whole families under the pending-family policy: an open mortgage or auto-refinance application masks the whole family, while a second card or personal-loan application is legitimate and is only penalized at valuation (Section 7.5). `tests/test_tiger.py::test_eligibility_mask_matches_scalar_reference` and `test_each_gate_individually` compare the vectorized engine against the scalar reference rule by rule.

### 4.3 The TIGER decoder and beam search

**Choice.** `models/tiger/model.py::TIGER` is decoder-only: history item tokens and target tokens live in the same Semantic-ID vocabulary, so one causal stack over `[TIER, STATE, history SIDs…, c₁, …, c₄]` needs one embedding table and makes beam search a plain next-token loop. Two member-context tokens (credit tier and state) sit physically first so that under left padding every real token can attend to them. The model is small (`d = 64`, two layers, two heads, feed-forward 128) and reads the last 20 history items, which at four tokens each is 80 tokens plus context.

**Training signal.** Next-Semantic-ID cross-entropy, restricted at each level to that level's token range so that it matches trie-masked decoding, and with `train_all_positions` every history item's codes are also predicted from its prefix, turning one record per member into roughly `L` training examples (`TIGER.next_sid_loss`, `tests/test_tiger.py::test_next_sid_loss_history_term_counts_only_valid_slots`). Checkpoints are selected on validation `Recall@100`, early stopping watches the validation loss (`docs/results/training_summary.md`: best checkpoint at step 200, stopped at step 800 of a 3000-step budget).

**Beam search.** Beams live in a `(B, W, ℓ)` tensor with `W = 100`; each level is one forward pass over `B·W` rows. Dead beams (a prefix with no allowed continuation) are marked `−1` and reported as beam yield rather than back-filled, because a back-fill would be a second, unmasked retrieval source.

**Evidence.** `tests/test_overfit.py::test_tiger_overfits_and_reaches_full_recall_on_memorized_examples` is the capacity oracle. The measured retrieval quality on the test split, with next-item positives and both baselines restricted to the same eligible mask and the same number of slots (`metrics/retrieval_baselines.py`), is:

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

**Reading the numbers honestly.** TIGER beats both baselines at `Recall@10` and beats random everywhere, but eligible-popularity is ahead at `Recall@50` and `Recall@100`. Two things explain this. First, the generator's next-item process is driven by family affinity and product economics with a strong popularity component, and a popularity list restricted to the eligible set is a strong baseline precisely because eligibility already removes most of the catalog (the tier slice shows why: for deep-subprime members, who are eligible for a few dozen products, any eligible list has recall 0.8750). Second, the beam is a *generative* top-100: it commits to a prefix at level one and spends its width on continuations, so its precision at the head is higher and its coverage at the tail is lower than a scored list of the same length. This is the expected shape of a generative retriever and the reason it is followed by a scorer rather than used alone. The credit-card family, the most popular and most populous, is where the beam's tail coverage is weakest (0.1154), and the model's `Recall@100` of 0.3105 at its best checkpoint (`docs/results/training_summary.md`) had already begun to drift by step 400, so the retriever is under-trained relative to its budget rather than mis-designed; the register lists the budget as the parameter that would change it. Positives are also scored on the served slates with click, apply and approve as the positive definition (the last block of `docs/results/retrieval_metrics.md`); recall is 0.2532 / 0.2376 / 0.2211 at 100, decreasing down the funnel as expected.

**Placement.** Retrieval sits first because it is the only stage whose cost grows with catalog size and the only one where compliance can be enforced on a logit vector. Moving it later would mean scoring the full catalog with the HSTU, which Section 5.4 shows is not affordable; moving the compliance mask later (post-filter only) would surrender the beam-yield argument above.
## 5. Stage 2 — HSTU scoring backbone

### 5.1 The layer

`layers/hstu.py::HSTULayer` implements one Hierarchical Sequential Transduction Unit (Zhai et al., 2024) over `X ∈ ℝ^{B×S×d}`:

```
[U, V, Q, K] = split(SiLU(f₁(LN(X))))                         # pointwise projections
A            = SiLU(Q Kᵀ / √d_h + rab_pos + rab_time)          # no softmax
A            = A ⊙ mask / n_valid_keys                          # normalized by valid keys
Y            = X + f₂(LN(A V) ⊙ U)                             # gated residual update
```

**Pointwise gated attention versus softmax.** Softmax normalizes each query's attention to sum to one, which erases *intensity*: a member who viewed twelve cards and one who viewed a single card produce equally normalized rows. Credit intent is largely intensity (how many pulls, how many applications, how recently), so the pointwise SiLU form, normalized only by the count of valid keys, is the right inductive bias. The gate `⊙ U` lets a layer suppress a whole channel of the update rather than merely reweight positions.

**Relative position and time biases.** No absolute positions are embedded. Each head carries a learned bias over relative position in `[−(L+K), L+K]` and over a 32-bucket log-spaced relative-time table (`layers/hstu.py::bucketize_time`, `tests/test_hstu.py::test_time_buckets_monotone_and_bounded`). This is what lets the candidate tokens described next sit "at now" without an absolute index: they are one relative position after the last real history token and at zero time offset from it.

**Embedding.** `models/hstu/model.py::HSTUBackbone.embed` builds history tokens as `W_item(i) + W_action(a) + W_time(bucket(Δt))` and candidate tokens as `W_item(c) + W_candtype + W_time(0)` from a shared item table with `padding_idx = 0`. The action embedding is what makes a `SCORE_CHANGE` event (item 0) informative while a padding slot (item 0, action 0) stays zero; `tests/test_collator.py::test_score_change_embedding_not_zero_but_pad_is` checks exactly this.

### 5.2 M-FALCON: one `B × (L + K)` pass

**Choice.** The 100 retrieved candidates are appended to the member's history as `K` extra tokens, and `layers/hstu.py::build_mfalcon_mask` produces a `(B, L+K, L+K)` boolean mask in which history attends causally to real history, each candidate attends to all real history and to itself, candidates never attend to each other, and history never attends to candidates. One forward pass of length `L + K` then yields the user state `h_user = hidden[last real history token]` and every candidate's contextual representation `h_cand ∈ ℝ^{B×K×d}` (`HSTUBackbone.score_candidates`).

**Why candidate independence is a correctness property, not a speed trick.** The mask makes each candidate's representation a function of the history and of that candidate only. If candidates could attend to each other, a product's score would depend on which other products the retriever happened to return, so the same member and product would score differently on two requests with different beams, calibration (Section 7.1) would be fitted on scores whose meaning drifts with the candidate set, and the audited `p̂` would not be a property of the (member, product) pair. `tests/test_hstu.py::test_candidate_independence_permutation_and_removal` permutes and removes candidates and asserts bit-level invariance of the survivors' representations; `test_batched_equals_per_candidate_loop` shows the batched pass equals scoring each candidate alone; `test_causality_under_left_padding`, `test_pad_content_does_not_leak` and `test_gradient_of_past_wrt_future_is_zero` cover the history half of the mask.

**Speed.** Scoring 100 candidates one at a time would run 100 passes of length `L + 1`; the batched pass runs one of length `L + 100` and costs 2.3920 ms p50 on the benchmark (`docs/results/latency.md`). Memory is `B × (L + K) × d` per layer, which is the quantity that fixes the ranker's batch size (Appendix A).

### 5.3 User–candidate fusion

`HSTUBackbone.fusion` builds `[h_cand ‖ h_user ‖ h_cand ⊙ h_user ‖ tabular]`, where `tabular` is the concatenation of the member's 26 features and the product's 18, standardized by buffers fitted on the training split inside the model (`models/ranker.py::HSTUPLERanker`) so that serving cannot drift from training. The fusion vector is the only input to the funnel; nothing about business value enters it.

**Alternative considered: the hybrid predecessor.** The design this replaces was a TransAct-style short-window transformer plus a PinnerFormer-style long-term embedding fused by a DCN-v2 cross network. Its three components each modelled part of what one HSTU layer does (recent intensity, long-term taste, feature crosses), and the DCN-v2 tower was the only place the candidate met the history, so candidate scores were not conditioned on the sequence itself. HSTU with candidate tokens gives candidate-conditioned sequence attention, relative-time awareness and the pointwise intensity signal in one module, with a single mask whose properties are testable.

### 5.4 Placement: why the scorer follows the retriever

The cascade argument is quantitative. From `docs/results/latency.md`, the HSTU pass over `L + 100` tokens costs 2.3920 ms p50. Scoring the full 2 000-item catalog with the same backbone would need twenty such passes (or one pass of length `L + 2000`, whose attention cost grows quadratically in the candidate count under this mask), which is far outside a 10 ms budget before valuation and re-ranking are counted. Conversely, the generative retriever reads only the last 20 items as 4-token Semantic IDs and never sees the candidate at ranking fidelity: it cannot, by construction, evaluate a candidate against a 64-event history with action types and time gaps. The retriever therefore proposes and the scorer disposes; neither can do the other's job inside the budget. The latency table in Section 8 is the evidence.

## 6. Stage 3 — Multi-task funnel

### 6.1 PLE on top of the fusion vector

**Choice.** `models/ple/model.py::PLE` stacks two Customized-Gate-Control blocks (`CGCBlock`) over the 236-dimensional fusion vector. Each task (click, apply, approve, amount) owns one expert and shares two; every expert is a 236 → 64 → 32 MLP (`Expert`); a task's gate mixes only its own experts and the shared ones, so task-specific parameters are shielded from other tasks' gradients while shared knowledge still flows. Four towers (32 → 1 for the three probabilities, 32 → 3 for the ZILN amount head, `losses/ziln_loss.py::ZILNHead`) emit raw logits.

**Alternatives and the seesaw argument.** A shared-bottom network places every task's gradient on one trunk: improving the approval tower degrades the click tower and vice versa, because a flashy high-APR card is clicky but rarely approved. MMoE softens this with one gate per task but every expert is still shared, so a click-hungry gradient can overwrite the expert the approval tower depends on. PLE isolates task experts and is the standard remedy. Per-task backbones would remove the seesaw entirely but would triple the HSTU cost and make the towers disagree about the member state, which the entire-space loss terms of Section 6.3 need to share.

**Evidence.** `tests/test_overfit.py::test_hstu_ple_ranker_overfits_unified_funnel_loss` is the capacity oracle for the whole ranker. The measured funnel quality is in Section 9; the towers' AUC ordering (approve ≫ apply > click) mirrors the oracle ceiling's ordering, which is what a correctly specified multi-task model should show.

### 6.2 The towers and what they do not see

The towers output `z₁ = logit p(click)`, `z₂ = logit p(apply | click)`, `z₃ = logit p(approve | apply)` and `[π_logit, μ, σ_raw]` for the amount. No payout, no benefit, no `α` and no guardrail enters the towers, the loss or the fusion vector. This is deliberate (Section 7.4): the probabilities must remain proper-scoring-rule estimates so they can be calibrated and audited, and a business weight embedded in them would have to be re-learned every time it changed.

### 6.3 The Unified Funnel Loss

**The problem it solves.** The apply tower is only observed on clicked impressions and the approval tower only on applications, but at serving time both are evaluated on every retrieved candidate. Training them on their observed sub-populations alone gives the classic sample-selection bias: `p(apply | click)` is learned on the clicky products and extrapolated to the rest. ESMM fixes this by supervising the *products* `p₁p₂` and `p₁p₂p₃` on the entire impression space, which is unbiased because every impression is a sample from it.

**The loss, term by term** (`losses/funnel_loss.py::UnifiedFunnelLoss`; `s_k = logsigmoid(z_k)`, `o` = approval-observed mask, `w` = the pending-policy weight of Section 6.4):

| term | sample space | weight | definition |
|---|---|---|---|
| `L_click` | all impressions | λ = 1 | `BCE(z₁, y_click)` |
| `L_apply` | clicked impressions | λ = 1 | `BCE(z₂, y_apply)` |
| `L_approve` | applied ∧ observed | λ = 1, row weight `w` | `w · BCE(z₃, y_approve)` |
| `L_ctcvr` | all impressions | μ = 1 | `−[y_apply (s₁+s₂) + (1−y_apply) log1mexp(s₁+s₂)]` |
| `L_ctcavr` | all impressions | μ = 1, row weight `w′` | `−w′[y_approve (s₁+s₂+s₃) + (1−y_approve) log1mexp(s₁+s₂+s₃)]` |
| `L_amount` | applied ∧ observed ∧ approved | λ = 0.1 | ZILN negative log-likelihood |

with `w′ = 1` on non-applied rows, `w` on resolved applications and 0 on pending ones. Every masked term is normalized by its mask sum rather than by `B·K` (`losses/stable.py::masked_mean`), so a batch with three applications does not see its approval term vanish, and an empty sub-population contributes exactly zero with a valid gradient (`tests/test_funnel_loss.py::test_mask_sum_normalization_and_empty_sub_populations`).

**Why this *is* ESMM without a second network.** The multiplicative terms supervise `z₂` and `z₃` on every impression, including non-clicked ones, through the product with `z₁`; the conditional terms keep each tower sharp on its observed sub-population. Both sets of terms act on the *same* towers, so there is no separate CTCVR network to keep consistent with the conditional heads and no way for the product estimate and the conditional estimate to disagree at serving time. `tests/test_funnel_loss.py::test_entire_space_terms_supervise_z2_on_unclicked_rows_and_detach_flag` shows a non-clicked row produces a gradient on `z₂` only through the entire-space term.

**Why log space, and why `BCE_with_logits` cannot be used.** The entire-space label is Bernoulli with probability `σ(z₁)σ(z₂)`, and a product of sigmoids is not a sigmoid of any sum, so there is no single logit to hand to a fused binary-cross-entropy. Writing the loss in probability space, `−log(1 − p₁p₂)`, underflows: at `|z| = 30` the product is `≈ 1e-26` in float32 and `1 − p` rounds to 1 for positives or to 0 for a confident product, killing the negative-class gradient exactly where a confident wrong prediction should be corrected. The implementation keeps `log p = s₁ + s₂` and computes `log(1 − p) = log1mexp(s₁ + s₂)` with the two-branch identity of Mächler (2012) (`losses/stable.py::log1mexp`), which is exact and gradient-safe everywhere. `tests/test_funnel_loss.py::test_log_space_product_equals_naive_product_at_moderate_logits` shows agreement with the naïve form where the latter is valid, and `test_finite_loss_and_gradients_at_extreme_logits` shows finiteness at `|z| = 30`, where the naïve form is not.

**`ssb_mode` options.** The default is `none` (entire-space terms only). `ips` reweights the conditional apply term by `1 / clip(p̂₁, 0.05)` and `dr` adds an ESCM²-style imputation tower with the doubly-robust estimate; both are implemented and exercised by `tests/test_funnel_loss.py::test_ssb_and_balancing_variants_run` but are not the default because on this generator the entire-space terms already remove the bias the conditional terms would introduce, and the IPS weights add variance without a measured gain. This choice is *unsupported by a results table*; the register flags it.

**Imbalance handling.** Three mechanisms, each placed where it belongs.

- *Negative down-sampling with logit correction* (`training/downsampling.py::NegativeDownsampler`, `models/ranker.py::HSTUPLERanker.predict`). Every clicked impression is kept and each non-clicked one with probability `r = 0.25`; the click tower then learns odds inflated by `1/r`, and the exact correction `z_true = z_train + log r` is applied at inference only, as a buffer set from the training rate. The keep mask multiplies every entire-space term (`tests/test_funnel_loss.py::test_train_mask_removes_rows_from_every_entire_space_term`), which is equivalent to dropping the rows while keeping tensor shapes static; `test_logit_correction_recovers_true_click_probability` shows the corrected logits recover the un-sampled probability. Applying `r` as a loss weight instead would leave the tower a proper scoring rule on the *weighted* distribution, which is not the serving distribution, and would still process every negative.
- *Loss balancing.* Fixed weights by default; `running_mean` normalization and Kendall-style uncertainty weighting are implemented (`UnifiedFunnelLoss._combine`) and tested but not used, since the measured term magnitudes at convergence are within one order of magnitude of each other.
- *Why not focal loss, `pos_weight` or positive oversampling.* All three change the minimizer of the tower away from the conditional probability: focal loss down-weights well-classified examples and is not a proper scoring rule; `pos_weight` and oversampling shift the intercept by a known amount that is only exactly correctable for a logistic model with the same features at both times, which a deep tower is not. The output of these towers is not a ranking score but a probability that is calibrated, multiplied into dollars and shown to auditors; the only imbalance remedy compatible with that is one whose correction is exact, which `+ log r` is.

**Measured positives per batch.** With 64 slates of 100 candidates and `r = 0.25`, the first twenty training batches carried on average 490.0 clicks, 102.3 applications, 97.3 resolved applications and 81.35 resolved approvals out of 1954.85 kept rows (`docs/results/positives_per_batch.md`); this is the count that Appendix A uses to justify the batch size.

### 6.4 Delayed feedback as a labeling concern

**Observability at the cut-off.** `data/delayed_feedback.py::observed_status` classifies each application by whether `served_at + delay ≤ snapshot`:

| status | `y_apply` | `approve_observed` | `y_approve` used | in `L_approve` | in `L_ctcavr` (`w′`) | `p₃` calibration | revenue attribution |
|---|---|---|---|---|---|---|---|
| NOT_APPLIED | 0 | yes | 0 | no (not applied) | yes, `w′ = 1` | no | none |
| APPROVED | 1 | yes | 1 | yes, `w` | yes, `w` | yes, `w` | payout counted |
| DECLINED | 1 | yes | 0 | yes, `w` | yes, `w` | yes, `w` | none |
| PENDING | 1 | no | 0 (oracle hidden) | no | no, `w′ = 0` | no | excluded, revenue is a lower bound |

**`pending_policy` semantics** (`approve_weights`): `drop` gives pending rows weight 0 and everything else weight 1; `ipw` keeps the same masks but gives resolved applications the Horvitz–Thompson weight `1 / max(F_{y,fam}(elapsed), w_floor = 0.05)` where `F` is the outcome-conditional delay CDF; `negative` is the wrong-by-construction baseline that trains pending rows as observed declines. The default is `drop`.

**Why this touches no architecture.** The policy is a per-row weight and an observability mask computed in the collator; the towers, the fusion vector and the serving path are unchanged. Serving never reads an application status: `P_funded = p̂₁p̂₂p̂₃` uses the predicted, calibrated probabilities, and a pending status affects only which rows train and calibrate `z₃` and which rows are counted as realized revenue (`valuation/expected_value.py`). `tests/test_funnel_loss.py::test_pending_rows_give_zero_gradient_to_z3` and `tests/test_calibration_and_valuation.py::test_p3_is_never_fitted_on_unobserved_rows_and_uses_weights` pin the two places where an oracle label could otherwise leak.

**Why `drop` over `negative` and `ipw`.** The ablation retrains the ranker three times on a mortgage-heavy dataset with a young cut-off (0.7500 of applied mortgage rows still pending) and compares the approval tower with the generator's truth on the test split's applied mortgage rows (`serving/evaluation.py::pending_policy_ablation`):

<!-- source: docs/results/pending_policy_ablation.md -->
| pending_policy | applied mortgage rows | pending share | mean p_approve (truth) | mean p̂3 | bias | AUC vs oracle | ECE vs oracle |
|---|---|---|---|---|---|---|---|
| drop | 48 | 0.7500 | 0.6255 | 0.6914 | 0.0659 | 0.7554 | 0.1081 |
| ipw | 48 | 0.7500 | 0.6255 | 0.7976 | 0.1721 | 0.7214 | 0.2143 |
| negative | 48 | 0.7500 | 0.6255 | 0.0346 | -0.5909 | 0.6321 | 0.5487 |

`negative` collapses the mortgage approval estimate to 0.0346 against a truth of 0.6255 (bias −0.5909), which would make every mortgage look unfundable and remove the family from every slate. `drop` is nearly unbiased (0.0659) with the best ranking and calibration against the oracle. `ipw` over-corrects (bias 0.1721): the weights `1/F` for the few resolved mortgages are large, the variance of the estimate on 48 rows dominates, and the correction assumes the delay law is known exactly, which is the strongest assumption in the pipeline. `drop` is the default because it is the only policy whose correctness does not depend on the delay model, at the price of ignoring the information that "still pending after `t` days" carries; `tests/test_delayed_feedback.py::test_pending_policy_bias_on_mortgage_slice_at_young_cutoff` shows the same ordering analytically at the label level.

## 7. Stage 4 — Calibration, valuation, re-ranking

### 7.1 Calibration before valuation, never after PRM

**Placement.** `serving/stages.py::ValuationStage` fixes the order `raw logits → calibrate → P_funded / EV / E[amount] → NB → U + guardrails`, and the PRM comes after. Calibration must precede valuation because valuation multiplies probabilities into dollars: `EV = p̂₁p̂₂p̂₃ · payout` is only meaningful if each factor is a probability, and a ranking score with the right order but the wrong scale gives dollar figures that are wrong by an unknown factor per tower. Nothing is calibrated after the PRM because the PRM outputs an ordering, not probabilities: the `p̂` that are logged, shown as approval odds and audited are the calibrated ones computed before it, so a member can be told "your approval odds are 0.79" and the number is the one the model was scored on.

**Isotonic by default, Platt as the low-data fallback.** `serving/calibration.py::IsotonicCalibrator` is weighted pool-adjacent-violators on probabilities; `PlattCalibrator` is `σ(a·z + b)` on logits fitted by weighted Newton steps on the exact Hessian. Both accept per-row weights so that `p̂₃` is fitted on resolved applications with the same `approve_weight` used in training (`fit_funnel_calibrators`). `fit_calibrator` requests isotonic and falls back to Platt when the weighted positive count is below 500, because isotonic overfits into a jagged staircase on scarce positives. Temperature scaling is omitted deliberately: for one binary logit it is Platt with `b = 0`, so it can only match or lose. `tests/test_calibration_and_valuation.py::test_isotonic_weighted_pava_hand_computed_and_monotone` and `test_platt_recovers_known_parameters_with_weights` check both fits; `test_fit_calibrator_fallback_and_auto_selection` checks the selection rule.

**Measured.** Calibrators are fitted on the calibration split and scored on the test split (`docs/results/calibration.md`); the served calibrators were isotonic on all three towers (weighted positives 8984.0, 1793.0 and 1435.0, all above the threshold).

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

Isotonic cuts click ECE from 0.0085 to 0.0019 and apply ECE from 0.0545 to 0.0117, at the price of a large MCE (0.1213 and 0.4882): a step function's worst bin is a sparse bin. Platt is the better calibrator on the apply tower here (ECE 0.0085, MCE 0.0236) and is close on click. On the approve tower, fitted on 1792 test rows, neither method beats the raw sigmoid on ECE (0.0193 raw against 0.0280 isotonic and 0.0251 platt), and the raw MCE of 0.5000 is a single empty-bin artefact; with 1435.0 weighted positives the tower is barely above the isotonic threshold and the calibration split is small. The per-tier block of `docs/results/calibration.md` shows the same pattern: on the approve tower Platt is better calibrated than isotonic in four of five tiers (for example 0.0178 against 0.0323 ECE on PRIME), while on click isotonic and Platt are within 0.004 of each other in every tier. The honest conclusion is that the default should remain isotonic where it has thousands of positives (click) and that `method = "auto"`, which picks the lower held-out weighted NLL per tower, is the setting to prefer when the calibration split is this small; the register lists `calibration.method` with that sensitivity.

### 7.2 Expected value

`valuation/expected_value.py` computes `P_funded = p̂₁ · p̂₂ · p̂₃`, `EV = P_funded · partner_payout` and `E[amount]`. The EV oracle for the test split is instructive: realized partner revenue on resolved rows is $363,467 and the model's expected revenue Σ P_funded · payout is $397,259 (`docs/results/funnel_metrics.md`); the realized figure is a lower bound because the 94 pending applications in the test split are excluded from it, which is exactly the accounting rule of Section 6.4.

### 7.3 The ZILN amount tower

Approved credit limits and funded loan sizes are zero for most impressions and heavy-tailed when positive. `losses/ziln_loss.py::ziln_loss` models the amount as a mixture: zero with probability `1 − π`, otherwise `LogNormal(μ, σ)`, with `L = BCE(π, 1[y > 0]) + 1[y > 0](log σ + log y + (log y − μ)² / 2σ²)` and `E[Y] = π · exp(μ + σ²/2)` (`ziln_expected_value`, exponent clamped). The tower is trained in Stage 3, on resolved approved rows only (`m_amt` in the loss), because that is where the label lives, but it is consumed only in Stage 4: the benefit formulas need `E[amount]` to size a balance transfer or a loan, while the three probabilities never depend on it. `tests/test_overfit.py::test_ziln_overfits` is the capacity oracle.

### 7.4 Net user benefit and the trade-off weight `α`

**`NB` is a deterministic formula** (`valuation/user_benefit.py::net_user_benefit`), vectorized over `(B, K)`, per family over a horizon `H = 2` years (mortgages `H_hold = 7`): a balance-transfer card saves `B · r_rev · m_intro/12` minus the transfer fee and the annual fee over the intro period with `B = min(revolving_balance, E[amount])`; a credit card earns rewards on annual spend plus the sign-up bonus minus annual fees; a personal loan saves the APR difference on `A = min(E[amount], other_debt)` over the term minus origination; auto refinance and mortgage refinance take the payment difference over the comparison window *minus the difference in principal still owed at the end of it* minus fees and closing costs. The last term keeps refinance comparisons honest: stretching a loan into a longer term lowers the payment but leaves more principal outstanding, and payment relief alone would flatter a costlier loan. Every product then pays the hard-pull cost `c_pull = 15` dollars, doubled when the member has two or more recent hard pulls. `tests/test_calibration_and_valuation.py::test_net_user_benefit_hand_computed_per_family` reproduces each formula by hand.

**Why `NB` is not a learned tower.** A learned benefit would have no ground truth to learn from (nobody observes the counterfactual savings), would entangle a business judgement with behavioural probabilities, and would make "why was this recommended?" unanswerable, whereas the formulas yield "this saves you $X over two years".

**Why `α` lives at valuation time and not in a loss.** `valuation/utility.py::utility` computes

```
U_i = P_funded,i · ( α · payout_i + (1 − α) · NB_i )          α ∈ [0, 1], dollars
```

Both terms are multiplied by the same funnel probability because the member only realizes the benefit if approved and funded; both are in dollars, so `α = 0.5` values a dollar of payout the same as a dollar of member savings, `α = 1` recovers `U = EV`, and `α = 0` ranks purely by expected member benefit. Placing this inside a loss would cost three things: *calibration entanglement* (a tower trained on a payout-weighted objective no longer estimates a probability, and Section 7.1 needs probabilities); *retraining to change a business weight* (a policy decision made by a committee should be a config change reviewed in minutes, not a training run); and *unexplainable recommendations* (with `α` in the weights there is no decomposition of a score into "expected payout" and "expected benefit", and no way to answer a regulator's question). At valuation time the decomposition is exact, the sweep of Section 7.7 is an offline evaluation rather than a retraining, and `tests/test_calibration_and_valuation.py::test_expected_value_and_utility_endpoints` pins the two endpoints.

### 7.5 Suitability guardrails between utility and PRM

`valuation/utility.py::apply_guardrails` enforces three hard rules on the valued candidates before anything learned sees them: *do-no-harm* excludes any candidate with `NB < −δ` (`δ = 25` dollars) whenever a same-family candidate with `NB ≥ 0` exists in the slate and otherwise penalizes it by 25 dollars; *refinance sanity* excludes auto or mortgage refinance unless the new APR is below the member's current rate; and the *pending-family policy* charges 15 dollars of `NB` for a second application in a family whose rule is `penalize` (families with rule `mask` were already removed by the eligibility engine). Excluded candidates get `U = −∞`.

**Placement.** The rules sit after utility (they need `NB`) and before the PRM because hard rules must not be learnable away: a re-ranker trained on click labels will happily learn that a harmful high-APR product gets clicks. The PRM only ever sees survivors, so no amount of training can reintroduce an excluded product, and gate ③ would catch it if one did. `tests/test_calibration_and_valuation.py::test_guardrails_do_no_harm_refinance_and_pending_family` exercises each rule.

### 7.6 PRM re-ranking with a family-cannibalization penalty

**Choice.** `models/prm/model.py::PRM` is a one-layer Pre-LN transformer (`d = 32`, two heads) over the ten slots with the highest utility. Its input per slot is `[h_cand ‖ p̂₁ p̂₂ p̂₃ ‖ EV NB U (÷100) ‖ family one-hot ‖ position]` concatenated with the user vector (`build_prm_features`), where the position is the incoming utility rank. It is trained listwise (`losses/listwise_loss.py::prm_listwise_loss`, ListNet cross-entropy against `q ∝ y_click` by default or `softmax(U / 25)` with `prm_target = "utility"`) on training slates that contain at least one positive, built by running the same valuation stage the serving path uses (`training/trainers.py::build_prm_examples`), so its training data and its serving input cannot drift apart. `tests/test_overfit.py::test_prm_overfits_listwise` is its oracle and `tests/test_calibration_and_valuation.py::test_prm_features_targets_and_rerank` covers the feature builder.

**Why a listwise slate model.** Pointwise scores treat each candidate in isolation; a slate is not a set of independent decisions. Two balance-transfer cards side by side split the same click and a flashy card next to a mortgage changes how the mortgage looks. Self-attention over the ten slots conditions every score on the whole slate.

**Why the cannibalization penalty is applied at inference and not in the loss.** `rerank` builds the slate greedily: pick the best remaining slot, subtract `β = 0.5` from every remaining candidate of the same family, repeat. This is a serving policy about diversity, and it is kept out of the loss so it can be tuned without retraining and so that the PRM's scores remain a pure estimate of slate-conditioned relevance; the loss has an optional same-family regularizer whose weight is 0 by default. With `β = 0` the rerank is a plain argsort, which is what the tests check first.

**What the PRM does not do.** It outputs an ordering. The `p̂`, `EV`, `NB` and `U` that were computed before it are what is logged with the slate (`serving/pipeline.py::SlateResult`); no probability is recomputed or rescaled after re-ranking.

### 7.7 The measured Pareto frontier

`valuation/pareto.py::pareto_sweep` re-values the test slates for `α ∈ {0, 0.1, …, 1}`, applies the guardrails, serves the top-10 by utility (the PRM is deliberately excluded so the sweep isolates the valuation policy) and reports expected revenue, expected member benefit and the harm rate per slate. `tests/test_calibration_and_valuation.py::test_alpha_sweep_is_weakly_monotone_and_renders` asserts revenue is weakly increasing and benefit weakly decreasing in `α`.

<!-- source: docs/results/pareto_sweep.md -->
| α | ExpectedRevenue@10 ($) | ExpectedUserBenefit@10 ($) | HarmRate@10 |
|---|---:|---:|---:|
| 0.0 | 34.63 | 438.00 | 0.000 |
| 0.1 | 34.80 | 437.99 | 0.000 |
| 0.2 | 34.94 | 437.96 | 0.000 |
| 0.3 | 35.16 | 437.89 | 0.000 |
| 0.4 | 35.52 | 437.69 | 0.000 |
| 0.5 | 35.97 | 437.31 | 0.000 |
| 0.6 | 36.64 | 436.47 | 0.000 |
| 0.7 | 37.81 | 434.20 | 0.000 |
| 0.8 | 39.81 | 427.99 | 0.002 |
| 0.9 | 43.46 | 405.57 | 0.033 |
| 1.0 | 47.41 | 308.46 | 0.160 |

The frontier is flat and safe up to `α ≈ 0.7`: revenue rises from $34.63 to $37.81 while benefit falls by less than four dollars and no member is shown a harmful product. Beyond that the trade-off turns sharply: at `α = 1` revenue is $47.41 but benefit collapses to $308.46 and 0.160 of slates contain a product with `NB < −δ` that could not be excluded. The default `α = 0.5` sits inside the flat region ($35.97, $437.31, harm 0.000). The per-tier block of the same table shows the harm at `α = 1` is spread across tiers (0.070 deep-subprime to 0.193 near-prime) rather than concentrated on the vulnerable end, which is the pattern one expects when payout, not creditworthiness, drives the ranking.

## 8. End-to-end serving

**Order.** `serving/pipeline.py::RecommendationPipeline.run` executes, per member at `B = 1`: eligibility mask (gates + held + pending + family policy) → TIGER beam over the trie with that mask → post-retrieval gate → one HSTU + PLE pass with `+ log r` on the click logit → per-tower calibration → EV, `E[amount]`, `NB`, `U` and guardrails → PRM over the top-10 by `U` with the greedy family penalty → output assertion and telemetry. `tests/test_pipeline.py::test_pipeline_serves_eligible_unique_guarded_slates` checks every served slate is eligible, unique, not held, not pending, at most ten long and reported with the stage list; `test_pipeline_handles_user_with_nothing_eligible` checks the empty case; `test_artifacts_round_trip_gives_identical_slates` checks that saving and reloading the artifacts reproduces the slates bit for bit.

**Precomputed versus per request.** Precomputed and frozen at serve time: the Semantic IDs and the trie, the catalog feature matrix and product economics, the calibrators, the tabular standardization buffers and the model weights. Computed per request: the eligibility mask, the beam, the member's sequence encoding and candidate scoring, valuation, guardrails and re-ranking. The user's financial state is read from the same profile the generator wrote, so no feature is inverted at serving time.

**Latency.** Measured over 50 test members after one warm-up call (`serving/evaluation.py::latency_markdown`):

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

The total p50 of 19.84 ms is above the 10 ms budget, and the table says exactly why: retrieval is 16.5931 ms and everything else together is under 3.3 ms. The beam re-runs the full history sequence at every one of the four levels for 100 beams, so the decoder processes `4 × 100` sequences of about 80 tokens per request; the compliance mask itself is not the cost (the eligibility stage is 0.0236 ms and the vectorized trie query is inside the retrieval figure but is a gather over `(P, 32)`). Three mitigations are available in order of engineering cost: a KV cache so that only the appended code tokens are recomputed per level (the largest win, expected to bring retrieval near the ranking stage's cost), a smaller beam for members whose eligible set is small (the beam yield already reports how many slots were fillable), and batching requests. None changes the design; all are recorded as the retrieval failure mode in Appendix B. The scorer at 2.3920 ms for 100 candidates, the valuation at 0.2681 ms and the PRM at 0.4297 ms are within budget with room to spare, which supports the cascade argument of Section 5.4: the expensive component is the one that must run before candidate count is known.
## 9. Experimental setup and results

### 9.1 Dataset

The benchmark is the synthetic dataset of Section 3: 2 000 products, 3 000 members, 85 312 history events and 1 200 000 impression rows (3 000 members × 4 slates × 100 candidates), generated at a cut-off of 365 days with slates served in the 90 days before it. Of the applications in the histories, 0.0458 are still pending at the cut-off (`docs/results/dataset_summary.md`); in the impression slates 94 test-split applications are pending (`docs/results/funnel_metrics.md`). Members are split 2100 / 300 / 300 / 300.

### 9.2 Training protocol

Each model has its own optimizer configuration (`training/config.py`), a warmup-then-cosine schedule (`training/optim.py::WarmupCosine`, `tests/test_training.py::test_warmup_cosine_schedule_peak_and_floor`), decoupled weight decay on weight matrices only (never on embeddings, biases or normalization gains; `test_build_optimizer_parameter_groups`), global-norm clipping, and early stopping with patience counted in evaluations on a per-model criterion with restoration of the best checkpoint (`training/early_stopping.py`, `test_early_stopping_triggers_after_patience_and_restores_best`). Calibrators are fitted afterwards on the calibration split, and the PRM is trained last on valued training slates. `tests/test_training.py::test_train_all_smoke_and_round_trip` runs the whole procedure on a tiny dataset and reloads the artifacts.

<!-- source: docs/results/training_summary.md -->
| model | steps run | step budget | selection criterion | best step | best value | early stop | seconds |
|---|---|---|---|---|---|---|---|
| rqvae | 2000 | 2000 | recon MSE s.t. min utilization >= 0.9 | 1900 | 0.0190 | no | 3.2733 |
| tiger | 800 | 3000 | val Recall@100 (checkpoint), val next-SID loss (stop) | 200 | 0.3105 | yes | 66.1745 |
| ranker | 700 | 2000 | val unified funnel loss (total) | 400 | 2.3747 | yes | 183.2982 |
| prm | 500 | 600 | val listwise loss | 350 | 2.3033 | yes | 0.9670 |

The RQ-VAE reaches utilization 1.0000 on all three levels. TIGER's best validation `Recall@100` is at its first evaluation and the validation loss rises afterwards, so it stops at step 800 of 3000; the ranker stops at step 700 of 2000 with its best total loss at step 400. Both stopped early on a rising validation loss, which is the overfitting signature of a small dataset rather than an optimization failure (the training losses keep falling in `artifacts/run.log`); the register lists the dataset size and the step budgets as the parameters that would change this. The ranker has 312032 trainable parameters; the PRM trains on 8400 slates with at least one positive.

### 9.3 Funnel metrics

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

<!-- source: docs/results/positives_per_batch.md -->
| quantity | mean per batch |
|---|---|
| rows_before_downsampling | 6400.0000 |
| rows_after_downsampling | 1954.8500 |
| clicks | 490.0000 |
| applies | 102.3000 |
| resolved_approvals | 81.3500 |
| resolved_applications | 97.3000 |

### 9.4 Calibration by tier

The per-tower table is in Section 7.1. Sliced by credit tier (`docs/results/calibration.md`, ECE / Brier on the test split):

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

### 9.5 Pareto frontier by tier

The overall sweep is in Section 7.7. By tier at the two endpoints and the default (`docs/results/pareto_sweep.md`):

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

Benefit scales with tier because prime members have larger balances to refinance and qualify for cheaper products; the `α = 1` policy halves the benefit of deep-subprime members (86.03 → 41.87) for less than eight dollars of extra revenue per slate, which is the concrete case for keeping `α` in the flat region.

### 9.6 Retrieval on served slates

<!-- source: docs/results/retrieval_metrics.md -->
| positive definition | users | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|
| click | 300 | 0.0349 | 0.1457 | 0.2532 |
| apply | 293 | 0.0313 | 0.1372 | 0.2376 |
| approve | 292 | 0.0307 | 0.1286 | 0.2211 |

### 9.7 The test suite as evidence

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
| `tests/test_docs.py` | this document's tables come from the results files, Appendix A equals the config defaults, the diagrams pass their checklist |

## 10. Limitations and future work

**Synthetic data.** The generator has no position bias (a real slate's first slot is clicked more, and the click tower would need a position feature or a debiased label), no seasonality, no competition between concurrent slates and no partner-side drift. Its labels are Bernoulli draws from smooth functions, which makes the oracle ceiling computable but also makes the click task nearly unlearnable (ceiling AUC 0.6316); a real click signal is both noisier and more structured.

**Delay laws assumed known.** The `ipw` policy uses the generator's own delay CDF. In production the delay distribution must be estimated, and the natural estimator is Kaplan–Meier on the resolved applications per family, which turns the pending problem into a survival problem and would let the "still pending after `t` days" information be used rather than dropped. Apply-side delays (a member who clicks today and applies next week) are not modelled at all; the apply label is observed at serve time plus a fixed window.

**Sample-selection correction.** The doubly-robust `ssb_mode = "dr"` path is implemented but unmeasured against `none` on this generator; a real dataset with stronger exposure bias is where that comparison matters.

**Retrieval quality and latency.** TIGER trails eligible-popularity at `Recall@100` and the beam is the whole latency overrun. A KV-cached beam, a popularity-aware Semantic-ID prior and a longer training budget with a larger dataset are the three obvious next steps, in that order.

**Calibration with few positives.** The approval tower's calibration split is small enough that neither isotonic nor Platt improves on the raw sigmoid; `method = "auto"` or a larger calibration split is the fix.

**Online learning and fairness.** Everything here is offline. An online loop would need the logged calibrated `p̂` (which is why they are logged) for off-policy evaluation, and a fairness audit would extend the per-tier slices to protected classes with the same machinery.

## 11. References

- Rajput, S. et al. (2023). *Recommender Systems with Generative Retrieval* (TIGER). NeurIPS.
- Lee, D. et al. (2022). *Autoregressive Image Generation using Residual Quantization* (RQ-VAE). CVPR.
- Zhai, J. et al. (2024). *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations* (HSTU, M-FALCON). ICML.
- Tang, H. et al. (2020). *Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations*. RecSys.
- Ma, X. et al. (2018). *Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate* (ESMM). SIGIR.
- Wang, H. et al. (2022). *ESCM²: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation*. SIGIR.
- Chapelle, O. (2014). *Modeling Delayed Feedback in Display Advertising*. KDD.
- Yasui, S. et al. (2020). *A Feedback Shift Correction in Predicting Conversion Rates under Delayed Feedback* (FSIW). WWW.
- Yang, J. et al. (2021). *Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling* (ES-DFM). AAAI.
- Wang, X., Liu, L. and Miao, N. (2019). *A Deep Probabilistic Model for Customer Lifetime Value Prediction* (ZILN). arXiv:1912.07753.
- Pei, C. et al. (2019). *Personalized Re-ranking for Recommendation* (PRM). RecSys.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*.
- Zadrozny, B. and Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. KDD.
- Niculescu-Mizil, A. and Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*. ICML.
- He, X. et al. (2014). *Practical Lessons from Predicting Clicks on Ads at Facebook*. ADKDD.
- Mächler, M. (2012). *Accurately Computing log(1 − exp(−|a|))*. CRAN vignette.
- Kendall, A., Gal, Y. and Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses*. CVPR.
- Kang, W.-C. and McAuley, J. (2018). *Self-Attentive Sequential Recommendation* (SASRec). ICDM.
## Appendix A — Decision register

Every row is generated from the config dataclasses' defaults (`training/config.py::decision_register`); `tests/test_docs.py::test_register_matches_config_defaults` asserts that the `parameter` and `value` columns below equal the code. The rationale columns are written by hand. "Unsupported by a table" marks a choice that no results table measures.

**Batch sizes.** The three sequence models (RQ-VAE, TIGER, HSTU) and the impression-batched PLE have different batch units, argued in their rows: the RQ-VAE takes the whole catalog per step; TIGER takes 256 sequences, each of which yields roughly `L` next-token examples; the ranker takes 64 slates × 100 candidates, chosen from (a) the measured positives per batch that the masked apply / approve terms need, (b) the `B × (L + K) × d` memory of the batched candidate pass, and (c) the down-sampling rate `r`; serving runs at `B = 1`.

| parameter | value | config location | why this value | alternatives / range considered | what would change it | sensitivity (measured, or expected if not measured) |
|---|---|---|---|---|---|---|
| `generator.num_users` | `3000` | `scripts/generate_dataset.py -> GeneratorConfig` | enough members for a 2100 / 300 / 300 / 300 user split with about 1 400 weighted approvals (1435.0) in the calibration split | 1 000–10 000 | a real dataset; the calibration split's positive count | measured: approval calibration is the split-size-limited tower (§7.1) |
| `generator.num_products` | `2000` | `scripts/generate_dataset.py -> GeneratorConfig` | catalog large enough that eligibility removes most of it (mean eligible 786.5 of 2 000) and that the 32³ code space is sparse | 500–10 000 | catalog size in production | expected: retrieval cost is flat in N (beam), scorer cost is flat (K fixed); trie build is O(N) |
| `generator.seed` | `42` | `scripts/generate_dataset.py -> GeneratorConfig` | fixed for reproducibility; `test_generator_deterministic` | any | nothing | none |
| `generator.min_history` | `5` | `scripts/generate_dataset.py -> GeneratorConfig` | every member has a usable sequence (≥ 5 events) for the collators | 3–10 | sparser real members | expected: low |
| `generator.max_history` | `60` | `scripts/generate_dataset.py -> GeneratorConfig` | longer than HSTU's window L = 64 for some members so truncation is exercised | 20–200 | real history lengths | expected: low |
| `generator.horizon_days` | `365` | `scripts/generate_dataset.py -> GeneratorConfig` | one year of activity before the cut-off | 90–730 | data retention | expected: low |
| `generator.mean_gap_days` | `3` | `scripts/generate_dataset.py -> GeneratorConfig` | ~120 events per year at most; matches min/max history | 1–7 | real event density | expected: low |
| `generator.score_change_prob` | `0.12` | `scripts/generate_dataset.py -> GeneratorConfig` | score-change events are ~12 % of history so the item-0 rule is exercised (measured 0.1201) | 0.05–0.2 | real bureau update frequency | expected: low |
| `generator.apply_prob` | `0.15` | `scripts/generate_dataset.py -> GeneratorConfig` | history apply rate that yields declines and pending events in sequences | 0.05–0.3 | real apply rates | expected: low |
| `generator.credit_pull_prob` | `0.1` | `scripts/generate_dataset.py -> GeneratorConfig` | hard pulls in history feed the fatigue rule in NB | 0.05–0.2 | real pull rates | expected: low |
| `generator.ineligible_browse_prob` | `0.2` | `scripts/generate_dataset.py -> GeneratorConfig` | members browse ineligible products so histories contain them and the trie must mask them | 0–0.5 | product-page exposure policy | expected: low |
| `generator.slates_per_user` | `4` | `scripts/generate_dataset.py -> GeneratorConfig` | 4 × 100 = 400 impressions per member: 1.2 M rows, about 1 400 weighted approvals in the 300-member calibration split | 2–8 | impression volume | measured via `positives_per_batch.md` |
| `generator.slate_size` | `100` | `scripts/generate_dataset.py -> GeneratorConfig` | K = 100 candidates per slate so the impression collator produces the same (B, K) shape the beam does | 10–200 | K | tied to `models.num_candidates` |
| `generator.ineligible_per_slate` | `10` | `scripts/generate_dataset.py -> GeneratorConfig` | 10 % ineligible rows per slate give the retrieval evaluation something to exclude (70 positives excluded) | 0–20 | how compliant the logged policy was | expected: low |
| `generator.future_window_days` | `(14, 28)` | `scripts/generate_dataset.py -> GeneratorConfig` | next-item targets drawn 14–28 days after the history end | (7, 14)–(30, 60) | label window definition | expected: low |
| `generator.max_future` | `16` | `scripts/generate_dataset.py -> GeneratorConfig` | cap on future events per member | 8–32 | label window definition | expected: low |
| `generator.snapshot_at_days` | `365` | `scripts/generate_dataset.py -> GeneratorConfig` | cut-off at the end of the horizon; slates in the last 90 days are then young enough to leave 0.0458 of applications pending | any ≤ horizon | the training cut-off in production is the training date | measured: `pending_rate_given_apply`; the ablation uses 40 d |
| `generator.recency_window_days` | `30` | `scripts/generate_dataset.py -> GeneratorConfig` | histories are anchored within 30 days of the cut-off | 7–90 | member activity | expected: low |
| `generator.served_window_days` | `90` | `scripts/generate_dataset.py -> GeneratorConfig` | slates served in the last 90 days so mortgages (~35 d delay) are often pending | 30–365 | how far back training slates go | measured: pending share; the ablation uses 30 d |
| `generator.family_mix` | `(0.4, 0.15, 0.2, 0.1, 0.15)` | `scripts/generate_dataset.py -> GeneratorConfig` | cards dominate the catalog (0.40 + 0.15) as in a marketplace; the ablation uses a mortgage-heavy mix | any simplex point | catalog composition | measured: per-family recall in `retrieval_metrics.md` |
| `delay.p_instant` | `((0.8, 0.8), (0.8, 0.8), (0, 0), (0, 0), (0, 0))` | `data/schema.py::DelayConfig` | cards decide instantly with probability 0.8; loans, refinance and mortgages never do | 0–1 per family | partner behaviour | measured indirectly via pending share |
| `delay.log_mu` | `((-0.356675, -0.356675), (-0.356675, -0.356675), (0.693147, 0.693147), (1.60944, 1.60944), (3.55535, 2.99573))` | `data/schema.py::DelayConfig` | log-normal medians: cards 0.7 d, personal loan 2 d, auto 5 d, mortgage 35 d approved / 20 d declined | per-family medians | partner behaviour; fit by Kaplan–Meier in production | measured: the mortgage slice is where the policy matters (§6.4) |
| `delay.log_sigma` | `((0.5, 0.5), (0.5, 0.5), (0.3, 0.3), (0.3, 0.3), (0.4, 0.4))` | `data/schema.py::DelayConfig` | moderate spread (0.3–0.5 in log days) | 0.2–1.0 | partner behaviour | expected: affects `ipw` variance |
| `delay.instant_delay_days` | `0.01` | `data/schema.py::DelayConfig` | an instant decision still has a positive delay so the CDF is a proper mixture | 0.001–0.1 | nothing | none |
| `models.d_model` | `64` | `training/config.py::ModelConfig` | d = 64: the smallest width at which the overfit oracles pass quickly on CPU; HSTU memory is B × (L + K) × d | 32–256 | a larger dataset; a GPU | expected: the scorer cost scales linearly in d |
| `models.hstu_layers` | `2` | `training/config.py::ModelConfig` | two HSTU layers: one for intensity, one for interaction, at 2.3920 ms for K = 100 | 1–4 | latency budget; dataset size | measured: ranking latency |
| `models.hstu_heads` | `2` | `training/config.py::ModelConfig` | two heads at d = 64 give d_h = 32 per head | 1–4 | d_model | expected: low |
| `models.hstu_max_len` | `64` | `training/config.py::ModelConfig` | L ≤ 64 covers the full history of most generated members (max 60 events) | 32–256 | real history lengths; latency (attention is quadratic in L + K) | measured: ranking latency |
| `models.num_candidates` | `100` | `training/config.py::ModelConfig` | K = 100 = beam width: enough recall headroom for a 10-slot slate at 2.3920 ms of scoring | 50–500 | latency; slate size | measured: `Recall@100` vs `Recall@50` |
| `models.slate_size` | `10` | `training/config.py::ModelConfig` | ten slots per served slate | 5–20 | product surface | measured: `@10` metrics |
| `models.num_time_buckets` | `32` | `training/config.py::ModelConfig` | 32 log-spaced buckets over 365 days give day resolution near zero and month resolution near one year | 16–64 | horizon | expected: low |
| `models.rq_levels` | `3` | `training/config.py::ModelConfig` | depth 3 with codebook 32 gives 32³ = 32 768 prefixes ≫ 2 000 items; each extra level is one more beam pass | 2–4 | catalog size (deeper for ≫ 10⁵ items) | measured: retrieval latency per level |
| `models.rq_codebook_size` | `32` | `training/config.py::ModelConfig` | 32 codes per level; the trie's child mask is (P, 32) | 16–256 | catalog size | measured: utilization 1.0000 on all levels |
| `models.rq_latent_dim` | `16` | `training/config.py::ModelConfig` | 16-dim latent for an 18-dim input: mild compression so the codes are semantic, not identity | 8–32 | feature dimensionality | measured: recon MSE 0.0190 |
| `models.rq_hidden` | `32` | `training/config.py::ModelConfig` | one 32-unit hidden layer in encoder and decoder | 16–64 | feature dimensionality | expected: low |
| `models.tiger_max_history` | `20` | `training/config.py::ModelConfig` | 20 items × 4 tokens = 80 history tokens per beam row; the dominant term in retrieval latency | 10–64 | latency budget (KV cache would relax it) | measured: retrieval 16.5931 ms |
| `models.tiger_layers` | `2` | `training/config.py::ModelConfig` | two decoder layers | 1–4 | dataset size | expected: low at this scale |
| `models.tiger_heads` | `2` | `training/config.py::ModelConfig` | two heads | 1–4 | d_model | expected: low |
| `models.tiger_d_ff` | `128` | `training/config.py::ModelConfig` | feed-forward width 2 × d | 64–256 | d_model | expected: low |
| `models.ple_levels` | `2` | `training/config.py::ModelConfig` | two CGC levels: one extraction level plus one progressive level | 1–3 | number of tasks | expected: low |
| `models.ple_shared_experts` | `2` | `training/config.py::ModelConfig` | two shared experts carry the common member-state signal | 1–4 | task count | expected: low |
| `models.ple_task_experts` | `1` | `training/config.py::ModelConfig` | one private expert per task is what isolates the seesaw | 1–2 | task conflict | expected: low |
| `models.ple_expert_hidden` | `64` | `training/config.py::ModelConfig` | expert MLP hidden width 64 | 32–128 | fusion dimensionality (236) | expected: low |
| `models.ple_expert_dim` | `32` | `training/config.py::ModelConfig` | expert output 32 | 16–64 | tower width | expected: low |
| `models.ple_tower_hidden` | `32` | `training/config.py::ModelConfig` | tower hidden 32 | 16–64 | expert dim | expected: low |
| `models.prm_d_model` | `32` | `training/config.py::ModelConfig` | the PRM sees ten slots; d = 32 is enough | 16–64 | slate size | measured: rerank 0.4297 ms |
| `models.prm_heads` | `2` | `training/config.py::ModelConfig` | two heads | 1–4 | d | expected: low |
| `models.prm_layers` | `1` | `training/config.py::ModelConfig` | one Pre-LN layer over ten slots | 1–2 | slate size | expected: low |
| `models.prm_d_ff` | `64` | `training/config.py::ModelConfig` | feed-forward 2 × d | 32–128 | d | expected: low |
| `models.prm_cannibalization_weight` | `0.5` | `training/config.py::ModelConfig` | β = 0.5 in score units: a second same-family product must beat the next family by half a unit | 0–2 | diversity policy | unsupported by a table; visible in the e2e slates |
| `models.prm_target` | `click` | `training/config.py::ModelConfig` | ListNet against clicks: the only slate-level label observed at every slot | click | utility | a business decision to rank by U inside the PRM | unsupported by a table |
| `models.prm_utility_temperature` | `25` | `training/config.py::ModelConfig` | τ = 25 dollars: the scale of a typical utility gap between slate candidates | 10–100 | the scale of U | unsupported by a table |
| `split.train` | `0.7` | `training/config.py::SplitConfig` | 70 % of members for training | 0.6–0.8 | dataset size | expected: low |
| `split.val` | `0.1` | `training/config.py::SplitConfig` | 10 % for early stopping and checkpoint selection | 0.05–0.2 | dataset size | expected: low |
| `split.calib` | `0.1` | `training/config.py::SplitConfig` | 10 % held out for calibrators only, disjoint from training and selection | 0.05–0.2 | positives needed for isotonic (500 weighted) | measured: approval calibration (§7.1) |
| `split.test` | `0.1` | `training/config.py::SplitConfig` | 10 % for every reported number | 0.1–0.2 | dataset size | expected: low |
| `split.seed` | `0` | `training/config.py::SplitConfig` | fixed split | any | nothing | none |
| `downsample.rate` | `0.25` | `training/config.py::DownsampleConfig` | r = 0.25 keeps 1954.85 of 6400 rows per batch and every click; the exact `+ log r` correction makes it free | 0.1–1.0 | CTR (lower CTR → smaller r) | measured: `positives_per_batch.md`; click ECE 0.0019 after correction + calibration |
| `downsample.seed` | `0` | `training/config.py::DownsampleConfig` | fixed keep mask per batch | any | nothing | none |
| `pending.policy` | `drop` | `training/config.py::PendingConfig` | `drop`: bias 0.0659 vs 0.1721 (`ipw`) and −0.5909 (`negative`) on the mortgage slice; does not depend on the delay law | drop | ipw | negative | a trustworthy delay estimator would favour `ipw` | measured: `pending_policy_ablation.md` |
| `pending.w_floor` | `0.05` | `training/config.py::PendingConfig` | caps `ipw` weights at 20 (1 / 0.05) so a barely-resolved mortgage cannot dominate a batch | 0.01–0.2 | delay-law variance | expected: `ipw` bias/variance trade-off |
| `funnel_loss.lambda_click` | `1` | `losses/funnel_loss.py::FunnelLossConfig` | unit weight; all probability terms are proper scoring rules on their own sample space | 0.5–2 | loss balancing mode | expected: low |
| `funnel_loss.lambda_apply` | `1` | `losses/funnel_loss.py::FunnelLossConfig` | unit weight | 0.5–2 | loss balancing mode | expected: low |
| `funnel_loss.lambda_approve` | `1` | `losses/funnel_loss.py::FunnelLossConfig` | unit weight | 0.5–2 | loss balancing mode | expected: low |
| `funnel_loss.mu_ctcvr` | `1` | `losses/funnel_loss.py::FunnelLossConfig` | unit weight on the entire-space apply term | 0.5–2 | exposure bias strength | expected: low |
| `funnel_loss.mu_ctcavr` | `1` | `losses/funnel_loss.py::FunnelLossConfig` | unit weight on the entire-space approve term | 0.5–2 | exposure bias strength | expected: low |
| `funnel_loss.lambda_amount` | `0.1` | `losses/funnel_loss.py::FunnelLossConfig` | the ZILN NLL is roughly an order of magnitude larger than a BCE term at convergence; 0.1 brings it to the same scale | 0.05–0.5 | amount scale | expected: moderate on E[amount], none on probabilities |
| `funnel_loss.loss_balancing` | `fixed` | `losses/funnel_loss.py::FunnelLossConfig` | fixed weights; running-mean and uncertainty weighting are implemented and tested but unmeasured | fixed | running_mean | uncertainty | a measured term-scale imbalance | unsupported by a table |
| `funnel_loss.ssb_mode` | `none` | `losses/funnel_loss.py::FunnelLossConfig` | `none`: the entire-space terms already correct selection bias; IPS adds variance | none | ips | dr | a real dataset with strong exposure bias | unsupported by a table |
| `funnel_loss.detach_upstream` | `False` | `losses/funnel_loss.py::FunnelLossConfig` | gradients from the product terms flow into z₁ and z₂ (ESMM behaviour) | True | False | tower interference | unsupported by a table |
| `funnel_loss.ips_eps` | `0.05` | `losses/funnel_loss.py::FunnelLossConfig` | clip for 1 / p̂₁ under `ips` / `dr` | 0.01–0.1 | ssb_mode | n/a under `none` |
| `funnel_loss.running_mean_decay` | `0.99` | `losses/funnel_loss.py::FunnelLossConfig` | EMA decay for `running_mean` balancing | 0.9–0.999 | loss_balancing | n/a under `fixed` |
| `calibration.method` | `isotonic` | `serving/calibration.py::CalibrationConfig` | isotonic where positives are plentiful; Platt was better on apply and approve on this split, so `auto` is the recommended change | isotonic | platt | auto | calibration split size | measured: `calibration.md` |
| `calibration.min_positives_isotonic` | `500` | `serving/calibration.py::CalibrationConfig` | 500 weighted positives: below this a PAVA staircase overfits; all three towers were above it (8984.0, 1793.0, 1435.0) | 200–2 000 | calibration split size | measured: approve tower is close to the threshold |
| `calibration.auto_holdout_fraction` | `0.2` | `serving/calibration.py::CalibrationConfig` | 80 / 20 split inside the calibration set for `auto` | 0.1–0.3 | calibration split size | n/a under `isotonic` |
| `calibration.seed` | `0` | `serving/calibration.py::CalibrationConfig` | fixed `auto` holdout | any | nothing | none |
| `user_benefit.horizon_years` | `2` | `valuation/user_benefit.py::UserBenefitConfig` | H = 2 years: intro periods and reward comparisons are quoted over two years | 1–5 | product policy | expected: scales NB roughly linearly |
| `user_benefit.hold_years_mortgage` | `7` | `valuation/user_benefit.py::UserBenefitConfig` | H_hold = 7 years: typical time a refinanced mortgage is held | 5–10 | product policy | expected: mortgage NB magnitude |
| `user_benefit.hard_pull_cost` | `15` | `valuation/user_benefit.py::UserBenefitConfig` | c_pull = 15 dollars: a hard pull costs a few score points, monetized | 5–30 | credit-score economics | expected: shifts every NB by a constant |
| `user_benefit.fatigue_pulls` | `2` | `valuation/user_benefit.py::UserBenefitConfig` | two or more recent pulls doubles the pull cost | 1–4 | credit-score economics | expected: low |
| `user_benefit.fatigue_multiplier` | `2` | `valuation/user_benefit.py::UserBenefitConfig` | ×2 | 1.5–3 | credit-score economics | expected: low |
| `utility.alpha` | `0.5` | `valuation/utility.py::UtilityConfig` | α = 0.5 sits inside the flat region of the frontier: $35.97 revenue, $437.31 benefit, harm 0.000 | 0–1 | a business decision; re-run `pareto_sweep.py` | measured: `pareto_sweep.md` |
| `utility.delta` | `25` | `valuation/utility.py::UtilityConfig` | δ = 25 dollars: a product that costs the member more than a hard pull and a fee is harmful | 10–100 | suitability policy | measured: HarmRate@10 by α |
| `utility.harm_penalty` | `25` | `valuation/utility.py::UtilityConfig` | 25 dollars off U when a harmful product cannot be excluded (no safe same-family alternative) | 10–100 | suitability policy | expected: low (rarely triggered) |
| `utility.pending_family_penalty` | `15` | `valuation/utility.py::UtilityConfig` | 15 dollars of NB for a second card / loan application in a family already pending | 0–50 | pending-family policy | expected: low |
| `rqvae_optimizer.name` | `adam` | `training/config.py::RQVAE_OPTIMIZER` | plain Adam (= AdamW with wd 0): decay would contract the residuals under EMA codebooks | adam | nothing | measured: utilization 1.0000 |
| `rqvae_optimizer.lr` | `0.001` | `training/config.py::RQVAE_OPTIMIZER` | 1e-3 constant | 3e-4–3e-3 | input scale | expected: low |
| `rqvae_optimizer.weight_decay` | `0` | `training/config.py::RQVAE_OPTIMIZER` | 0: see `name` | 0 | nothing | n/a |
| `rqvae_optimizer.warmup_steps` | `0` | `training/config.py::RQVAE_OPTIMIZER` | no warmup for a full-batch Adam | 0 | nothing | n/a |
| `rqvae_optimizer.total_steps` | `2000` | `training/config.py::RQVAE_OPTIMIZER` | 2 000 full-catalog steps converge in 3 s; best at step 1900 | 1 000–5 000 | catalog size | measured: `training_summary.md` |
| `rqvae_optimizer.final_lr_fraction` | `1` | `training/config.py::RQVAE_OPTIMIZER` | constant lr (1.0) | 0.1–1 | nothing | expected: low |
| `rqvae_optimizer.grad_clip` | `None` | `training/config.py::RQVAE_OPTIMIZER` | no clipping needed for an MSE objective | None | nothing | n/a |
| `rqvae_optimizer.batch_size` | `0` | `training/config.py::RQVAE_OPTIMIZER` | 0 = the whole catalog per step: 2 000 rows fit trivially and the EMA statistics see every item | 0 | catalog ≫ 10⁵ | n/a |
| `rqvae_optimizer.eval_every` | `100` | `training/config.py::RQVAE_OPTIMIZER` | utilization check every 100 steps | 50–500 | nothing | n/a |
| `rqvae_optimizer.patience` | `3` | `training/config.py::RQVAE_OPTIMIZER` | three evaluations | 2–5 | nothing | n/a |
| `rqvae_optimizer.betas` | `(0.9, 0.98)` | `training/config.py::RQVAE_OPTIMIZER` | (0.9, 0.98): the transformer-standard second-moment decay | (0.9, 0.999) | nothing | expected: low |
| `tiger_optimizer.name` | `adamw` | `training/config.py::TIGER_OPTIMIZER` | AdamW with decoupled decay on matrices only | adamw | nothing | n/a |
| `tiger_optimizer.lr` | `0.001` | `training/config.py::TIGER_OPTIMIZER` | 1e-3 peak | 3e-4–3e-3 | dataset size | expected: moderate |
| `tiger_optimizer.weight_decay` | `0.01` | `training/config.py::TIGER_OPTIMIZER` | 0.01 on weight matrices | 0–0.1 | overfitting (it stopped early) | expected: moderate |
| `tiger_optimizer.warmup_steps` | `500` | `training/config.py::TIGER_OPTIMIZER` | 500 warmup steps before cosine | 100–1 000 | total_steps | expected: low |
| `tiger_optimizer.total_steps` | `3000` | `training/config.py::TIGER_OPTIMIZER` | 3 000-step budget; stopped at 800 with best recall at 200 | 1 000–20 000 | dataset size (more data → longer) | measured: `training_summary.md` |
| `tiger_optimizer.final_lr_fraction` | `0.1` | `training/config.py::TIGER_OPTIMIZER` | cosine to 10 % of peak | 0.01–0.5 | nothing | expected: low |
| `tiger_optimizer.grad_clip` | `1` | `training/config.py::TIGER_OPTIMIZER` | global norm 1.0 | 0.5–5 | nothing | expected: low |
| `tiger_optimizer.batch_size` | `256` | `training/config.py::TIGER_OPTIMIZER` | 256 sequences × ~80 tokens: one CPU-friendly step; each sequence yields ~L next-SID examples | 64–1 024 | memory | expected: low |
| `tiger_optimizer.eval_every` | `200` | `training/config.py::TIGER_OPTIMIZER` | beam-search evaluation on the validation split every 200 steps (it is the expensive evaluation) | 100–500 | evaluation cost | measured: 66 s total |
| `tiger_optimizer.patience` | `3` | `training/config.py::TIGER_OPTIMIZER` | three evaluations | 2–5 | nothing | expected: low |
| `tiger_optimizer.betas` | `(0.9, 0.98)` | `training/config.py::TIGER_OPTIMIZER` | (0.9, 0.98) | (0.9, 0.999) | nothing | expected: low |
| `ranker_optimizer.name` | `adamw` | `training/config.py::RANKER_OPTIMIZER` | AdamW | adamw | nothing | n/a |
| `ranker_optimizer.lr` | `0.001` | `training/config.py::RANKER_OPTIMIZER` | 1e-3 peak | 3e-4–3e-3 | dataset size | expected: moderate |
| `ranker_optimizer.weight_decay` | `0.01` | `training/config.py::RANKER_OPTIMIZER` | 0.01 on matrices | 0–0.1 | overfitting | expected: moderate |
| `ranker_optimizer.warmup_steps` | `500` | `training/config.py::RANKER_OPTIMIZER` | 500 warmup steps | 100–1 000 | total_steps | expected: low |
| `ranker_optimizer.total_steps` | `2000` | `training/config.py::RANKER_OPTIMIZER` | 2 000-step budget; stopped at 700, best at 400 | 1 000–20 000 | dataset size | measured: `training_summary.md` |
| `ranker_optimizer.final_lr_fraction` | `0.1` | `training/config.py::RANKER_OPTIMIZER` | cosine to 10 % | 0.01–0.5 | nothing | expected: low |
| `ranker_optimizer.grad_clip` | `1` | `training/config.py::RANKER_OPTIMIZER` | global norm 1.0 | 0.5–5 | nothing | expected: low |
| `ranker_optimizer.batch_size` | `64` | `training/config.py::RANKER_OPTIMIZER` | 64 slates × K = 100 = 6 400 rows → 1954.85 kept rows with 490.0 clicks, 102.3 applies, 97.3 resolved applications and 81.35 approvals per batch: the masked apply / approve terms and their mask-sum normalization are non-degenerate; memory is 64 × (64 + 100) × 64 per layer, trivial on CPU; at r = 0.25 a smaller batch (16) would leave ~24 resolved applications per step, too few for the approval term | 16–256 slates | CTR, r, K | measured: `positives_per_batch.md` |
| `ranker_optimizer.eval_every` | `100` | `training/config.py::RANKER_OPTIMIZER` | validation loss every 100 steps | 50–500 | nothing | expected: low |
| `ranker_optimizer.patience` | `3` | `training/config.py::RANKER_OPTIMIZER` | three evaluations | 2–5 | nothing | expected: low |
| `ranker_optimizer.betas` | `(0.9, 0.98)` | `training/config.py::RANKER_OPTIMIZER` | (0.9, 0.98) | (0.9, 0.999) | nothing | expected: low |
| `prm_optimizer.name` | `adamw` | `training/config.py::PRM_OPTIMIZER` | AdamW | adamw | nothing | n/a |
| `prm_optimizer.lr` | `0.0005` | `training/config.py::PRM_OPTIMIZER` | 5e-4: a small model on ~8 400 slates | 1e-4–1e-3 | dataset size | expected: low |
| `prm_optimizer.weight_decay` | `0.01` | `training/config.py::PRM_OPTIMIZER` | 0.01 | 0–0.1 | overfitting | expected: low |
| `prm_optimizer.warmup_steps` | `200` | `training/config.py::PRM_OPTIMIZER` | 200 warmup steps | 50–500 | total_steps | expected: low |
| `prm_optimizer.total_steps` | `600` | `training/config.py::PRM_OPTIMIZER` | 600-step budget; stopped at 500, best at 350 | 300–3 000 | dataset size | measured: `training_summary.md` |
| `prm_optimizer.final_lr_fraction` | `0.1` | `training/config.py::PRM_OPTIMIZER` | cosine to 10 % | 0.01–0.5 | nothing | expected: low |
| `prm_optimizer.grad_clip` | `1` | `training/config.py::PRM_OPTIMIZER` | global norm 1.0 | 0.5–5 | nothing | expected: low |
| `prm_optimizer.batch_size` | `128` | `training/config.py::PRM_OPTIMIZER` | 128 slates of 10 slots: listwise loss needs whole slates, and 128 gives ≥ 128 positives per step | 32–512 | slate count | expected: low |
| `prm_optimizer.eval_every` | `50` | `training/config.py::PRM_OPTIMIZER` | every 50 steps (evaluation is cheap) | 25–200 | nothing | n/a |
| `prm_optimizer.patience` | `3` | `training/config.py::PRM_OPTIMIZER` | three evaluations | 2–5 | nothing | expected: low |
| `prm_optimizer.betas` | `(0.9, 0.98)` | `training/config.py::PRM_OPTIMIZER` | (0.9, 0.98) | (0.9, 0.999) | nothing | expected: low |
| `rqvae_min_utilization` | `0.9` | `training/config.py::TrainingConfig` | a checkpoint is only acceptable if every level uses ≥ 90 % of its codes; otherwise the hierarchy degenerates and the trie loses discrimination | 0.5–0.95 | codebook size vs catalog size | measured: 1.0000 on all levels |
| `serving.batch_size` | `1` | `serving/pipeline.py::RecommendationPipeline.run` | B = 1: one member per request; latency is measured at this batch size | 1 (batched serving would change the numbers) | serving architecture | measured: `latency.md` |
## Appendix B — Failure modes and what catches them

| stage | failure mode | what catches it |
|---|---|---|
| Stage 1 | **Stale trie** after RQ-VAE retraining or a catalog update: codes map to different items, the logit mask allows an ineligible leaf | gate ② (`EligibilityEngine.filter_candidates`) removes it and the telemetry shows `num_after_post_filter < num_retrieved`; gate ③ raises `ComplianceViolation` if anything reaches the slate; `tests/test_tiger.py` zero-violation tests on the trie itself |
| Stage 1 | **Codebook collapse** (utilization falls, many items share prefixes, the beam loses coverage) | the RQ-VAE early-stopping criterion refuses checkpoints below `rqvae_min_utilization = 0.9`; utilization per level is reported in `docs/results/training_summary.md` |
| Stage 1 | **Beam starvation** for heavily gated members (few eligible leaves, dead beams) | beam yield is reported per run (`docs/results/retrieval_metrics.md`); dead beams are `−1`, never back-filled from an unmasked source; `test_tiger_empty_eligibility_returns_no_items` |
| Stage 1 | **Latency overrun** from the beam (the measured 16.5931 ms) | the per-stage latency table (`docs/results/latency.md`) names the stage; the KV-cache mitigation is scoped in Section 8 |
| Stage 2 | **Candidate cross-talk** (a mask regression lets candidates attend to each other, scores depend on the beam) | `tests/test_hstu.py::test_candidate_independence_permutation_and_removal` and `test_batched_equals_per_candidate_loop` |
| Stage 2 | **Future leakage** into the user state (a causal-mask regression under left padding) | `test_causality_under_left_padding`, `test_pad_content_does_not_leak`, `test_gradient_of_past_wrt_future_is_zero` |
| Stage 2 | **Feature drift** between training and serving standardization | standardization buffers live inside the model and round-trip with it; `tests/test_pipeline.py::test_artifacts_round_trip_gives_identical_slates` |
| Stage 3 | **Down-sampling rate changed without updating the logit correction** (click probabilities inflated by `1/r`) | the correction is a buffer set from the training rate at train time, not a config read at serve time; `tests/test_funnel_loss.py::test_logit_correction_recovers_true_click_probability`; click ECE in `docs/results/funnel_metrics.md` would jump |
| Stage 3 | **Shift in the delay law** (partners slow down; more applications pending at the cut-off than the policy assumes) | `pending_rate_given_apply` in `docs/results/dataset_summary.md`; the `drop` default does not depend on the law; the ablation in `docs/results/pending_policy_ablation.md` bounds the damage of a wrong policy |
| Stage 3 | **Entire-space term underflow** (a numerics regression re-introduces probability-space products) | `test_finite_loss_and_gradients_at_extreme_logits` at `|z| = 30` |
| Stage 3 | **Degenerate approval term** in a batch with no resolved applications | mask-sum normalization returns zero with a valid gradient; `test_mask_sum_normalization_and_empty_sub_populations`; `docs/results/positives_per_batch.md` shows the expected counts |
| Stage 4 | **Calibration drift** (the base rate moves, `p̂` no longer matches realized rates) | ECE / MCE / Brier per tower and per tier in `docs/results/calibration.md`; calibrators are a separate artifact refitted without retraining |
| Stage 4 | **Mis-set `α`** (a business change pushes `α` past the flat region) | `HarmRate@10` in `docs/results/pareto_sweep.md` rises from 0.000 at `α = 0.7` to 0.160 at `α = 1`; the guardrails still exclude harmful products whenever a safe same-family alternative exists |
| Stage 4 | **Guardrail bypass** by a re-ranker that learns to prefer harmful products | guardrails run before the PRM and excluded slots carry `U = −∞`; `test_guardrails_do_no_harm_refinance_and_pending_family` |
| Stage 4 | **PRM collapsing to one family** (every slot from the most clickable family) | the greedy family penalty `β = 0.5` at inference; `test_prm_features_targets_and_rerank` checks the rerank; family mix of served slates is visible in the e2e script's output |
| Stage 4 | **Probabilities rescaled after re-ranking** (a future change logs PRM scores as odds) | the PRM interface returns an ordering only; `SlateResult` carries the calibrated `p̂` computed before it; the calibration section fixes the order |
| Serving | **Held or pending product served** (a bug anywhere after retrieval) | gate ③ `assert_all_eligible` raises; `tests/test_pipeline.py::test_output_assertion_catches_violations` |
| Serving | **Silent contract change** in the write-up or the register drifting from the code | `tests/test_docs.py` fails when a table number is not in `docs/results/` or an Appendix A value differs from the config default; `scripts/check_diagrams.py` fails when a diagram names a symbol that does not exist |
