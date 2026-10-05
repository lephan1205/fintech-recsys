# A four-stage, compliance-constrained recommender for a credit marketplace

*Generative retrieval → HSTU scoring → multi-task funnel → calibrated valuation and slate re-ranking, with underwriting eligibility enforced three times and honest probabilities as the audited output.*

**Contents:** [1 Abstract](#1-abstract) · [2 Problem setting](#2-problem-setting) · [3 Data contract](#3-data-contract) · [4 Stage 1: generative retrieval](#4-stage-1--generative-retrieval-under-a-compliance-mask) · [5 Stage 2: HSTU scoring](#5-stage-2--hstu-scoring-backbone) · [6 Stage 3: multi-task funnel](#6-stage-3--multi-task-funnel) · [7 Stage 4: calibration, valuation, re-ranking](#7-stage-4--calibration-valuation-re-ranking) · [8 End-to-end serving](#8-end-to-end-serving) · [9 Limitations](#9-limitations-and-future-work) · [10 References](#10-references) · [Appendix: failure modes](#appendix--failure-modes-and-what-catches-them)

---

## 1. Abstract

A credit marketplace recommends financial products (credit cards, balance-transfer cards, personal loans, auto refinance, mortgages) to members whose eligibility is decided by hard underwriting rules, whose funnel (click → apply → approve) is steeply imbalanced, and whose most valuable outcomes arrive days to weeks after the impression. The recommender described here is a four-stage cascade. Stage 1 is generative retrieval: an RQ-VAE turns the 18-dimensional product record into a hierarchical Semantic ID, a TIGER-style decoder generates the IDs of the top-100 candidates by beam search, and a prefix trie masks every code whose continuation cannot end in an eligible product, so ineligible items are never generated. Stage 2 scores the 100 candidates in a single batched pass of an HSTU backbone whose attention mask keeps every candidate independent of the others. Stage 3 is a PLE multi-task funnel producing raw logits for `p(click)`, `p(apply | click)`, `p(approve | apply)` and a zero-inflated log-normal amount, trained with a unified funnel loss whose entire-space terms are written in log space on the same towers, with pending applications handled as a labeling concern. Stage 4 calibrates the logits per tower, converts them to expected partner value and a deterministic net user benefit, combines both under a serving-time weight `α` with suitability guardrails, and re-ranks the survivors with a small transformer whose output is only an ordering.

Headline results on the synthetic benchmark: TIGER beats an eligible-popularity baseline at the top of its list (`Recall@10` 0.0868 vs 0.0548) but trails it at `Recall@100` (0.3379 vs 0.3836); the funnel probabilities are well calibrated (expected calibration error at most 0.0280 across the three towers); and the serving path takes 19.84 ms against a 10 ms target, almost all of it in the retrieval beam. No served slate contained an ineligible product.

**Figure 1 — Serving pipeline.** One request from eligibility mask to final slate, with the three eligibility gates, tensor shapes and per-stage latency (Section 8).

![Serving pipeline](diagrams/serving_pipeline.svg)

**Figure 2 — Training pipeline.** Synthetic generator → observed labels at the training cut-off → per-model training → calibrators → results tables (Sections 3, 6 and 7.1).

![Training pipeline](diagrams/training_pipeline.svg)

## 2. Problem setting

**The marketplace.** A member sees a slate of up to ten products. Clicking opens an offer; applying triggers a hard credit pull and a partner decision; an approval funds a loan or opens a line and pays the platform a partner payout. The platform therefore has two stakeholders whose interests only partly overlap: partners pay for approvals, members want the product that improves their finances. A recommender that maximizes payout alone will push high-payout, high-APR products at members who would be better off with a cheaper one, which is both a trust problem and, for a regulated lender, a suitability problem.

**Hard underwriting.** Every product declares a minimum FICO, a maximum debt-to-income ratio, a minimum income and a set of licensed states. A member outside these bounds is not "unlikely to be approved"; they are ineligible, and showing them the product is a compliance defect, not a ranking error. The same holds for products the member already holds and products with an open application. This is why eligibility is enforced as a hard mask at three points of the pipeline rather than learned as a feature (Section 4.2 and Section 8).

**Delayed, censored outcomes.** Card decisions are mostly instant; personal loans take one to three days; auto refinance three to seven; mortgages weeks. At any training cut-off a fraction of applications is still pending, and the fraction is largest exactly for the family with the largest amounts. Treating a pending application as a decline biases the approval tower downward for mortgages; dropping the row removes signal; inverse-propensity weighting needs the delay law. The pipeline drops pending rows from the approval loss and calibration (`data/delayed_feedback.py`; ablation in `docs/results/pending_policy_ablation.md`).

**Funnel imbalance.** On the benchmark dataset the click-through rate is 0.0762, the apply rate given a click is 0.2016 and the approval rate given an application is 0.8232, so approvals occur on 0.0126 of impressions (`docs/results/dataset_summary.md`). A batch of impressions carries hundreds of clicks, tens of applications and a few dozen resolved approvals; the loss design must keep every term well defined at those counts (Section 6.3).

**Latency budget.** The target is a sub-10 ms CPU scoring budget per member at batch size one. That budget is why the design is a cascade: no scorer that reads the full interaction history can score a 2 000-item catalog at ranking fidelity in that time, and no retriever that runs in that time can score candidates against the full history. The measurements bear this out: scoring 100 candidates takes 2.3920 ms p50, so scoring the whole catalog would need twenty times that, while the retriever never sees a candidate against the full history (Section 8).

**Why a cascade at all.** The four stages are not four models bolted together; each stage exists because a different constraint binds there. Retrieval is where compliance is cheapest to enforce (a mask on a logit vector) and where catalog size is the cost driver. Scoring is where the history matters and where candidate count is the cost driver. The funnel stage is where the labels live and where imbalance and censoring must be handled. Valuation is where business policy lives, and it is kept out of every learned component so that a policy change is a config change, not a retraining. A cascade also lets each concern be tested in isolation: that retrieval never emits an ineligible product, that each candidate's score is independent of the others, that the funnel loss is finite and unbiased where it claims to be, and that the dollar arithmetic of valuation is correct ([RESULTS.md](RESULTS.md), Section 8).

## 3. Data contract

**Schema as the single source of truth.** `data/schema.py` defines the typed records every stage consumes: products (eligibility gates, economics, family), members (credit attributes, held and pending products, and the financial state the benefit formulas read), one interaction timeline per member, and impression slates (the served candidates with their funnel labels, serve time and the partner's decision delay). The same module bundles them with dense product and member feature matrices. The data generation, the batching code, the models and the evaluation all import these types, so no stage re-derives a feature from raw fields.

**The reserved-index rule.** Item id 0 is reserved. It is the padding index of every embedding table, and it is also the item id of credit-score-change events, which belong to the member rather than to a product. Attention masks are therefore derived from the action type, never from the item id: a mask built from item ids would silently drop every score-change event. Tests pin both halves of the rule. Product 0 is never eligible, so no model can generate or score the padding item.

**Left padding and the user state.** Sequences are left-padded, so the most recent real event is always in the last position. Every sequential model reads the member's state from that position without a per-row lookup, and recency lines up across the batch. A test confirms that left and right padding give the same representation. This alignment is what lets HSTU place its candidate tokens at "now" without knowing where each member's history starts (Section 5.2).

**Observed view at the cut-off.** Every application has a partner decision delay, and each slate stores both the *oracle* (true) outcome and that delay. The models only ever see the *observed* view at the training cut-off: each application is classified as not applied, approved, declined or still pending as of that date. On pending rows the approval label and loan amount are zeroed in the training data, while the oracle values are kept in separate fields that only the evaluation reads. A test asserts that the outcome of a pending application never leaks into training. Member histories follow the same rule: a pending application appears as an event without its outcome.

**Benchmark size.** Two thousand products and three thousand members, split by member into 2100 / 300 / 300 / 300 for train / validation / calibration / test. Members, not events, are split so that no member's behaviour appears on both sides of any boundary. The calibration split is used only to fit calibrators, so it is disjoint from everything the models were trained or selected on.

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

**Choice.** Each product's standardized 18-dimensional feature vector (gate thresholds, APR, rewards, fees, family one-hot, economics) is encoded by `layers/rq_vae.py::RQVAE` (Lee et al., 2022) into a 16-dimensional latent and residually quantized at three levels with a codebook of 32 codes per level; a fourth disambiguation level makes every ID unique (`RQVAE.assign_semantic_ids`, `tests/test_tiger.py::test_semantic_ids_unique_after_disambiguation`). The result is a coarse-to-fine tuple in which products with similar underwriting and economics share prefixes.

**Alternatives.** A flat-ID retriever (SASRec with a softmax over the catalog, Kang and McAuley, 2018; or a two-tower model with MIPS) was considered. Eligibility can be enforced there too, by masking logits or using filtered nearest-neighbour search, so compliance does not decide the choice. The deciding issue is cold start: a flat-ID model learns one embedding per item, so a newly launched product cannot be retrieved until it has accumulated interactions and the model is retrained, and product launches are frequent in this marketplace. A Semantic ID is computed from the product's features by the RQ-VAE, so a new product lands under the prefixes of similar existing products and is retrievable as soon as it is added to the trie. The costs are a sequential beam (Section 8) and weaker tail recall ([RESULTS.md](RESULTS.md), Section 6).

**Why depth 3 and codebook 32.** The code space has `32³ = 32,768` prefixes for a catalog of 2,000 items, so collisions are rare before disambiguation and the fourth level is short; a shallower tree would put many items under one prefix and weaken the trie's ability to say "no eligible leaf here", a deeper one would lengthen the beam (each level is one forward pass over `B·W` rows; Section 8). 

**Collision handling.** The RQ-VAE learns only three codebooks; the fourth token of a Semantic ID is not learned. It is a counter: products that quantize to the same three codes are numbered 0, 1, 2, … in catalog order, so `(7, 19, 3)` shared by two cards becomes `(7, 19, 3, 0)` and `(7, 19, 3, 1)`. The trie's leaves are therefore unique and the `item_for` round trip is exact. This still leaves a subtler failure, discussed next: a prefix can be allowed because *some* leaf under it is eligible even though the leaf finally decoded is not.

### 4.2 The prefix trie as a compliance mechanism

**Choice.** All Semantic IDs in the catalog are stored in a prefix trie, built once. For each request, the member's eligible products define which branches are open: at every decoding step, a code is allowed only if at least one eligible product lies beneath it, and every other code's logit is set to `−∞`. The beam therefore cannot generate an ineligible product. The tests make this adversarial: a model deliberately trained to prefer a product the member already holds or has pending still never emits it ([RESULTS.md](RESULTS.md), Section 8).

**Why masking at generation beats post-filtering.** Beam search keeps a fixed number of partial IDs alive at each step. If ineligible products were removed only after generation, part of that fixed budget would be spent on candidates that are later thrown away, and for a heavily gated member, eligible for only a few dozen products, most of the returned list could disappear. Masking during decoding prunes ineligible branches before they compete for a beam slot, so the whole beam is spent on products the member can actually be offered.

**Why the trie alone is insufficient: the three enforcement points.** The pipeline checks eligibility three times, and each check catches something the previous one cannot.

1. *The trie mask* (gate ①) stops ineligible products from being generated. But it is only correct if the trie, the Semantic IDs and the catalog are all in sync. Retraining the RQ-VAE, or adding a product that shifts the counter tokens, can leave a stale trie mapping IDs to the wrong products. That is too many moving parts to be the only defence.
2. *The post-retrieval filter* (gate ②) re-checks eligibility on the returned product ids. It knows nothing about Semantic IDs, so it still works if TIGER is replaced by another retriever, a fallback is added, or results are served from a cache. On the benchmark it never removes anything, which is evidence that gate ① works; it exists for the day that stops being true.
3. *The output assertion* (gate ③) checks the final slate and raises an error if any product is ineligible. It covers every stage after retrieval, so a bug in valuation, guardrails or re-ranking that brings back a filtered product cannot reach the member.

**What each gate masks.** The eligibility engine builds a true/false grid with one row per member and one column per product, where a cell is true if that member may be offered that product. Each of the four hard rules (FICO floor, DTI ceiling, income floor, licensed state) fills the whole grid in a single array comparison: a member with FICO 720 gets true for a card with a 650 floor and false for a mortgage with a 740 floor. A cell stays true only if all four rules pass. The engine then removes held products, products with an open application, and whole families under the pending-family policy: an open mortgage or auto-refinance application masks the whole family, while a second card or personal-loan application is legitimate and is only penalized at valuation (Section 7.4).

### 4.3 The TIGER decoder and beam search

**Choice.** `models/tiger/model.py::TIGER` (Rajput et al., 2023) is a decoder-only transformer that writes the next product's Semantic ID one code at a time, like a language model writing a four-word sentence. History items and the item being generated use the same Semantic-ID vocabulary, so one causal stack reads both: this needs one embedding table, and generation is a plain next-token loop. (The original TIGER is an encoder-decoder. At four target tokens a separate encoder buys nothing.) The model is small: `d = 64`, two layers, two heads, feed-forward width 128.

**Input: from events to vectors.** A member's last 20 events become a sequence of 82 tokens (`SemanticIdTokenizer.encode_history`). Each event contributes its product's four Semantic-ID codes. Two member-context tokens, credit tier and state, go in front. Missing history is left-padded:

```
token:   TIER  STATE  PAD … PAD   c₁ c₂ c₃ c₄   c₁ c₂ c₃ c₄   …   c₁ c₂ c₃ c₄
action:   –      –     –  …  –    VIEW ×4       APPLY_APPROVED ×4  …   CREDIT_PULL ×4
         └ context ┘  └ padding ┘ └ event 1 ┘   └── event 2 ───┘      └ event 20 ┘
```

The vocabulary has 159 tokens: PAD, one token for a credit-score change (an event with no product, written as four copies), 5 tier and 51 state tokens, and each code level in its own range (32 + 32 + 32 + 5). Code 7 at level 1 and code 7 at level 2 are therefore different tokens. Each token's input vector is the sum of three learned 64-dimensional embeddings: `tok_emb[token] + action_emb[action] + pos_emb[position]`. The action is what the member did (view, credit pull, approved / declined / pending application, score change). It is copied onto all four code tokens of its event, so the model sees what kind of product was involved and what happened with it. The position counts real tokens only, so the tier token is always position 0 however much padding precedes it. Context tokens, and the codes added during generation, carry the PAD action, whose embedding is fixed at zero. Padding slots are zeroed out entirely. Time gaps between events are not used here; only the Stage 2 scorer reads them.

**What the transformer produces.** Two pre-norm blocks (layer norm → causal self-attention → residual; layer norm → feed-forward → residual) and a final layer norm turn the input into one 64-dimensional hidden state per position. The code calls the class `TransformerEncoder`, but the causal mask makes it a decoder: token `i` attends only to itself and earlier real tokens, never to padding. Hidden state `hᵢ` is therefore a summary of the member's context and history up to token `i`. A linear head maps each `hᵢ` to 159 logits, a guess at token `i + 1`. When the next token is a level-`k` code, only that level's slice of the logits is soft-maxed (32 codes, or 5 at level 4), so the model never spends probability on a token of the wrong kind.

**Training: teacher-forced next-ID prediction.** Each training record pairs a history with the member's next positive event (a view, credit pull, or approved or pending application) as the target item, with Semantic ID `(c₁, c₂, c₃, c₄)`. The first three target codes are appended to the history, and one forward pass predicts all four at once:

```
position:   … last history token │ c₁  │ c₂  │ c₃
predicts:            c₁          │ c₂  │ c₃  │ c₄
```

The target loss is the mean of the four cross-entropies. Feeding the true codes rather than the model's own guesses is teacher forcing. It lets one pass train every level, and it reproduces exactly the positions decoding reads at inference. The same pass also scores the history itself: the last token of event `j` predicts `c₁` of event `j + 1`, and each code of event `j + 1` predicts the next one. Score-change events are skipped as targets but kept as context. This turns one record into roughly `L` next-item examples at no extra forward cost. The two losses are averaged separately and then added, so the single target item counts as much as all the history predictions together. Training never sees the eligibility mask; compliance is enforced only at decoding.

**Inference: trie-constrained beam search.** `TIGER.generate` builds the ID level by level, keeping the `W = 100` best partial IDs (beams):

1. *Level 1.* One forward pass over the history. The hidden state at the last real token gives log-probabilities over the 32 level-1 codes. The trie (Section 4.2) sets to `−∞` every code with no eligible product beneath it, and the surviving codes become the first beams.
2. *Levels 2–4.* Each beam's codes so far are appended to the history, and all `B·W` sequences are run in one batched forward pass. The last position gives log-probabilities for the next level. These are masked by the trie for that beam's prefix and added to the beam's running score. The best 100 of all (beam, code) pairs survive.
3. *Output.* After level 4, each full ID is looked up in the trie to recover its product. A beam's score is `log p(c₁|h) + log p(c₂|h, c₁) + log p(c₃|h, c₁, c₂) + log p(c₄|h, c₁, c₂, c₃)`, the model's log-probability of that product as the member's next item.

Because relevance is built from prefixes, the beam commits early to a few coarse regions of the catalog (level 1 roughly separates family and underwriting profile) and spends its width refining them. On the test split this makes TIGER more precise than an eligible-popularity list at the top of its output and thinner further down ([RESULTS.md](RESULTS.md), Section 6), which is why a scorer follows it rather than TIGER being used alone. Beams that reach a prefix with no allowed continuation are marked `−1` and reported as beam yield rather than back-filled, because a back-fill would be a second, unmasked retrieval source. The loop has no key/value cache, so every level re-reads the full 82-token history for every beam; that is the whole latency overrun in Section 8.

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

**Relative position and time biases.** No absolute positions are embedded. Each head carries two learned biases added to every attention score: one over relative position in `[−(L+K), L+K]`, and one over the real time gap between the two events. Position alone cannot tell three card views in one afternoon from three spread over a year, and for credit intent that difference is most of the signal. Gaps are bucketed on a log scale up to 365 days, so each bucket is about 21% wider than the one before: the finest is about five hours and roughly a third of the buckets fall in the first week. This spends resolution where it matters (same session versus next day versus next week, not day 200 versus day 201) and keeps every bucket populated, since behavioural gaps are heavily skewed toward short intervals. The count of 32 buckets is a common default rather than a tuned value. This is what lets the candidate tokens described next sit "at now" without an absolute index: they are one relative position after the last real history token and at zero time offset from it.

**Embedding.** Each past event becomes one token, the sum of three learned vectors: which product it involved, what the member did (view, credit pull, application, and so on), and how long after their previous event it happened. A credit-score change has no product, so its token comes from the action and timing alone. Each candidate product is appended as one more token, built from the same product vectors plus a marker that says "being scored", and placed at the moment of the request.

### 5.2 M-FALCON: scoring all candidates in one pass

**How it works.** Instead of running the backbone once per candidate, M-FALCON appends all 100 retrieved candidates to the member's history as extra tokens and runs one forward pass over the combined sequence of length `L + K`. An attention mask (`layers/hstu.py::build_mfalcon_mask`) controls what each token can see:

- each history event sees itself and earlier events only;
- each candidate sees the whole history and itself;
- candidates never see each other, and history never sees candidates.

When a candidate attends to the history, its query asks which of this member's past events are relevant to that product, so different candidates pull out different parts of the same history: a travel card picks up last year's card browsing, a personal loan picks up last week's credit pull. The output at each candidate token is that product-specific reading, `h_cand`, and the output at the last history event is the member's general representation, `h_user`.

**Why candidates must not see each other.** In attention, each token's output is built only from the tokens it is allowed to see. Because a candidate sees just the history and itself, its score is a function of the member and that product alone. If candidates could see each other, the score would also depend on which other 99 products the retriever happened to return, and that breaks two things. Calibration (Section 7.1) maps raw scores to probabilities on the assumption that a score means the same thing every time; scores that move with the candidate list make it fit a moving target. And decisions could no longer be explained: in credit, "the member was shown this offer because of their history and its terms" is an acceptable reason, "because the retriever also returned 99 loans" is not.

### 5.3 User–candidate fusion

`HSTUBackbone.fusion` builds one input vector per candidate for the PLE multi-task funnel (Section 6) by concatenating four blocks, `[h_cand ‖ h_user ‖ h_cand ⊙ h_user ‖ tabular]`:

- `h_cand`: how this product relates to what the member has done, the candidate's own reading of the history (Section 5.2).
- `h_user`: who this member is overall, behaviourally, the state at their most recent event.
- `h_cand ⊙ h_user`: how well the two match, dimension by dimension. An MLP learns multiplicative interactions poorly from concatenated inputs, so the element-wise product is supplied directly; PLE's first layer can then form any weighted dot product of the two.
- `tabular`: the hard facts the sequence does not carry, the member's 26 features (credit profile, income) and the product's 18 (terms, fees, thresholds). Approval depends mostly on these.

The fusion vector is the only input to PLE; nothing about business value enters it.

**Alternative considered: the hybrid predecessor.** The design this replaces was a TransAct-style short-window transformer plus a PinnerFormer-style long-term embedding fused by a DCN-v2 cross network. Its three components each modelled part of what one HSTU layer does (recent intensity, long-term taste, feature crosses), and the DCN-v2 tower was the only place the candidate met the history, so candidate scores were not conditioned on the sequence itself. HSTU with candidate tokens gives candidate-conditioned sequence attention, relative-time awareness and the pointwise intensity signal in one module, with a single mask whose properties are testable.

## 6. Stage 3 — Multi-task funnel

### 6.1 PLE on top of the fusion vector

**Choice.** `models/ple/model.py::PLE` (Tang et al., 2020) stacks two Customized-Gate-Control blocks (`CGCBlock`) over the 236-dimensional fusion vector. Each level holds six experts, every one a 236 → 64 → 32 MLP (`Expert`): one private expert per task (click, apply, approve, amount) and two shared. A task's gate is a softmax over its own expert and the two shared ones, so task-specific parameters are shielded from other tasks' gradients while shared knowledge still flows; the shared gate mixes all six and feeds the next level. Four towers (32 → 1 for the three probabilities, 32 → 3 for the ZILN amount head, `losses/ziln_loss.py::ZILNHead`) emit raw logits.

**Why amount is a task.** The fourth tower is a regression head, not a probability. Its label is the approved credit limit or funded loan size, observed on every resolved approved application, so it is supervised on those rows only (`L_amount` in Section 6.3) and contributes nothing elsewhere. It lives inside PLE because it conditions on the same member-product match as the funnel probabilities. Its private expert holds what is specific to amount. It is consumed only in Stage 4, where the net-user-benefit formulas of Section 7.2 use it to compute the interest saving on a balance transfer or a personal-loan consolidation; the three probabilities never read it.

**Why one private expert and two shared.** One private expert is enough to isolate the seesaw: its parameters receive gradient from its own task alone, and a second would double the private parameters for no identified gain. Two shared experts rather than one give each task gate a choice of *which* shared representation to draw on, not only *how much*, so the shared experts can specialize under gradient from different tasks and click can lean on a different mixture than approve.

**Alternatives and the seesaw argument.** Two families were rejected. *Four separate models, each with its own HSTU backbone,* would run the expensive part of the ranker, the attention pass over the 82-token history plus 100 candidates, four times on the same input instead of once. They would also break the funnel product: `p(click) · p(apply | click) · p(approve | apply)` is only coherent when all three towers read the same member state, and the entire-space terms of Section 6.3 train the towers jointly through that product in any case. *One shared model* removes both costs but replaces them with the seesaw. In its plain form, a shared bottom with four heads on one HSTU trunk, every task's gradient lands on the same trunk: a flashy high-APR card is clicky but rarely approved, so improving the approval tower degrades the click tower and vice versa. Its gated form, MMoE, softens this with one gate per task, but every expert is still shared, so a click-hungry gradient can overwrite the expert the approval tower depends on. PLE keeps the one backbone and one forward pass of the shared bottom and adds the private experts MMoE lacks: it shares where sharing helps and isolates where it hurts. Sharing also matters for data: apply is observed only on clicks, approve only on applications and amount only on approvals, so the deep-funnel towers borrow the member-product representation learned from abundant click labels through the shared experts, while their private experts hold what is specific to them. 

### 6.2 What the towers output and who consumes it

The three probability towers output raw logits, `z₁ = logit p(click)`, `z₂ = logit p(apply | click)` and `z₃ = logit p(approve | apply)`. Stage 4 calibrates each one per tower (Section 7.1), multiplies the calibrated probabilities into `P_funded = p̂₁p̂₂p̂₃`, and turns that into dollars: expected partner value, the funnel factor on net user benefit, and the utility the slate is ranked by (Sections 7.2 and 7.3). The amount tower outputs `[π_logit, μ, σ_raw]`, the raw parameters of a zero-inflated log-normal over the dollar amount (Section 7.2): `π` is the probability the amount is nonzero, and `μ` and `σ` are the mean and spread of its logarithm when it is. Its only consumer is the valuation stage, which collapses the three into `E[amount] = π · exp(μ + σ²/2)` for the net-user-benefit formulas of Section 7.2. The amount parameters are never calibrated, never enter the funnel probabilities, and are not read by the re-ranker.

### 6.3 The Unified Funnel Loss

**What it is.** One training objective for the whole ranker (`losses/funnel_loss.py::UnifiedFunnelLoss`): a weighted sum of six terms, one per tower plus two that tie the towers together,

```
L = L_click + L_apply + L_approve + L_ctcvr + L_ctcavr + 0.1 · L_amount
```

Notice that `L_amount` is scaled by 0.1. It is the one term trained on a small, heavy-tailed sample, about 80 approved rows per batch against roughly 2,000 impressions for click, so its gradient is noisy. The shared experts receive gradient from every task, so at full weight this one noisy term would move them more than the three probability terms combined. The 0.1 is a hand-set constant that damps that term to the same order of influence as the others; it was not learned or swept, and the [decision register](DECISION_REGISTER.md) gives its plausible range. The loss is *unified* because all six terms act on the same four towers; there is no second network.

**The six terms.** Notation: `BCE(z, y)` is the binary cross-entropy from a logit and `BCE_log(ℓ, y) = −[y·ℓ + (1−y)·log1mexp(ℓ)]` the same cross-entropy from a log-probability `ℓ`; `w` is 1 on a resolved application and 0 on a pending one, and `w′` is the same weight extended with 1 on non-applied rows.

| term | sample space | definition |
|---|---|---|
| `L_click` | all impressions | `BCE(z₁, y_click)` |
| `L_apply` | clicked impressions | `BCE(z₂, y_apply)` |
| `L_approve` | applied ∧ observed | `w · BCE(z₃, y_approve)` |
| `L_ctcvr` | all impressions | `BCE_log(s₁ + s₂, y_apply)` |
| `L_ctcavr` | all impressions | `w′ · BCE_log(s₁ + s₂ + s₃, y_approve)` |
| `L_amount` | applied ∧ observed ∧ approved | ZILN negative log-likelihood |

where `z_k` is the raw logit of tower `k` from Section 6.2 and `s_k = logsigmoid(z_k) = log p_k` is its log-probability.

**Why `L_ctcvr` and `L_ctcavr`.** The apply and approve towers are trained only where their labels exist, on clicked and on applied rows, but at serving time they score every retrieved candidate. That is sample-selection bias: `p(apply | click)` is learned on the clicky products and extrapolated to the rest. Following ESMM (Entire Space Multi-Task Model, Ma et al., 2018), the two extra terms supervise the funnel *products* on all impressions, where every row is an unbiased sample with a label for "was applied for" and "was approved". Every tower already produces a logit for every impression, since all four sit on the same fusion vector; what differs between loss terms is the mask that selects which rows contribute. `L_apply` is averaged over clicked rows only, because "applied given clicked" is undefined elsewhere. `L_ctcvr` is averaged over all rows: its prediction is `p₁p₂` and its label is "did this impression end in an application", which is a true 0 on every non-clicked row. That label exists on the full set of served impressions with no click, which is what *entire space* means, and because the `p₂` in the product comes from the apply tower, the apply tower is updated by every impression. `L_ctcavr` does the same for the approve tower with the label "did this impression end in an approval".

The funnel products are formulated in log space for two numerical reasons:
1. **No single logit:** Standard `BCEWithLogitsLoss` cannot be used directly because a product of sigmoids `σ(z₁)σ(z₂)` is not a sigmoid of any single combined logit.
2. **Gradient safety:** Multiplying raw probabilities `p₁p₂` in `float32` underflows to zero when logits are negative, causing `NaN`s or zero gradients on false negatives (`y = 1`) where strong gradient correction is needed.

Adding log-probabilities multiplies probabilities stably (`s₁ + s₂ = log p₁p₂` and `s₁ + s₂ + s₃ = log p₁p₂p₃`), and `BCE_log` computes the negative branch `log(1 − exp(ℓ))` using `log1mexp` (Mächler, 2012). `tests/test_funnel_loss.py` verifies agreement with naive products on moderate logits and numerical stability at extreme logits (`|z| = 30`).

**Imbalance handling.** *Negative down-sampling with logit correction* (`training/downsampling.py::NegativeDownsampler`, `models/ranker.py::HSTUPLERanker.predict`). All clicked impressions are kept, while non-clicked impressions are downsampled with retention rate $r = 0.25$. The resulting boolean mask zeroes out dropped rows across every loss term (including entire-space terms), reducing training compute while maintaining static tensor dimensions. Downsampling inflates the empirical odds learned by the click tower by $1/r$; the true serving probability is recovered at inference via an exact logit shift:
  $$z_{\text{true}} = z_{\text{train}} + \log r$$
  Applying $r$ as a loss weight instead would alter the optimization target to a weighted empirical distribution rather than the true serving distribution, and would require evaluating gradients over all negatives.


## 7. Stage 4 — Calibration, valuation, re-ranking

### 7.1 Calibration before valuation, never after PRM

**Placement.** `serving/stages.py::ValuationStage` fixes the order `raw logits → calibrate → P_funded / EV / E[amount] → NB → U + guardrails`, and the PRM comes after. Calibration must precede valuation because valuation multiplies probabilities into dollars: `EV = p̂₁p̂₂p̂₃ · payout` is only meaningful if each factor is a probability, and a ranking score with the right order but the wrong scale gives dollar figures that are wrong by an unknown factor per tower. Nothing is calibrated after the PRM because the PRM outputs an ordering, not probabilities: the `p̂` that are logged, shown as approval odds and audited are the calibrated ones computed before it, so a member can be told "your approval odds are 0.79" and the number is the one the model was scored on.

**Isotonic by default, Platt as the low-data fallback.** Each tower is calibrated with isotonic regression (Zadrozny and Elkan, 2002; `serving/calibration.py::IsotonicCalibrator`) when it has at least 500 weighted positives, and with Platt scaling (Platt, 1999; `PlattCalibrator`) below that. Isotonic sorts rows by raw score, groups neighbouring rows, and outputs each group's observed positive rate, so its curve is a staircase that can take any shape as long as it never goes down. With thousands of positives the steps are small and follow the true curve; with only a few dozen, each step is set by where a handful of positives happened to land, so the curve copies the noise of that one sample, jumping abruptly or sitting flat at 0 or 1. Platt fits a smooth S-curve (`σ(a·z + b)` on the logit) with only two parameters, which cannot follow noise, so it is the safer choice on small data. 

**Choosing between them.** Which calibrator a tower gets is controlled by one setting, `calibration.method`, with three options: `isotonic` (the default, which applies the 500-positive rule above), `platt` (Platt on every tower), and `auto`. The 500-positive threshold is only a rule of thumb, and `auto` replaces it with a test: each tower sets aside 20% of its calibration rows, fits both calibrators on the other 80%, keeps whichever predicts the set-aside rows better by weighted negative log-likelihood (NLL), and refits the winner on all the rows.

### 7.2 Valuation: what an offer is worth in dollars

Valuation turns each candidate's calibrated probabilities and its amount prediction into three dollar figures: what the partner pays, how much credit is involved, and what the member gains. The re-ranker never sees raw model scores, only these figures and the utility built from them (Section 7.3).

**Expected partner value.** `valuation/expected_value.py` computes `P_funded = p̂₁ · p̂₂ · p̂₃`, the probability that an impression ends in a funded product, and `EV = P_funded · partner_payout`. The partner pays only at the end of the funnel, so the payout is discounted by the chance of getting through every step. This multiplication is why each factor must be a calibrated probability (Section 7.1).

**Expected amount from the ZILN tower.** Approved credit limits and funded loan sizes are zero for most impressions and heavy-tailed when positive. `losses/ziln_loss.py::ziln_loss` (Wang et al., 2019) models the amount as a mixture: zero with probability `1 − π`, otherwise `LogNormal(μ, σ)`, with `L = BCE(π, 1[y > 0]) + 1[y > 0](log σ + log y + (log y − μ)² / 2σ²)` and `E[Y] = π · exp(μ + σ²/2)` (`ziln_expected_value`, exponent clamped). The tower is trained in Stage 3, on resolved approved rows only (`m_amt` in the loss), because that is where the label lives, but it is consumed only in Stage 4: the benefit formulas need `E[amount]` to size a balance transfer or a loan, while the three probabilities never depend on it.

**`NB` is a deterministic formula** (`valuation/user_benefit.py::net_user_benefit`), per family over a horizon `H = 2` years (mortgages `H_hold = 7`): a balance-transfer card saves `B · r_rev · m_intro/12` minus the transfer fee and the annual fee over the intro period with `B = min(revolving_balance, E[amount])`; a credit card earns rewards on annual spend plus the sign-up bonus minus annual fees; a personal loan saves the APR difference on `A = min(E[amount], other_debt)` over the term minus origination; auto refinance and mortgage refinance take the payment difference over the comparison window *minus the difference in principal still owed at the end of it* minus fees and closing costs. 

**Why `NB` is not a learned tower.** A learned benefit would have no ground truth to learn from (nobody observes the counterfactual savings), would entangle a business judgement with behavioural probabilities, and would make "why was this recommended?" unanswerable, whereas the formulas yield "this saves you $X over two years".

### 7.3 Utility and the trade-off weight `α`

`valuation/utility.py::utility` combines the partner's value and the member's value into one score per candidate:

```
U_i = P_funded,i · ( α · payout_i + (1 − α) · NB_i )          α ∈ [0, 1], dollars
```

Both terms are multiplied by the same funnel probability because the member only realizes the benefit if approved and funded; both are in dollars, so `α = 0.5` values a dollar of payout the same as a dollar of member savings, `α = 1` recovers `U = EV`, and `α = 0` ranks purely by expected member benefit.

**Why `α` lives at valuation time and not in a loss.** Placing this inside a loss would cost two things: *retraining to change a business weight* (a policy decision made by a committee should be a config change reviewed in minutes, not a training run); and *unexplainable recommendations* (if `α` is built into the model, each score is a single number and nobody can tell how much of it came from the partner's payout and how much from the member's benefit; kept separate, every recommendation can be explained as "expected payout $X, expected member saving $Y", which is the answer a regulator asks for). At valuation time the decomposition is exact, and the sweep below is an offline evaluation rather than a retraining.

**Choosing `α`.** Because `α` is applied only at valuation, its effect can be seen by re-scoring the same slates at several values and comparing partner revenue, member benefit and harm rate at each. The table below is one such sweep on the test split (top 10 by utility, PRM excluded so only `α` changes); it illustrates the method rather than fixing the value.

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

The pattern is what to look for: up to about `α = 0.7`, revenue rises while member benefit barely moves and no harmful product is shown; beyond it, benefit drops sharply and harmful products appear. The default `α = 0.5` sits inside the flat region.

### 7.4 Suitability guardrails between utility and PRM

`valuation/utility.py::apply_guardrails` enforces three hard rules on the valued candidates before anything learned sees them: *do-no-harm* treats a candidate as harmful if it would cost the member more than 25 dollars (`NB < −25`); it is removed if the slate has a same-family product that does not cost the member (`NB ≥ 0`), and otherwise kept but with its utility lowered by 25 dollars, so the member is not left without any option in that family; *refinance sanity* excludes auto or mortgage refinance unless the new APR is below the member's current rate; and the *pending-family policy* lowers `NB` by 15 dollars for every product in a family where the member already has an open application, reflecting the cost of a second application (this applies to cards and personal loans; for mortgages and auto refinance a second application is not sensible, so those families were already removed by the eligibility engine). Excluded candidates get `U = −∞`.

**Placement.** The rules sit after utility (they need `NB`) and before the PRM because hard rules must not be learnable away: a re-ranker trained on click labels will happily learn that a harmful high-APR product gets clicks. The PRM only ever sees survivors, so no amount of training can reintroduce an excluded product, and gate ③ would catch it if one did. 

### 7.5 PRM re-ranking with a family-cannibalization penalty

**Choice.** The PRM (Pei et al., 2019; `models/prm/model.py::PRM`) is a small one-layer transformer over the ten candidates with the highest utility. For each candidate it sees the ranker's representation of the product, the calibrated probabilities, `EV`, `NB` and `U`, the product family, and the candidate's utility rank, together with the member's profile. It is trained on whole slates to put the products most likely to be clicked first, or, with `prm_target = "utility"`, to follow utility.

**Why a listwise slate model.** Pointwise scores treat each candidate in isolation; a slate is not a set of independent decisions. Two balance-transfer cards side by side split the same click and a flashy card next to a mortgage changes how the mortgage looks. Self-attention over the ten slots conditions every score on the whole slate.

**Why the cannibalization penalty is applied at inference and not in the loss.** `rerank` builds the slate one position at a time: it places the candidate with the highest PRM score, lowers the PRM score of every remaining candidate in the same family by `β = 0.5` (a score unit, not dollars), and repeats, so a second product from a family already shown must beat the alternatives by a margin. The value 0.5 is a hand-set starting point, not a tuned one. It can be tuned the same way as `α` (Section 7.3): rerank the validation slates with several values of `β` and compare, for each value, how many product families appear in the top positions against click NDCG (how close to the top the clicked products sit), expected revenue and expected member benefit. The chosen value is the smallest one that gives most of the diversity gain without lowering those metrics by more than a preset tolerance or raising the harm rate (the share of shown products with `NB < 0`, i.e. that would cost the member money). The click data only shows what members clicked in the order they were originally shown, so it cannot tell whether a product moved higher would have been clicked; the final value should be confirmed with a live A/B test. This is a serving policy about diversity, and it is kept out of the loss so it can be tuned without retraining and so that the PRM's scores remain a pure estimate of slate-conditioned relevance.


## 8. End-to-end serving

**Order.** `serving/pipeline.py::RecommendationPipeline.run` executes, per member: eligibility mask → TIGER beam over the trie with that mask → post-retrieval gate → one HSTU + PLE scoring pass → per-tower calibration → valuation and guardrails → PRM re-ranking of the top 10 by utility → output assertion. This is the serving pipeline of Figure 1.

**Precomputed versus per request.** Precomputed and frozen at serve time: the Semantic IDs and the trie, the catalog feature matrix and product economics, the calibrators, the tabular standardization buffers and the model weights. Computed per request: the eligibility mask, the beam, the member's sequence encoding and candidate scoring, valuation, guardrails and re-ranking. The user's financial state is read from the same profile the generator wrote, so no feature is inverted at serving time.

**Latency.** The serving path costs 19.84 ms p50 on CPU against a 10 ms budget, and almost all of it is the TIGER beam (16.5931 ms): the decoder re-runs the full history at each of the four levels for 100 beams. The compliance mask is not the cost, and scoring, valuation and re-ranking together take under 3.3 ms, which supports the cascade argument of Section 2: the expensive component is the one that must run before the candidate count is known. The main fix is a KV cache, so each level processes only the newly appended code tokens; a smaller beam for members with few eligible products and request batching are further options. None changes the design. The per-stage breakdown is in [RESULTS.md](RESULTS.md) (Section 7) and the failure mode in the Appendix.

## 9. Limitations and future work

**Synthetic data.** The generator has no position bias (a real slate's first slot is clicked more, and the click tower would need a position feature or a debiased label), no seasonality, no competition between concurrent slates and no partner-side drift. Its labels are Bernoulli draws from smooth functions, which makes the oracle ceiling computable but also makes the click task nearly unlearnable (ceiling AUC 0.6316); a real click signal is both noisier and more structured.

**Partner decision delays are taken as known.** One way to handle applications still pending at the cut-off is to keep only the decided ones and give extra weight to those that were unlikely to have been decided by then, so that slow-deciding products are not under-represented (inverse-propensity weighting, the `ipw` option in the pending-policy comparison). That requires knowing how long partners take to decide, and here it is read from the data generator; in production it would have to be estimated from past applications in each product family.

**Retrieval quality and latency.** TIGER trails eligible-popularity at `Recall@100` and the beam is the whole latency overrun. A KV-cached beam, a popularity-aware Semantic-ID prior and a longer training budget with a larger dataset are the three obvious next steps, in that order.

**Calibration with few positives.** The approval tower's calibration split is small enough that neither isotonic nor Platt improves on the raw sigmoid; `method = "auto"` or a larger calibration split is the fix.

**Online learning and fairness.** Everything here is offline. An online loop would need the logged calibrated `p̂` (which is why they are logged) for off-policy evaluation, and a fairness audit would extend the per-tier slices to protected classes with the same machinery.

## 10. References

- Rajput, S. et al. (2023). *Recommender Systems with Generative Retrieval* (TIGER). NeurIPS.
- Lee, D. et al. (2022). *Autoregressive Image Generation using Residual Quantization* (RQ-VAE). CVPR.
- Zhai, J. et al. (2024). *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations* (HSTU, M-FALCON). ICML.
- Tang, H. et al. (2020). *Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations*. RecSys.
- Ma, X. et al. (2018). *Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate* (ESMM). SIGIR.
- Wang, X., Liu, L. and Miao, N. (2019). *A Deep Probabilistic Model for Customer Lifetime Value Prediction* (ZILN). arXiv:1912.07753.
- Pei, C. et al. (2019). *Personalized Re-ranking for Recommendation* (PRM). RecSys.
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*.
- Zadrozny, B. and Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass Probability Estimates*. KDD.
- Mächler, M. (2012). *Accurately Computing log(1 − exp(−|a|))*. CRAN vignette.
- Kang, W.-C. and McAuley, J. (2018). *Self-Attentive Sequential Recommendation* (SASRec). ICDM.

## Appendix — Failure modes and what catches them

| stage | failure mode | what catches it |
|---|---|---|
| Stage 1 | **Stale trie** after RQ-VAE retraining or a catalog update: Semantic IDs point to the wrong products, so the mask can let an ineligible product through | The post-retrieval filter re-checks eligibility on product ids and removes it, and the output assertion blocks anything that still gets through. Tests confirm the trie itself never emits an ineligible product. |
| Stage 1 | **Codebook collapse**: many products end up sharing codes, so the beam covers less of the catalog | Training rejects any RQ-VAE checkpoint whose code usage falls below a minimum, and code usage is reported for every run. |
| Stage 1 | **Beam starvation**: a member eligible for very few products leaves beams with nowhere to go | Empty beams are reported, not filled from an unchecked source. The slate gets shorter, but never non-compliant. |
| Stage 1 | **Latency overrun** from the beam | The per-stage latency report shows which stage is slow. The KV-cache fix is described in Section 8. |
| Stage 2 | **Candidate cross-talk**: a mask bug lets candidates see each other, so a product's score depends on what else was retrieved | Tests check that shuffling or removing other candidates leaves each score unchanged, and that scoring all candidates at once gives the same result as scoring them one by one. |
| Stage 2 | **Future leakage**: a mask bug lets the member's state see later events or padding | Tests check that earlier positions are unaffected by anything after them. |
| Stage 2 | **Feature drift**: training and serving standardize features differently | The standardization values are saved inside the model, so serving uses exactly what training used. A test checks that reloaded artifacts produce identical slates. |
| Stage 3 | **Down-sampling rate changed** without updating the click correction, inflating click probabilities | The correction is stored with the model at training time rather than read from config at serving. A test checks it recovers the true probability, and click calibration error in the results would jump. |
| Stage 3 | **Partners slow down**, so more applications are pending at the cut-off than expected | The pending rate is reported for every dataset. The default policy, dropping pending rows, doesn't depend on any assumption about delays. The pending-policy comparison shows how much a wrong policy would cost. |
| Stage 4 | **Calibration drift**: real approval rates move away from the predicted odds | Calibration error is reported per tower and per credit tier. Calibrators are a separate artifact that can be refitted without retraining the model. |
| Stage 4 | **`α` set too high** by a business change | The harm rate rises sharply past the flat region of the `α` sweep. Guardrails still remove harmful products whenever a safe product of the same kind exists. |
| Stage 4 | **Re-ranker learns to prefer harmful products** | Guardrails run before the re-ranker, so removed products can't come back. A test checks each guardrail rule. |
| Stage 4 | **Re-ranker fills the slate from one family** | The family penalty `β` pushes repeats down. The family mix of served slates is visible in the end-to-end run. |
| Stage 4 | **Re-ranker scores logged as approval odds** by a later change | The re-ranker returns only an order. The odds shown and logged are the calibrated ones computed before it. |
| Serving | **Held or pending product served** because of a bug after retrieval | The output assertion fails the request instead of serving it. A test injects a violation and checks that it's caught. |
