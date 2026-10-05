# Decision register

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
| `delay.log_mu` | `((-0.356675, -0.356675), (-0.356675, -0.356675), (0.693147, 0.693147), (1.60944, 1.60944), (3.55535, 2.99573))` | `data/schema.py::DelayConfig` | log-normal medians: cards 0.7 d, personal loan 2 d, auto 5 d, mortgage 35 d approved / 20 d declined | per-family medians | partner behaviour; fit by Kaplan–Meier in production | measured: the mortgage slice is where the policy matters (`docs/results/pending_policy_ablation.md`) |
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
| `models.prm_cannibalization_weight` | `0.5` | `training/config.py::ModelConfig` | hand-set starting point, not swept: a second same-family product must beat the next family by half a PRM score unit | 0–2 | diversity policy | unsupported by a table; visible in the e2e slates |
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
