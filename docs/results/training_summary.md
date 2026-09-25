# Training summary (from artifacts/histories.json)

| model | steps run | step budget | selection criterion | best step | best value | early stop | seconds |
|---|---|---|---|---|---|---|---|
| rqvae | 2000 | 2000 | recon MSE s.t. min utilization >= 0.9 | 1900 | 0.0190 | no | 3.2733 |
| tiger | 800 | 3000 | val Recall@100 (checkpoint), val next-SID loss (stop) | 200 | 0.3105 | yes | 66.1745 |
| ranker | 700 | 2000 | val unified funnel loss (total) | 400 | 2.3747 | yes | 183.2982 |
| prm | 500 | 600 | val listwise loss | 350 | 2.3033 | yes | 0.9670 |

## RQ-VAE codebook utilization at the last evaluation

| level | utilization |
|---|---|
| 0 | 1.0000 |
| 1 | 1.0000 |
| 2 | 1.0000 |

Utilization constraint met: yes.

Ranker trainable parameters: 312032.
PRM training slates (>= 1 positive): 8400.
