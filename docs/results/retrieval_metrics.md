# Retrieval metrics (test split, next-item positives)

| retriever | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|
| TIGER (beam) | 0.0868 | 0.2283 | 0.3379 |
| eligible-popularity | 0.0548 | 0.2877 | 0.3836 |
| eligible-random | 0.0274 | 0.0959 | 0.1598 |

Users counted: 219; positives excluded as ineligible: 70; beam yield |C_u| / 100: 0.9807

## Recall@100 by credit tier

| tier | users | Recall@100 |
|---|---|---|
| DEEP_SUBPRIME | 8 | 0.8750 |
| SUBPRIME | 14 | 0.7857 |
| NEAR_PRIME | 72 | 0.4167 |
| PRIME | 73 | 0.2740 |
| SUPER_PRIME | 52 | 0.1154 |

## Recall@100 by target family

| family | users | Recall@100 |
|---|---|---|
| CREDIT_CARD | 52 | 0.1154 |
| BALANCE_TRANSFER_CARD | 47 | 0.4468 |
| PERSONAL_LOAN | 41 | 0.3171 |
| AUTO_REFINANCE | 48 | 0.4583 |
| MORTGAGE | 31 | 0.3871 |

## Slate-level positives (same beams)

| positive definition | users | Recall@10 | Recall@50 | Recall@100 |
|---|---|---|---|---|
| click | 300 | 0.0349 | 0.1457 | 0.2532 |
| apply | 293 | 0.0313 | 0.1372 | 0.2376 |
| approve | 292 | 0.0307 | 0.1286 | 0.2211 |
