# Funnel metrics (test split, calibrated probabilities)

| task | rows | positives | AUC | PR-AUC | GAUC | NCE | ECE |
|---|---|---|---|---|---|---|---|
| click | 120000 | 9162 | 0.5353 | 0.0852 | 0.5268 | 1.0098 | 0.0019 |
| apply | click | 9162 | 1886 | 0.5754 | 0.2554 | 0.5354 | 0.9990 | 0.0117 |
| approve | apply (resolved, weighted) | 1792 | 1460 | 0.7843 | 0.9338 | 0.7813 | 0.9419 | 0.0280 |

Realized partner revenue on resolved rows: $363,467 (a **lower bound**: pending applications are excluded, D4); expected revenue Σ P_funded · payout: $397,259.

## Oracle ceiling (the generator's own probabilities scored on the same rows)

Labels are Bernoulli draws from these probabilities, so no model can beat this ordering in expectation.

| task | AUC | PR-AUC | GAUC | NCE |
|---|---|---|---|---|
| click | 0.6316 | 0.1294 | 0.6199 | 0.9697 |
| apply | click | 0.6143 | 0.2826 | 0.5901 | 0.9733 |
| approve | apply (resolved, weighted) | 0.9251 | 0.9811 | 0.9109 | 0.4800 |
Pending applications in the test split (excluded from approval metrics): 94.
