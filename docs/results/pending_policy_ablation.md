# Pending-policy ablation (mortgage slice, young cut-off)

Dataset: 800 users, 150 products, family mix (0.05, 0.05, 0.1, 0.1, 0.7), snapshot 40.0 d, served window 30.0 d; ranker trained 300 steps per policy.  Bias = mean(p̂3) − mean(generator p_approve) on applied mortgage rows of the test split; AUC / ECE against the *oracle* approval label.

| pending_policy | applied mortgage rows | pending share | mean p_approve (truth) | mean p̂3 | bias | AUC vs oracle | ECE vs oracle |
|---|---|---|---|---|---|---|---|
| drop | 48 | 0.7500 | 0.6255 | 0.6914 | 0.0659 | 0.7554 | 0.1081 |
| ipw | 48 | 0.7500 | 0.6255 | 0.7976 | 0.1721 | 0.7214 | 0.2143 |
| negative | 48 | 0.7500 | 0.6255 | 0.0346 | -0.5909 | 0.6321 | 0.5487 |
