# Serving latency (CPU, B = 1, 50 test users)

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

Total p50 = 19.84 ms, ABOVE the 10 ms budget.
Served slates: mean size 10.00, mean eligible 786.5, mean retrieved 99.4.
