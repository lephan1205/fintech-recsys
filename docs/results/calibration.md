# Calibration (fitted on the calibration split, scored on the test split)

| tower | method | rows | ECE | MCE | Brier |
|---|---|---|---|---|---|
| click | uncalibrated (sigmoid) | 120000 | 0.0085 | 0.0221 | 0.0705 |
| click | isotonic | 120000 | 0.0019 | 0.1213 | 0.0705 |
| click | platt | 120000 | 0.0022 | 0.0174 | 0.0705 |
| click | served: isotonic (requested; n=120000, w+=8984.0) |  |  |  |  |
| apply | uncalibrated (sigmoid) | 9162 | 0.0545 | 0.0675 | 0.1644 |
| apply | isotonic | 9162 | 0.0117 | 0.4882 | 0.1616 |
| apply | platt | 9162 | 0.0085 | 0.0236 | 0.1615 |
| apply | served: isotonic (requested; n=8984, w+=1793.0) |  |  |  |  |
| approve | uncalibrated (sigmoid) | 1792 | 0.0193 | 0.5000 | 0.1244 |
| approve | isotonic | 1792 | 0.0280 | 0.1714 | 0.1253 |
| approve | platt | 1792 | 0.0251 | 0.2018 | 0.1250 |
| approve | served: isotonic (requested; n=1727, w+=1435.0) |  |  |  |  |

## By credit tier (ECE / Brier)

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
