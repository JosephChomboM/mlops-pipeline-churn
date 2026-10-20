# Simulated production: six months

| Month | Shift | Validation | Max feature PSI | Prediction PSI | Realized Gini [95% CI] (matured cohort) | Predicted / actual churn | Signals | Action | Champion |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01-01 | 0.0 | ok | 0.042 (monthly_charges) | 0.042 | pending | - | - | none | v1 |
| 2026-02-01 | 0.0 | ok | 0.033 (streaming_tv) | 0.059 | 0.516 [0.40-0.64] (2026-01-01, v1) | 26.0% / 26.8% | - | none | v1 |
| 2026-03-01 | 1.0 | ok | 0.460 (contract) | 0.744 | 0.516 [0.40-0.64] (2026-01-01, v1) | 26.0% / 26.8% | drift | promoted v2 | v1 -> v2 |
| 2026-04-01 | 1.5 | quarantined | - | - | 0.523 [0.40-0.64] (2026-03-01, v1) | 42.1% / 40.8% | - | none | v2 |
| 2026-05-01 | 2.0 | ok | 0.322 (internet_service) | 0.547 | 0.523 [0.40-0.64] (2026-03-01, v1) | 42.1% / 40.8% | drift | rejected v3 | v2 |
| 2026-06-01 | 2.0 | ok | 0.202 (monthly_charges) | 0.489 | 0.622 [0.52-0.71] (2026-05-01, v2) | 41.0% / 43.6% | drift | promoted v4 | v2 -> v4 |

## Promotion decisions (frozen holdout)

| Challenger | Gini | KS | ECE | Net value | vs champion | Result | Reason |
|---|---|---|---|---|---|---|---|
| v1 | 0.6668 | 0.5225 | 0.0222 | $61,655 | - | promoted | no champion yet |
| v2 | 0.6734 | 0.5143 | 0.0294 | $61,926 | v1: Gini 0.6668, $61,655 | promoted | value $61,926 > $61,655, Gini within tolerance |
| v3 | 0.6694 | 0.517 | 0.0239 | $61,719 | v2: Gini 0.6734, $61,926 | rejected | value $61,719 <= champion $61,926 |
| v4 | 0.6725 | 0.5151 | 0.0274 | $62,363 | v2: Gini 0.6734, $61,926 | promoted | value $62,363 > $61,926, Gini within tolerance |
