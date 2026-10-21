# Simulated production: six months

| Month | Shift | Validation | Max feature PSI | Prediction PSI | Realized Gini [95% CI] (matured cohort) | Predicted / actual churn | Signals | Action | Champion |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01-01 | 0.0 | ok | 0.042 (monthly_charges) | 0.088 | pending | - | - | none | v1 |
| 2026-02-01 | 0.0 | ok | 0.033 (streaming_tv) | 0.040 | 0.534 [0.42-0.67] (2026-01-01, v1) | 26.0% / 26.8% | - | none | v1 |
| 2026-03-01 | 1.0 | ok | 0.460 (contract) | 0.656 | 0.534 [0.42-0.67] (2026-01-01, v1) | 26.0% / 26.8% | drift | rejected v2 | v1 |
| 2026-04-01 | 1.5 | quarantined | - | - | 0.527 [0.42-0.63] (2026-03-01, v1) | 43.2% / 40.8% | - | none | v1 |
| 2026-05-01 | 2.0 | ok | 0.321 (internet_service) | 0.591 | 0.527 [0.42-0.63] (2026-03-01, v1) | 43.2% / 40.8% | drift | rejected v3 | v1 |
| 2026-06-01 | 2.0 | ok | 0.203 (monthly_charges) | 0.437 | 0.640 [0.54-0.74] (2026-05-01, v1) | 40.6% / 43.6% | drift | rejected v4 | v1 |

## Promotion decisions (frozen holdout)

| Challenger | Gini | KS | ECE | Net value | vs champion | Difference [95% CI] | Wins | Result |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.6757 | 0.531 | 0.0203 | $62,715 | - | - | - | promoted |
| v2 | 0.6745 | 0.5211 | 0.0252 | $61,725 | v1: Gini 0.6757, $62,715 | -$990 [-$2,078, -$34] | 2% | rejected |
| v3 | 0.6712 | 0.5161 | 0.0241 | $61,742 | v1: Gini 0.6757, $62,715 | -$974 [-$2,351, +$373] | 8% | rejected |
| v4 | 0.6753 | 0.5215 | 0.0215 | $62,003 | v1: Gini 0.6757, $62,715 | -$712 [-$1,877, +$420] | 11% | rejected |

A challenger is promoted only if it wins at least 95% of 2,000 paired bootstrap resamples of the holdout.
