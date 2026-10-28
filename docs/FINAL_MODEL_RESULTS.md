# Final model results for reviewers

These are historical research-prototype results. Model families and thresholds are frozen. Validation chose champions; reported test values come from the owner evidence. MAE/RMSE units differ across tasks and must not be compared directly.

## Solar — modelled New Cairo PV reference

| Horizon | Family | May 2024 validation MAE (kW) | June test MAE (kW) | Status |
|---|---|---:|---:|---|
| 24 h | Seasonal Persistence | 1.623 | 0.361 | Champion |
| 24 h | Ridge Regression | 1.636 | — | Candidate |
| 24 h | XGBoost | 2.236 | — | Candidate |
| 48 h | Seasonal Persistence | 1.794 | — | Candidate |
| 48 h | Ridge Regression | 1.756 | 1.257 | Champion |
| 48 h | XGBoost | 2.231 | — | Candidate |

The committed solar selection report records MAE only; RMSE was not reported. Source: `team/mazen_solar/reports/integration_hardening/selection.json`.

## EV demand — Jiaxing completion-hour proxy

| Family | Validation energy MAE / RMSE (kWh) | Test energy MAE / RMSE (kWh) | Validation arrivals MAE / RMSE | Test arrivals MAE / RMSE |
|---|---:|---:|---:|---:|
| Baseline | 127.65 / 165.24 | 146.38 / 188.95 | 8.35 / 11.44 | 8.63 / 11.37 |
| XGBoost **champion** | 91.87 / 117.46 | 103.67 / 134.74 | 6.24 / 8.43 | 6.34 / 8.29 |
| LSTM | 93.37 / 120.00 | 108.68 / 140.64 | 6.36 / 8.56 | 6.68 / 8.75 |

Source: `team/waseem_demand/reports/metrics.json`. Energy is assigned to completion hour, not measured feeder load.

## EV session-start anomaly risk

| Model | Test precision | Recall | F1 | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Location baseline | 0.226 | 0.815 | 0.354 | 0.632 | 0.222 | — |
| CatBoost `anomaly-v1` | 0.215 | 0.888 | 0.346 | 0.642 | 0.238 | 0.40 |

The CatBoost threshold was selected on validation. Frequent false positives remain; the target is a later recorded abnormal end cause, not a diagnosed hardware fault. Source: `team/fatima_anomaly/reports/integration_hardening/metrics.json`.

## Historical charging-load interval anomaly

| Detector | Natural held-out alert rate | Detection on 39 injected test hours |
|---|---:|---:|
| Causal robust seasonal rule **selected** | 0.70% | 24/39 (61.54%) |
| Isolation Forest comparison | 14.29% | 27/39 (69.23%) |

The rule threshold is 5.827, calibrated on natural January–June 2021 validation scores. Natural alert rate is **not** false-positive rate; injected sensitivity is **not** real fault recall. There are no confirmed feeder-fault labels. Source: `team/fatima_anomaly/reports/grid_load/comparison.json`.

## Charger availability — Jiaxing duration

| Family | Normal-session validation MAE / RMSE (min) | Normal-session test MAE / RMSE (min) |
|---|---:|---:|
| Median baseline | 44.108 / 109.294 | 35.520 / 84.781 |
| Random Forest | 43.629 / 107.146 | — |
| Default XGBoost **champion** | 43.404 / 107.276 | 34.948 / 83.049 |
| Tuned XGBoost | 43.398 / 107.173 | — |

The tuned validation gain was below the documented 1% promotion threshold. The original test partition had been scored before tuning, so the test scores are retrospective. The distinct portable rebuild produced validation MAE 43.401 and test MAE/RMSE 34.980/82.836 minutes; it does not replace the owner report. Sources: `team/mariam_availability/reports/availability_duration_*`, `docs/MARIAM_AVAILABILITY_REVIEW.md`.

Recommendation is a deterministic weighted rule, with no trained ranking target or predictive ranking metric. Its replay did not establish wait improvement: a free charger existed at every evaluated arrival. See `team/mariam_availability/reports/availability_recommendation_evaluation.md`.
