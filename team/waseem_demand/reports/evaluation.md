# Repaired EV demand evaluation

## Data audit and target semantics

441,077 raw rows; 17,547 continuous hourly rows. Coverage: 2019-12-31 21:00:00 to 2021-12-31 23:00:00. No valid event hours are discarded.

Energy is recognized in each session's completion hour, when final session energy becomes observable. Session count is arrivals in the start hour. All energy and arrivals are conserved. This replaces the old start-hour energy simplification and is not physical intra-session grid load.

No raw rows are altered. Full schema, missingness, duplicate and identifier checks, source SHA256, coverage and zero counts are in `data_audit.json`. UserID is not a transaction ID. No recorded events are treated as zero under the assumption of complete capture; telemetry outages are unknown.

Source timezone not independently verified. Local/naive operational time is retained, never relabeled UTC.

## Features and leakage controls

Calendar hour/day-of-week/weekend/month and cyclic hour/day encodings; energy/session lags 1, 2, 3, 24, 48, 168; shifted mean/std over 6 and 24 hours. Every row at target hour u uses demand through u-1 only. No weather, TOU price, fault flags or post-event outcome fields enter features. Completion energy is used only after its completion hour closes. No current target enters its features.

Weather is excluded because its observed daily aggregates were unavailable at origin. TOU is optional and excluded without a pricing provenance/ablation claim.

## Chronological evaluation

Train targets precede 2021-07-01; validation is July–September 2021; test begins 2021-10-01. Target timestamp sets are disjoint. Scalers and seasonal profiles use train only. Optuna tunes on validation only. LSTM uses validation for checkpoint selection. Per-target champion is chosen by validation MAE before final test reporting. All families use identical evaluation hours with real historical context across split boundaries. This is rolling one-step-ahead evaluation, not an open-loop multihour forecast.

| Split | Rows | First target | Last target |
|---|---:|---|---|
| train | 12940 | 2020-01-08 20:00:00 | 2021-06-30 23:00:00 |
| validation | 2208 | 2021-07-01 00:00:00 | 2021-09-30 23:00:00 |
| test | 2208 | 2021-10-01 00:00:00 | 2021-12-31 23:00:00 |

## Model comparison

Metrics: MAE / RMSE / sMAPE (%). Both-zero sMAPE terms count as zero.

| Family | Validation energy | Test energy | Validation sessions | Test sessions |
|---|---|---|---|---|
| Baseline | 127.6521 / 165.2446 / 28.8299 | 146.3826 / 188.9539 / 29.0516 | 8.3486 / 11.4375 / 30.5895 | 8.6278 / 11.3659 / 29.8579 |
| XGBoost | 91.8705 / 117.4610 / 21.6161 | 103.6747 / 134.7429 / 19.6862 | 6.2351 / 8.4282 / 23.5120 | 6.3405 / 8.2940 / 21.3568 |
| LSTM | 93.3739 / 120.0046 / 21.7653 | 108.6791 / 140.6443 / 20.8681 | 6.3629 / 8.5560 / 23.8565 | 6.6831 / 8.7504 / 22.3379 |

Champions selected on validation: {'energy_kwh': 'XGBoost', 'sessions': 'XGBoost'}.

Seasonal baseline retains the original 50% lag24 / 50% training calendar median blend. XGBoost uses bounded, seeded Optuna trials (see `optuna.json`). Compact LSTM retains the original 32-unit architecture, fixes sequence alignment, and scales both inputs and targets. Every sequence includes the latest feature row; no padded predictions are evaluated.

## EDA

`eda.png` shows demand distributions, hourly and monthly seasonality, and arrival/energy correlation. Zero hours and source missingness are audited; large observations are retained rather than silently clipped as outliers. `holdout.png` compares all families on the first test week. Numerical distributions/correlations are recorded in `eda_summary.json`.

## Limitations

- Source timezone provenance remains unresolved.
- Completed-session energy is not instantaneous station load or energy consumed within that hour.
- End-of-hour transaction availability and complete source capture are assumed; reporting delays are not modeled.
- Weather and pricing are excluded; the model has not been validated on Egyptian stations.
- Requires at least 192 real continuous hourly records ending immediately before origin.
- Expected sessions remain a nonnegative float statistical estimate, not an exact future arrival count.
- Small CPU model/tuning budgets and one chronological holdout limit generalization evidence.

## MLflow

Experiment: `chargesentinel-demand`. Real runs and bundle URIs:

- Baseline: `runs:/3a9753a01ece44ddaf67472831d0c3f7/bundle`
- XGBoost: `runs:/93b9d57c77884a618b513c4ae08b1085/bundle`
- LSTM: `runs:/a34cf7e5426e4decaf3b750eb8ecc04c/bundle`

The SQLite store remains local under `runtime/`. These URIs require that store; the committed `bundle/v1` is the portable handoff. No remote registry is claimed.

## Execution verification

- Automated suite: **34 passed, 0 failed** with the pinned training stack.
- Independently reloaded all three frozen model families and reproduced all 36 validation/test metric values to numerical tolerance; no retraining was used in that check.
- Verified all three MLflow runs are FINISHED, use `chargesentinel-demand`, contain 12 metrics each, the required parameters/tags, and byte-identical bundle artifacts.
- Verified original source SHA256 remained unchanged after execution.
- Fixture inference passed in fresh processes from repository root and another working directory. Each model family was exercised separately.
- Champion inference also passed in a second environment containing only runtime requirements: no scikit-learn, PyTorch, MLflow, or Optuna installed. Missing history raised an error.
- Inspected both EDA and holdout figures. No secrets, user-specific paths, large raw data, or MLflow databases are included in the handoff.

Exact training packages are in `environment.txt`. MLflow logging resumed from the frozen bundle after a SQLAlchemy compatibility pin; those packages are recorded separately in `tracking_environment.txt`. This did not alter model weights or evaluation metrics.
