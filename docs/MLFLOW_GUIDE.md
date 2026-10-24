# MLflow Guide

Use the shared tracking URI from `CHARGESENTINEL_MLFLOW_TRACKING_URI`. Keep training local and hand off an immutable run/model URI; do not copy undocumented notebook state.

| Module | Experiment name | Log |
|---|---|---|
| Solar | `chargesentinel-solar` | Horizon, features, split/fold, model parameters, Optuna trial, MAE/RMSE/sMAPE, plots, schemas, pipeline/model artifact |
| Demand | `chargesentinel-demand` | Aggregation/version, features, split/fold, parameters, Optuna trial, MAE/RMSE/sMAPE, plots, schemas, pipeline/model artifact |
| Anomaly | `chargesentinel-anomaly` | Feature/leakage version, label source, method, threshold, parameters, Optuna trial, precision/recall/F1/PR-AUC, confusion matrix, SHAP/reason artifacts |
| Availability | `chargesentinel-availability` | Feature/version, split/fold, parameters, Optuna trial, MAE/RMSE minutes, recommendation config, schemas, pipeline/model artifact |

## Handoff

1. Tag the selected run with code, data, feature, and model versions.
2. Log preprocessing, schemas, evaluation evidence, limitations, and the loadable model/pipeline as artifacts.
3. Give Faris a resolvable URI such as `runs:/<run_id>/model` or `models:/<registered_name>/<version>`, plus the items in `docs/MODEL_HANDOFF_CONTRACT.md`.
