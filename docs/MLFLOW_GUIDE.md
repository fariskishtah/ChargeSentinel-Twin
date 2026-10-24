# MLflow Guide

## Local setup

Install `requirements/base.txt`, set `MLFLOW_TRACKING_URI` as shown in `.env.example`, then start a local server from the repository root:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

`MLFLOW_TRACKING_URI` selects another server when needed. In training code, call `get_or_create_experiment("solar")` (or `demand`, `anomaly`, `availability`) from `src.common.mlflow_config` before starting a run.

| Module | Experiment | Metrics |
|---|---|---|
| Solar | `chargesentinel-solar` | MAE, RMSE, sMAPE |
| Demand | `chargesentinel-demand` | MAE, RMSE, sMAPE |
| Anomaly | `chargesentinel-anomaly` | Precision, recall, F1, PR-AUC |
| Availability | `chargesentinel-availability` | MAE/RMSE minutes |

Every run logs `model_name`, hyperparameters, `dataset_version`, and `feature_version`; plots, evaluation report, schema/feature information, and the loadable model artifact; plus `owner`, `git_commit`, and `module` tags.

## Handoff

Log the final pipeline with the appropriate MLflow model flavor and register it only when ready. Give Faris a real URI such as `runs:/<run_id>/model` or `models:/<registered_name>/<version>` plus the items in `docs/MODEL_HANDOFF_CONTRACT.md`. Never create placeholder URIs.
