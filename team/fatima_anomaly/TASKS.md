# Fatima — Anomaly Detection + XAI

## Role

Own anomaly detection, thresholding, and explanations.

## Data and target

- **Data:** Jiaxing `Charging_Data.csv`.
- **Target:** `is_abnormal`.

## What to learn

Classification, anomaly detection, data leakage, class imbalance, logistic regression, XGBoost, Isolation Forest, autoencoders, SHAP, MLflow, and Optuna.

## Critical leakage gate

Before training, inspect every actual column and classify it as **safe**, **leakage risk**, **label-derived**, or **post-event only**. Do not use direct outcome fields such as `end_cause` when they reveal the label.

## Contract

- **Models:** baseline classifier; XGBoost classifier; Isolation Forest or autoencoder comparison.
- **Metrics:** precision, recall, F1, PR-AUC, and confusion matrix.
- **Function:** `predict_anomaly(payload)`.

```json
{
  "charger_id": "...",
  "anomaly": true,
  "score": 0.0,
  "reason": [],
  "model_version": "..."
}
```

## Main tasks

1. Audit data and target distribution; complete EDA for missingness, outliers, imbalance, distributions, correlations, and time patterns.
2. Document the feature-by-feature leakage classification before selecting features.
3. Build the baseline, XGBoost, and Isolation Forest or autoencoder comparison; keep supervised and unsupervised evidence separate.
4. Tune thresholds and key hyperparameters systematically with Optuna; track parameters, metrics, versions, and artifacts in MLflow.
5. Produce global and local SHAP explanations for the supervised tree model and concise reason codes for alerts.
6. Build and test `predict_anomaly(payload)`.

## Files to deliver

- `features.py`, `train.py`, `inference.py`, and `explain.py`.
- Threshold/config, schemas, evaluation report, SHAP artifacts, MLflow URI, and tests.

## Definition of done

- Data/target audit, EDA, imbalance analysis, and leakage register are saved.
- Unsafe, label-derived, and post-event fields are excluded from inference.
- Required methods and thresholds are compared with the stated metrics.
- MLflow runs and bounded Optuna study are reproducible.
- Schemas, SHAP/reason evidence, evaluation, and champion URI are versioned.
- `predict_anomaly(payload)` passes tests without notebook state.
