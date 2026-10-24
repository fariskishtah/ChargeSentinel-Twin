# Mariam — Availability + Recommendation

## Role

Own charger availability prediction and smart recommendation.

## Data

Jiaxing `Charging_Data.csv`.

## What to learn

Regression, session duration, occupancy reconstruction, availability prediction, Random Forest, XGBoost, recommendation logic, optimization, OR-Tools, MLflow, and Optuna.

## Goals and contract

1. Predict session duration and expected free time.
2. Recommend the best charger or time slot.

- **Availability inputs:** Charging Post ID, Start Time, safe calendar/session features, and historical charger behavior.
- **Target:** `session_duration_minutes`. End Time may calculate the target but must never be an inference feature.
- **Models:** median baseline; Random Forest; XGBoost.
- **Metrics:** MAE minutes and RMSE minutes.
- **Recommendation inputs:** user ETA, expected free time, demand forecast, solar forecast, and anomaly status.
- **Functions:** `predict_availability(payload)` and `recommend(payload)`.

```json
{
  "recommended_charger": "...",
  "expected_free_time": "...",
  "expected_wait_minutes": 0,
  "recommended_slot": "...",
  "model_version": "..."
}
```

Start with a transparent deterministic score. Use OR-Tools only if it is useful and stable. Return alternative chargers as part of the recommendation contract once the exact schema is agreed.

## Main tasks

1. Audit sessions, reconstruct occupancy, and complete EDA for duration/occupancy distributions, missingness, outliers, seasonality, correlations, and time patterns.
2. Build leakage-safe availability features and time-aware splits; explicitly test that End Time is absent from inference inputs.
3. Train the median, Random Forest, and XGBoost models; tune key parameters with Optuna and track all runs/artifacts in MLflow.
4. Compare held-out duration results and export the evidence-based champion.
5. Build and version the deterministic recommendation score; add OR-Tools only if justified by evidence.
6. Build and test `predict_availability(payload)` and `recommend(payload)`.

## Files to deliver

- Availability features, `train.py`, `inference.py`, and `recommendation/ranker.py`.
- Config, schemas, evaluation report, MLflow URI, and tests.

## Definition of done

- Session audit, occupancy reconstruction, and EDA evidence are saved.
- End Time is target-only and leakage tests pass.
- Baseline, Random Forest, and XGBoost are fairly compared.
- MLflow runs, Optuna study, evaluation, and champion URI are versioned.
- Recommendation inputs, weights, alternatives, and limitations are documented.
- Both inference functions pass tests without notebook state.
