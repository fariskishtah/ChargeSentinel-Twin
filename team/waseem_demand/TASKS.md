# Waseem — EV Demand Forecasting

## Role

Own hourly EV demand forecasting.

## Data

- Jiaxing `Charging_Data.csv`.
- Jiaxing `Weather_Data.csv` only if it is safely useful and time-aligned.

## What to learn

Time series, aggregation, lag/rolling features, XGBoost, LSTM, time-aware validation, MLflow, and Optuna.

## Goal and contract

Convert charging transactions into an hourly demand table and predict future demand.

- **Inputs:** historical hourly energy, session count, charger activity, calendar fields, and safe lag/rolling features.
- **Targets:** `expected_energy_kwh` and `expected_sessions`.
- **Models:** historical/seasonal baseline; XGBoost; compact LSTM.
- **Metrics:** MAE, RMSE, and sMAPE.
- **Function:** `predict_demand(payload)`.

```json
{
  "timestamp": "...",
  "expected_energy_kwh": 0.0,
  "expected_sessions": 0,
  "model_version": "..."
}
```

## Main tasks

1. Audit transactions and complete EDA for coverage, missingness, outliers, seasonality, demand distributions, correlations, and time-series patterns.
2. Build the hourly demand table and identify missing hours without silently filling them.
3. Create leakage-safe lag/rolling features and chronological/rolling-origin splits.
4. Train the listed baselines/models; use Optuna only within training folds and track all runs/artifacts in MLflow.
5. Compare held-out results and export the evidence-based champion.
6. Build and test `predict_demand(payload)`.

## Files to deliver

- `build_hourly.py`, `features.py`, `train.py`, and `inference.py`.
- Input/output schemas and tests.
- Evaluation report and MLflow URI.

## Definition of done

- Transaction audit and EDA evidence are saved.
- Hourly aggregation and missing-hour policy are reproducible.
- Leakage-safe time splits and features pass checks.
- Baseline, XGBoost, and compact LSTM are fairly compared.
- MLflow runs, Optuna study, champion URI, and evaluation are versioned.
- `predict_demand(payload)` passes tests without notebook state.
