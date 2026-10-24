# Mazen — Solar Forecasting

## Role

Own the 24-hour and 48-hour solar forecasting pipeline.

## Data

- PVGIS v6 2024 modeled/reference solar data.
- Open-Meteo Previous Runs / GFS weather data.

## What to learn

Time series, regression, feature engineering, time-aware splitting, XGBoost, LSTM, MLflow, Optuna, and SHAP.

## Contract

- **Inputs:** timestamp/forecast origin, 24h or 48h horizon, weather forecast fields, calendar fields, and justified leakage-safe lag/rolling features.
- **Target:** future PVGIS solar power in kW.
- **Models:** persistence/simple baseline; XGBoost or LightGBM; compact LSTM.
- **Metrics:** MAE, RMSE, and sMAPE.
- **Function:** `predict_solar(payload)`.

```json
{
  "timestamp": "...",
  "horizon_hours": 24,
  "forecast_kw": 0.0,
  "model_version": "..."
}
```

## Main tasks

1. Inspect the data; audit timestamps and missing values; complete EDA covering distributions, seasonality, correlations, missingness, outliers, and time-series patterns.
2. Align PVGIS and Open-Meteo without pretending the sources describe a metered station; create leakage-safe features.
3. Create chronological/rolling-origin splits and a persistence baseline.
4. Train the listed model families; use Optuna only within training folds and track parameters, metrics, versions, and artifacts in MLflow.
5. Compare held-out results, create global/local SHAP evidence where applicable, and save the evidence-based champion.
6. Build and test `predict_solar(payload)`.

## Files to deliver

- Preprocessing/features, training, and inference modules.
- Input/output schemas and tests.
- Evaluation report and model/MLflow URI.

## Definition of done

- Data audit and EDA evidence are saved and limitations are stated.
- 24h/48h leakage-safe splits and features are reproducible.
- Baseline, tree model, and compact LSTM are fairly compared.
- MLflow runs and bounded Optuna study are reproducible.
- Champion, schemas, evaluation, and SHAP evidence are versioned.
- `predict_solar(payload)` passes tests without notebook state.
