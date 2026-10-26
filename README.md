# EV Demand Forecasting & Smart Energy System (Person 2)

Comprehensive end-to-end implementation of **Person 2: EV Demand Forecasting** for the Smart Energy & EV Charging Architecture (SIC 8 / 2026).

---

## 📌 Architecture & Responsibilities Overview

| Component | File / Path | Responsibility |
| :--- | :--- | :--- |
| **Hourly Builder** | `src/demand/build_hourly.py` | Aggregates raw transactions into continuous hourly intervals (17,544 hours) and aligns weather. |
| **Feature Engineering** | `src/demand/features.py` | Builds leakage-safe lags, rolling statistics, calendar cyclical encodings, and verifies zero leakage. |
| **Model Training & MLflow** | `src/demand/train.py` | Trains 3 model families (Baseline, LSTM, XGBoost), runs Optuna tuning, and tracks runs in MLflow. |
| **Inference Service** | `src/demand/inference.py` | Production adapter exposing `predict_demand(payload)` for Person 5 (FastAPI Backend). |
| **Input Schema** | `schemas/demand_input.json` | JSON Schema specifying the input payload contract. |
| **Output Schema** | `schemas/demand_output.json` | JSON Schema specifying the output prediction response. |
| **Evaluation Report** | `reports/demand_evaluation.md` | Complete scientific and engineering evaluation report with metrics and figures. |

---

## 🚀 Execution Instructions

### 1. Build the Hourly Continuous Dataset
```bash
python src/demand/build_hourly.py
```
Outputs `data/processed/hourly_demand.csv` (17,544 rows from 2020-01-01 to 2021-12-31).

### 2. Verify Zero Data Leakage
```bash
python src/demand/features.py
```
Runs the automated unit test ensuring future transaction changes do not impact past/current feature vectors.

### 3. Train Models & Track Experiments (MLflow + Optuna)
```bash
python src/demand/train.py 10
```
Trains:
1. **Seasonal Persistence Baseline**
2. **Compact LSTM (PyTorch)**
3. **XGBoost Regressor (Optuna Tuned - Champion)**

Saves champion model artifacts to `models/` and charts to `reports/figures/`.

### 4. Test Production Inference
```bash
python src/demand/inference.py
```
Validates the handoff contract for Person 5.

---

## 📊 Summary of Model Performance (Held-Out Test Set: Q4 2021)

### Next-Hour Energy Demand (`expected_energy_kwh`):
- **Seasonal Baseline:** MAE = 145.14 kWh | RMSE = 184.49 kWh | sMAPE = 30.12% | $R^2$ = 0.6526
- **Compact LSTM (PyTorch):** MAE = 135.04 kWh | RMSE = 177.80 kWh | sMAPE = 26.71% | $R^2$ = 0.6774
- **XGBoost (Champion):** **MAE = 110.39 kWh** | **RMSE = 143.57 kWh** | **sMAPE = 22.05%** | **$R^2$ = 0.7896**

### Next-Hour Charging Sessions (`expected_sessions`):
- **Seasonal Baseline:** MAE = 8.62 sessions | RMSE = 11.35 sessions | sMAPE = 29.82% | $R^2$ = 0.6816
- **Compact LSTM (PyTorch):** MAE = 8.05 sessions | RMSE = 11.02 sessions | sMAPE = 26.62% | $R^2$ = 0.6994
- **XGBoost (Champion):** **MAE = 6.30 sessions** | **RMSE = 8.30 sessions** | **sMAPE = 21.34%** | **$R^2$ = 0.8298**

**Champion MLflow Model URI:** `runs:/8bf6a21eef3449e39fbf662ed4405679/model_energy`
