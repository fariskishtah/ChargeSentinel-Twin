# EV Demand Forecasting Evaluation Report (Person 2)

**Author:** Person 2 - EV Demand Forecasting Specialist  
**Project:** Smart Energy & EV Charging System (SIC 8 / 2026)  
**Target Handoff:** Person 5 (Backend / Integration / Digital Twin / Deployment)  
**Date:** October 2026  
**Champion MLflow Model URI:** `runs:/8bf6a21eef3449e39fbf662ed4405679/model_energy`

---

## 1. Executive Summary & Stakeholder Objectives

The primary objective of **Person 2** is to deliver accurate, leakage-safe, next-hour forecasts of electric vehicle (EV) charging demand across the Jiaxing municipal charging network. Accurate demand forecasting enables:
- **Grid Stability & Peak Shaving:** Prevents substation overload by anticipating surge periods.
- **Smart Renewable Energy Matching:** Feeds into Person 1 (Solar Forecasting) and Person 5 (Digital Twin) to balance PV power generation with EV load.
- **Dynamic Charger Allocation:** Assists Person 4 (Availability & Recommendations) in guiding drivers to optimal charging stations before congestion peaks.

### Key Deliverables Completed:
1. `src/demand/build_hourly.py`: Data pipeline that aggregates 441,077 raw transactions into 17,544 continuous chronological hourly buckets with aligned weather context.
2. `src/demand/features.py`: Leakage-safe feature pipeline (shift before rolling, cyclical encodings, automated leakage unit tests).
3. `src/demand/train.py`: Multi-family training pipeline with Optuna hyperparameter optimization and MLflow tracking.
4. `src/demand/inference.py`: Production-grade `predict_demand(payload)` interface meeting strict JSON schema contracts.
5. `schemas/demand_input.json` & `schemas/demand_output.json`: Typed API contracts for Person 5.
6. `reports/figures/`: High-resolution figures illustrating demand distributions, diurnal seasonality, correlations, feature importances, and residual distributions.

---

## 2. Data Audit, Provenance & Preprocessing

### 2.1 Raw Data Sources
- **Charging Transactions (`Charging_Data.csv`):** 441,077 transaction rows, 24 columns, covering `2020-01-01 00:00:00` to `2021-12-31 23:58:03` (2 full years). 0 missing values, 0 duplicate rows.
- **Weather Observations (`Weather_Data.csv`):** 2,232 daily records across 3 municipal districts (`Nanhu`, `Tongxiang`, `Xiuzhou`). Daily metrics: Temperature (°C), Relative Humidity (%), and Precipitation (mm).

### 2.2 Useless & Leakage Column Elimination
Out of 24 raw transaction columns, the majority represent post-session billing outcomes, operator maintenance logs, or individual user identifiers that either introduce future data leakage or have zero predictive value for macro hourly demand forecasting:

| Column Name | Action | Detailed Justification |
| :--- | :--- | :--- |
| `UserID` | **Dropped** | High cardinality (56,115 unique IDs). Individual user tracking is irrelevant for system-level hourly demand and introduces noise. |
| `Order creation time` | **Dropped** | Redundant with `Start Time` (99.8% identical to within seconds). `Start Time` is the physical moment charging commences. |
| `End Time` | **Dropped** | **Future Data Leakage.** The exact session end time is unknown when the session initiates or at the forecast origin. Reserved exclusively for Person 4. |
| `Payment time` | **Dropped** | **Future Data Leakage.** Billing completion happens after session conclusion. |
| `Electricity cost/Yuan` | **Dropped** | Post-session financial calculation ($kWh \times tariff$). Irrelevant to physical power forecasting. |
| `Service charge/Yuan` | **Dropped** | Financial outcome fee. |
| `Transaction Amount/Yuan` | **Dropped** | Total financial cost. |
| `Actual Payment/Yuan` | **Dropped** | Final settled billing amount. |
| `end_cause` | **Dropped** | Post-session categorization (e.g. normal stop, fault). Reserved for Person 3 anomaly detection. |
| `is_full_charge` | **Dropped** | Session outcome flag unknown at forecast origin. |
| `is_user_stop` | **Dropped** | Session outcome flag unknown at forecast origin. |
| `is_nonsystem_fault` | **Dropped** | Fault flag; outcome of the session. |
| `is_system_issue` | **Dropped** | Fault flag; outcome of the session. |
| `is_EV_fault` | **Dropped** | Fault flag; outcome of the session. |
| `is_other_fault` | **Dropped** | Fault flag; outcome of the session. |
| `is_abnormal` | **Dropped** | Target for Person 3; post-session label. |
| `Charging Post ID` / `Location Information` | **Aggregated** | Used for station-level validation; summed into network-wide hourly targets. |

### 2.3 Preprocessing & Aggregation Steps
1. **Encoding & Text Normalization:** Decoded with `gb18030` to preserve Chinese text and temperature symbols (`℃`). All string columns were stripped of trailing tab (`\t`) characters and leading/trailing whitespace.
2. **Hourly Grid Construction:** Built a strict, complete date range spanning `2020-01-01 00:00:00` to `2021-12-31 23:00:00` (731 days $\times$ 24 hours = 17,544 continuous rows).
3. **Zero-Filling Inactive Hours:** 477 inactive hours (e.g., specific late-night intervals) were imputed with `energy_kwh = 0.0` and `sessions = 0`, ensuring time series continuity without temporal gaps.
4. **Weather Harmonization:** Aligned daily mean temperature, humidity, and precipitation across the 3 districts to each hourly timestamp.

---

## 3. Exploratory Data Analysis (EDA) & Demand Dynamics

### 3.1 Demand Statistics (Aggregated Network Level)
- **Hourly Energy (kWh):** Mean = 413.21 kWh, Median = 385.28 kWh, Std = 293.22 kWh, Min = 0.00 kWh, Max = 1,363.87 kWh.
- **Hourly Charging Sessions:** Mean = 25.14 sessions, Median = 23.00 sessions, Std = 18.88 sessions, Min = 0, Max = 129 sessions.

### 3.2 Diurnal & Weekly Seasonality
- **Peak Charging Window:** 12:00 PM – 4:00 PM (averaging > 650 kWh/hour and > 40 sessions/hour), coinciding with afternoon commercial and taxi/fleet fast charging.
- **Secondary Peak:** 7:00 PM – 10:00 PM (averaging ~500 kWh/hour), corresponding to residential and commuter evening charging.
- **Trough Window:** 1:00 AM – 5:00 AM (averaging < 100 kWh/hour).
- **Weekly Seasonality:** Strong correlation ($r = 0.86$) between identical hours across consecutive weeks (`lag_168h`), proving persistent weekly commute schedules.

### 3.3 Correlation Analysis (Pearson & Spearman)
- `energy_lag_24h` exhibits a **0.879** correlation with next-hour energy.
- `energy_lag_48h` exhibits a **0.872** correlation.
- `energy_lag_168h` exhibits a **0.861** correlation.
- Temperature shows moderate non-linear correlation with demand (extreme cold and hot days increase HVAC cabin conditioning load during charging).

---

## 4. Leakage-Safe Feature Engineering

To guarantee zero data leakage into future predictions:
1. **Lags:** Relative to target $t+1$, lag 1 is origin $t$, lag 2 is $t-1$, lag 24 is same hour yesterday ($t-23$), lag 48 is $t-47$, and lag 168 is $t-167$.
2. **Rolling Statistics:** Shifted strictly before computation; computed on windows of 6h and 24h over historical observations up to origin $t$.
3. **Cyclical Calendar Features:** Target timestamp $t+1$ is encoded using sine and cosine trigonometric transforms for `hour` (period 24), `dayofweek` (period 7), and `month` (period 12).
4. **Automated Leakage Verification:** A unit test (`test_leakage_protection()`) programmatically verifies that perturbing future row $i+1$ causes **0.0000** difference in features at row $i$.

---

## 5. Model Training, Optuna Tuning & Evaluation

### 5.1 Chronological Splits
- **Training Set:** `2020-01-08 00:00:00` to `2021-06-30 23:00:00` (12,961 hours, ~75%)
- **Validation Set (Optuna Tuning):** `2021-07-01 00:00:00` to `2021-09-30 23:00:00` (2,208 hours, ~12.5%)
- **Holdout Test Set (Untouched Evaluation):** `2021-10-01 00:00:00` to `2021-12-31 22:00:00` (2,207 hours, ~12.5%)

### 5.2 Model Comparison on Held-Out Test Set

#### Target 1: Next-Hour Energy Demand (`expected_energy_kwh`)

| Model Family | Description / Architecture | MAE (kWh) | RMSE (kWh) | sMAPE (%) | $R^2$ Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Baseline** | Lag-24 + DayOfWeek/Hour Median Blend | 145.14 | 184.49 | 30.12% | 0.6526 | Benchmark |
| **Compact LSTM** | PyTorch Sequence-to-One (SeqLen=24, Hidden=32) | 135.04 | 177.80 | 26.71% | 0.6774 | Deep Learning |
| **XGBoost (Optuna)** | Gradient Boosted Trees (Tuned on Val fold) | **110.39** | **143.57** | **22.05%** | **0.7896** | **CHAMPION** |

> **Improvement:** XGBoost achieves a **23.9% reduction in MAE** and a **22.2% reduction in RMSE** compared to the seasonal persistence baseline, explaining nearly 79% of variance ($R^2 = 0.7896$) on the future held-out test quarter.

#### Target 2: Next-Hour Charging Sessions (`expected_sessions`)

| Model Family | Description / Architecture | MAE (sessions) | RMSE (sessions) | sMAPE (%) | $R^2$ Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Baseline** | Lag-24 + DayOfWeek/Hour Median Blend | 8.62 | 11.35 | 29.82% | 0.6816 | Benchmark |
| **Compact LSTM** | PyTorch Sequence-to-One (SeqLen=24, Hidden=32) | 8.05 | 11.02 | 26.62% | 0.6994 | Deep Learning |
| **XGBoost (Optuna)** | Gradient Boosted Trees (Tuned on Val fold) | **6.30** | **8.30** | **21.34%** | **0.8298** | **CHAMPION** |

> **Improvement:** XGBoost delivers an error of only **6.3 sessions/hour** across the entire municipal network, achieving an $R^2$ of **0.8298**.

### 5.3 Extended Comprehensive Metrics: MSE, Accuracy, Precision, Recall, F1-Score

*(Persisted in machine-readable JSON at `models/all_models_detailed_metrics.json`)*

#### A. Detailed Metrics for Next-Hour Energy (`expected_energy_kwh`):
- Peak Demand Threshold: $\ge 500\text{ kWh}$

| Model Family | MSE | Forecast Accuracy (100 - sMAPE) | Peak Accuracy | Peak Precision | Peak Recall | Peak F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Baseline** | 34,036.44 | 69.88% | 89.90% | **95.63%** | 87.48% | 91.37% |
| **Compact LSTM** | 31,612.77 | 73.29% | 91.84% | 91.31% | **95.78%** | 93.49% |
| **XGBoost (Champion)** | **20,611.43** | **77.95%** | **92.43%** | 92.22% | 95.70% | **93.93%** |

#### B. Detailed Metrics for Next-Hour Sessions (`expected_sessions`):
- Peak Demand Threshold: $\ge 30\text{ sessions}$

| Model Family | MSE | Forecast Accuracy (100 - sMAPE) | Peak Accuracy | Peak Precision | Peak Recall | Peak F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Baseline** | 128.72 | 70.18% | 91.44% | **97.21%** | 88.44% | 92.62% |
| **Compact LSTM** | 121.53 | 73.38% | 92.75% | 91.97% | 96.50% | 94.18% |
| **XGBoost (Champion)** | **68.81** | **78.66%** | **94.06%** | 93.71% | **96.72%** | **95.19%** |

---

## 6. MLflow Tracking & Artifact Audit

All experiments, hyperparameter configurations, metrics, and models were tracked via MLflow:
- **Experiment Name:** `ev_demand_forecasting`
- **Tracking Database:** `sqlite:///mlflow.db`
- **Champion XGBoost Run ID:** `8bf6a21eef3449e39fbf662ed4405679`
- **Champion Model Artifact URI:** `runs:/8bf6a21eef3449e39fbf662ed4405679/model_energy`
- **Logged Artifacts:**
  - `holdout_forecast_comparison.png`: 168-hour continuous forecast overlay vs ground truth.
  - `residuals_distribution.png`: Centered error distribution curves.
  - `feature_importance.png`: Feature weight ranking.
  - `feature_pipeline.pkl`: Fitted scikit-learn standard scaling pipeline.

---

## 7. Person 5 Handoff & Integration Contract

### 7.1 Python Calling Contract
Person 5 can directly import and call the clean inference adapter without any notebook dependencies:

```python
from src.demand.inference import predict_demand

# Example payload
payload = {
    "timestamp": "2022-01-01T14:00:00",
    "energy_lag_1h": 420.5,
    "energy_lag_24h": 450.0,
    "sessions_lag_1h": 26.0,
    "sessions_lag_24h": 28.0,
    "temp_c": 12.5,
    "humidity": 55.0,
    "precip_mm": 0.0
}

response = predict_demand(payload)
print(response)
```

### 7.2 Response Output
```json
{
  "timestamp": "2022-01-01T14:00:00",
  "expected_energy_kwh": 500.89,
  "expected_sessions": 32.14,
  "model_name": "XGBoost_Champion",
  "model_version": "runs:/8bf6a21eef3449e39fbf662ed4405679/model_energy",
  "horizon": "1h"
}
```

---

## 8. Definition of Done Checklist Verification

- [x] **Hourly energy and session targets validated:** 17,544 continuous hours built without gaps.
- [x] **All lag/rolling tests block future information:** Automated zero-leakage unit test passes.
- [x] **Three model families tracked:** Seasonal Baseline, Compact LSTM (PyTorch), and XGBoost (Optuna).
- [x] **Metrics reported by horizon and target:** MAE, RMSE, sMAPE, and $R^2$ documented for both `energy_kwh` and `sessions`.
- [x] **Inference returns agreed JSON payload:** Verified against `schemas/demand_output.json`.
- [x] **Zero notebook dependency:** All production code lives in modular `.py` files with clean interfaces.
