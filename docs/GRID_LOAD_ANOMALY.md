# Historical grid-load interval anomaly screen

The Samsung Smart Energy brief asks for abnormal generation or grid-load **intervals**. Session-start abnormal-ending risk is a separate EV session task, so `anomaly-v1` remains unchanged. This detector screens hourly Jiaxing completed-session energy and arrival counts as a **demand proxy**, not physical grid telemetry.

## Method and evidence

- Source: committed `jiaxing-hourly-demand-v1/hourly_history.csv`, SHA256 recorded in `src/grid_load/grid-load-v1.json`. Energy is assigned to completion hour; sessions count arrivals at start hour. Source timezone is unverified.
- For hour *t*, compare observed energy and sessions with the eight observations at *t − 168h, −336h, …, −1344h*. Use the median and a robust MAD scale with documented floors. No centered window or later observation enters the score.
- Score is the larger absolute standardized deviation. Threshold **5.827239498** is the 99th percentile of natural, unlabeled January–June 2021 validation scores. The July–December 2021 test was not used to set it.
- A 100-tree Isolation Forest trained on 2020 causal features is a comparison only. Its score threshold also uses the January–June 2021 natural-score 99th percentile. The rule was selected for causal interpretability, portable serving, and better validation sensitivity at the same alert budget.
- The natural test alert rate is **0.70%** for the rule and **14.29%** for Isolation Forest. These are *not* false-positive rates because there are no maintenance-confirmed hourly fault labels.
- Held-out sensitivity check: 13 fixed, spaced three-hour test windows have **SIMULATED / INJECTED VALIDATION CASES** (`observed energy × 1.8 + 200 kWh`, `arrivals × 1.5 + 10`). Rule detection is **24/39 hours (61.54%)**; Isolation Forest is **27/39 (69.23%)**. Injection never changes source history. These rates are not real fault recall.

The exact split boundaries, thresholds, parameters, each injected window, and results are in `team/fatima_anomaly/reports/grid_load/`. Reproduce with `python -m src.grid_load.train` in an environment with NumPy and scikit-learn. The served rule uses only the Python standard library and a small JSON configuration.

## Serving contract and limits

`POST /api/v1/anomaly/grid-load` accepts `{"timestamp":"2021-10-01T00:00:00"}` in naive source operational time, for a recorded historical hour with eight earlier weekly peers. It returns timestamp, anomaly, nonnegative score, threshold, reason, detector version, recorded energy/arrivals, and `is_simulated: false`. Unknown intervals are 422; corrupt/missing assets are 503. The Streamlit card is labeled separately from **EV Session Risk**.

An alert means **unusual load interval for review**. It does not diagnose a charger, panel, inverter, or grid fault. The test has injected sensitivity evidence, not fault-label performance. The Jiaxing proxy is neither measured Egyptian grid load nor production validated in Egypt.
