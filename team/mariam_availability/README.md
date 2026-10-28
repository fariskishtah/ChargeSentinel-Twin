# Charger Availability + Smart Recommendation

**Owner:** Mariam  
**Module:** Charger Availability + Smart Recommendation  
**Data:** Jiaxing EV charging sessions.

## Current implementation

- `notebooks/01_data_cleaning.ipynb` writes `data/processed/Charging_Data_clean.csv`, applies the shared column schema, retains each charger's first session when its own start/end are valid, removes later overlaps, and saves row-level cleaning evidence.
- `src/occupancy.py` loads the cleaned schema and provides tested interval, hourly occupancy, and replay-state functions.
- `notebooks/02_eda.ipynb` and `notebooks/03_occupancy_intervals.ipynb` use the shared occupancy functions and save EDA tables.
- `src/availability/features.py` builds start-time-safe features, chronological splits, and a train-fitted encoder. `notebooks/04_features_and_split.ipynb` writes the separate `artifacts/features_all.pkl` feature dataset.
- `tests/` covers cleaned-data timestamp and overlap invariants, occupancy invariants, leakage, split boundaries, and encoder behavior.
- `notebooks/05_models_baseline_rf_xgboost.ipynb` and `notebooks/06_optuna_mlflow.ipynb` record the model comparison, rolling-origin Optuna search, MLflow runs, validation-based champion selection, and disclosed champion test score.
- `notebooks/07_availability_inference.ipynb` contains the task 7 inference prototype that loads the duration bundle and returns predicted duration, expected free time, and remaining wait from `as_of`.
- `src/availability/train.py` is the reusable training entry point. It fits the median lookup, Random Forest, and XGBoost candidates; tunes XGBoost only on chronological folds inside training; selects by validation MAE; logs runs to MLflow; and exports comparison, selection, test, and handoff reports.
- `src/availability/inference.py` loads the saved bundle and exposes `predict_availability(payload)`. `src/recommendation/service.py` exposes integrated `recommend(payload)`, combining live/simulated busy state, availability predictions, historical risk, and the deterministic ranker.
- `schemas/` contains the availability and recommendation input/output contracts. `tests/fixtures/` contains example requests.
- The current evidence selects default XGBoost: its normal-session validation MAE is 43.404 minutes versus 43.398 for tuned XGBoost. The tuned gain (0.013%) is below the 1% threshold. The selected model's recorded test MAE is 34.948 minutes and RMSE is 83.049 minutes for normal sessions.
- The data contains 441,064 cleaned sessions (SHA-256 prefix `e518e9c6d140`). The test partition had already been scored before tuning; that fact and the earlier metrics file are disclosed in the selection and MLflow handoff reports. Treat the reported test result as a documented retrospective score, not a pristine untouched estimate.
- Held-out normal-session test MAE is 34.95 minutes for the champion and 35.52 for the median baseline. The test partition was previously scored; this comparison is disclosed in `availability_duration_evaluation.md`.
- Peak replay covers 39,204 held-out arrivals. A charger was free at every replay arrival, so ranker, first-free, and random-free all had zero realized wait. Random-free distributed assignments more evenly (CV 0.433 vs ranker 1.387); the ranker's observed later-fault proxy was 23.2% vs 16.5% for first-free. These replay measures are descriptive, with counterfactual and queue-data limitations documented in `availability_recommendation_evaluation.md`.

## Reproduce

From `team/mariam_availability`, install `requirements.txt`, prepare `data/processed/Charging_Data_clean.csv` by running notebook `01` (notebooks `02`–`04` reproduce EDA and feature artifacts), then reproduce training with `python -m src.availability.train --trials 25 --timeout 2700`. Compare the saved champion with the median baseline using `python -m src.availability.evaluate`, run peak-hour recommendation replay with `python -m src.recommendation.replay`, and run `python -m pytest tests -q -p no:cacheprovider` for module tests. The default MLflow store and model bundle are local under `artifacts/`; set `CHARGESENTINEL_MLFLOW_TRACKING_URI` for a shared tracking server. Local model and MLflow binaries are ignored by Git and must be regenerated elsewhere.

## Person 5 handoff

`HANDOFF_PERSON5.md` documents the exact imports, example calls and runtime files. `recommend()` defaults to the saved training-period charger risk table; Person 3 is disabled by `use_person3_score: false`. Demand and solar remain neutral placeholders until Persons 1 and 2 provide their signals. Additive Person 5 schemas are in this folder's `schemas/` directory because the team handoff rule limits edits to this folder.

## Recommendation ranker

`src/recommendation/ranker.py` and `config.yaml` implement the versioned deterministic score and alternatives. Scores are lower-is-better: normalized wait, historical anomaly risk, and demand pressure add cost; normalized solar benefit subtracts cost. If a same-location charger is actually free, the ranker limits scoring to the free chargers; when all are busy, it ranks predicted release times. Until demand and solar models are integrated, configured neutral placeholders apply when signals are omitted and the response marks `mock_signals_used: true`. Historical risk comes from `artifacts/charger_risk_table.json`, built from pre-2021-07-01 training sessions; unseen chargers use its global rate. Person 3 scores are ignored by default. Use consistent local timestamps because the dataset timezone is unresolved.

