# Charger Availability + Smart Recommendation

**Owner:** Mariam  
**Module:** Charger Availability + Smart Recommendation  
**Data:** Jiaxing EV charging sessions.

## Current implementation

- `notebooks/01_data_cleaning.ipynb` writes `data/processed/Charging_Data_clean.csv`, applies the shared column schema, retains each charger's first session when its own start/end are valid, removes later overlaps, and saves row-level cleaning evidence.
- `src/occupancy.py` loads the cleaned schema and provides tested interval, hourly occupancy, and replay-state functions.
- `notebooks/02_eda.ipynb` and `notebooks/03_occupancy_intervals.ipynb` use the shared occupancy functions and save EDA tables.
- `src/features.py` builds start-time-safe features, chronological splits, and a train-fitted encoder. `notebooks/04_features_and_split.ipynb` writes the separate `artifacts/Charging_Data_featureandclean.pkl` feature dataset.
- `tests/` covers cleaned-data timestamp and overlap invariants, occupancy invariants, leakage, split boundaries, and encoder behavior.
- `notebooks/05_models_baseline_rf_xgboost.ipynb` contains an existing model experiment. Reusable training, Optuna tuning, MLflow tracking, and inference APIs are still pending.

## Reproduce

From `team/mariam_availability`, run `python -m pytest tests -q`. Run notebooks `01` through `04` in order for the data preparation and feature workflow. The model notebook is an existing experiment and is outside the current task scope.

## Deliverables still pending

- Reusable `train.py`, validation/test evaluation export, Optuna study, and MLflow run URI.
- `predict_availability(payload)` and the deterministic `recommend(payload)` function with schemas and tests.

