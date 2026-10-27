# Waseem — repaired EV demand handoff

Waseem owns the Jiaxing hourly aggregation, causal lag/rolling feature pipeline, chronological evaluation, baseline/XGBoost/LSTM comparison, bounded Optuna study, training code, reports, fixtures and portable bundle. His source history is on `feature/waseem-demand`; the older `person2-demand-forecasting` branch is superseded.

The integrated `demand-completion-v1` endpoint receives a naive whole-hour `forecast_origin` in source operational time and uses exactly 192 previous continuous recorded hours. Energy is credited at session completion; arrivals at session start. The source timezone is unverified. This is a completed-session energy demand proxy, not physical within-hour grid power. Missing or future history is rejected; no fabricated default is used. `expected_sessions` remains a nonnegative float.

The current runtime uses the unchanged XGBoost champion files in `bundle/v1/`, `demand/features.py`, `demand/inference.py`, and the small committed runtime history. Additional original evidence restored from the owner branch includes `demand/train.py`, `demand/models.py`, the baseline and LSTM candidates, scaling, evaluation/EDA/audit/Optuna/MLflow reports, fixtures, and owner tests. Those files are for reproducibility; the integrated API does not load the LSTM or overwrite current runtime files.

For local reproduction, use Python 3.12, `requirements-training.txt`, and the original Jiaxing `Charging_Data.csv` outside Git:

```bash
python -m team.waseem_demand.demand.train --raw /absolute/path/to/Charging_Data.csv --trials 6 --epochs 12
python -m pytest team/waseem_demand/tests -q
```

Raw input is read-only and hashed. Reports contain real local MLflow run receipts, but local stores are not portable/public URIs. See `reports/evaluation.md`, `bundle/v1/metadata.json`, `schemas/`, and `docs/DEMAND_INTEGRATION.md` for metrics, contract and limits. The selected champion and integrated API were not retrained in final hardening.
