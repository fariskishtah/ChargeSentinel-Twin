# Waseem — repaired EV demand handoff

All implementation is owned by this directory. Original submission `450c11c` is preserved on `person2-demand-forecasting`. Reused concepts: seasonal/calendar blend, XGBoost/Optuna, compact LSTM, lag/rolling features, and chronological evaluation. Old model weights are deliberately replaced after removing leaking observed weather.

## Serving contract

`forecast_origin` is the boundary starting the one-hour forecast interval. Supply at least 192 continuous, real, closed hourly buckets ending at `forecast_origin - 1 hour`. Each history record has `timestamp`, `energy_kwh`, and integer `sessions`. Energy means final session energy recognized in its **completion hour**; sessions mean arrivals in their **start hour**. This choice prevents assigning final future session outcomes to earlier historical hours. It does **not** estimate physical within-hour charging load.

Source timezone not independently verified. Preserve local/naive operational time; do not label it UTC. Aware timestamps are rejected. The adapter must obtain a documented source-time mapping before translating an aware Faris request. No such translation is guessed here.

Missing/gapped/invalid/future history raises `ValueError`; there are no fabricated history defaults. The caller/history provider owns real event capture and hourly finalization. Reporting delays are not modeled. `expected_sessions` is a nonnegative **float** expectation; Faris must deliberately adapt its integer response field. Faris also needs to supply actual hourly history or a provider before wiring this module. No shared API has been changed.

From the repository root:

```python
import json
from pathlib import Path
from team.waseem_demand.demand import predict_demand

module = Path("team/waseem_demand")
request = json.loads((module / "fixtures/request.json").read_text())
result = predict_demand(request)
```

When installed via `PYTHONPATH=/absolute/path/to/team/waseem_demand`, use `from demand import predict_demand` from any CWD. Default artifacts resolve from the module location. Optional `CHARGESENTINEL_DEMAND_BUNDLE` must be an absolute path. Bundle `v1` contains stable XGBoost JSON, seasonal JSON, LSTM weights, train-only scaling, ordered features, metadata, and checksums. Inference only imports PyTorch if a target selects LSTM; check `bundle/v1/metadata.json` before choosing deployment dependencies.

## Reproduce

Use Python 3.12 and a dedicated environment. Install `requirements-training.txt`. Runtime dependencies are separate in `requirements-runtime.txt`; `reports/environment.txt` records the exact training environment.

```sh
python -m team.waseem_demand.demand.train --raw /absolute/path/to/Charging_Data.csv --trials 6 --epochs 12
python -m pytest team/waseem_demand/tests -q
```

Raw input is read-only and hashed before/after. Full derived hourly data, MLflow SQLite store and MLflow artifacts remain under ignored `runtime/`; no large dataset or database is committed. Set `MLFLOW_TRACKING_URI` explicitly for a shared server. Real run IDs and local bundle artifact URIs are in `reports/mlflow_receipt.json`. The committed bundle is independently portable; local run URIs are not advertised as public.

The first environment hit a native XGBoost crash and produced no metrics. Training completed with the pinned NumPy/XGBoost/PyTorch stack; its exact packages are in `reports/environment.txt`. Tracking was then resumed from the frozen bundle after pinning SQLAlchemy 2.0.44 for MLflow compatibility; `reports/tracking_environment.txt` records that environment. `python -m team.waseem_demand.demand.train --log-only` resumes tracking without retraining or changing metrics.

## Handoff and limitations

- Function: `demand/inference.py`, `predict_demand(payload)`.
- Preprocessing: `demand/features.py`; train-only scaling in bundle.
- Artifacts/version/features: `bundle/v1/`.
- Contracts: `schemas/input.json` and `schemas/output.json`.
- Real historical fixture and expected output: `fixtures/`.
- Metrics, audit, plots, bounded tuning, MLflow receipt: `reports/`.
- Tests: `tests/`; dependencies: the two module-specific manifests.
- Explicit limitations: `reports/evaluation.md`.

The model is a rolling next-hour forecast for the Jiaxing source population. It is not an Egyptian station model or an open-loop multihour forecast. Zero event hours assume complete capture, not verified station uptime. Weather/TOU are excluded. Completed energy is not physical grid load. Recent real history is required; statistical expected sessions are not exact arrivals. Train/validation/test use one historical chronological partition. These constraints must travel with any downstream integration.
