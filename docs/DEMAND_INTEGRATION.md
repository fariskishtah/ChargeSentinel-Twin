# Demand integration — Jiaxing historical demo

The demand endpoint uses Waseem's repaired source commit `64eadd73175f1a063e134ecc7e346b628b8309cf`, model `demand-completion-v1`, feature version `causal-completion-v1`, and the two unchanged XGBoost champion artifacts. Final hardening restored Waseem's missing training code, fixtures, comparison artifacts, and evaluation reports from his owner branch without replacing the integrated runtime. Training code, local MLflow stores, and fixtures are not runtime dependencies. The source checksum manifest is preserved as provenance for the full original bundle; the deployment includes only its required champion subset.

`POST /api/v1/demand/forecast` accepts only `forecast_origin`, a **naive whole-hour source operational timestamp**. Source timezone is not independently verified. No UTC/Cairo conversion is guessed. This deliberately changes demand's former aware-datetime schema; all other model timestamp contracts are unchanged. Example:

```json
{"forecast_origin": "2021-10-01T00:00:00"}
```

The adapter owns a `HistoryProvider` and retrieves exactly 192 consecutive real hours immediately before origin. Origin itself and future records are excluded. The bundled asset is `team/waseem_demand/runtime_data/jiaxing-hourly-demand-v1/hourly_history.csv`: 17,547 rows, only timestamp/energy_kwh/sessions, derived using the repaired source's `build_hourly.py`. Metadata records source/derived SHA256, version, coverage and assumptions. Raw transactions remain local and immutable. No production path reads fixtures.

Supported forecast origins, inclusive: **2020-01-08T21:00:00 through 2022-01-01T00:00:00**. The last origin forecasts one hour beyond history; it needs no future actual values. Earlier/later/gapped history is rejected with 422; no history is generated or forward-filled. Zero recorded-event hours follow the training aggregation's complete-capture assumption, not verified station uptime. Reproduce from the original local dataset with:

```sh
python -m team.waseem_demand.tools.package_history --raw /path/to/Charging_Data.csv
```

Energy is final session energy recognized in its completion hour. Sessions are arrivals by start hour. The response returns timestamp, nonnegative float `expected_energy_kwh`, nonnegative float `expected_sessions`, and model_version. Expected sessions retain the statistical fractional count; they are not truncated. These are next-hour estimates and are **not physical grid-load telemetry**. Demand is deliberately **not connected to the Digital Twin or simulator power/arrival inputs**.

The API image includes history, champion bundle, NumPy/pandas/XGBoost, and libgomp; it does not require PyTorch, Optuna, SHAP, or a training MLflow store. Lazy process caches load history/boosters once and are explicitly clearable in tests. Required model files and history are checksum-verified before use. Bundle/source/feature versions are logged, without raw histories or internal file paths. Invalid inputs/history return 422; component/model failures return sanitized 503.

Limitations: historical Jiaxing demo coverage only; no Egyptian validation; source timezone and reporting latency remain unverified; complete event capture and closed-hour availability are assumed. Replacing the bundle/history requires a deployment restart to refresh cached state. A deployment requiring live station forecasts must provide a real compatible history source and documented timezone policy.

## Validation commands

```sh
MLFLOW_HOST_PORT=5050 docker compose up -d --build
RUN_DOCKER_TESTS=1 MLFLOW_HOST_PORT=5050 DOCKER_MLFLOW_URL=http://localhost:5050 python -m pytest tests -q
DOCKER_MLFLOW_URL=http://localhost:5050 python deployment/smoke_test.py
```

The existing MLflow port override is used here because macOS Control Center occupies host port 5000. Service-to-service URLs remain unchanged. The 679,720-byte history asset is the only packaged CSV; raw transactions are never copied into the API image.

Verification: **131 tests passed**, including **19 Docker tests**. The containerized Streamlit demand form displayed 670.38 kWh, 43.7 expected sessions, and `demand-completion-v1` for the supported default origin. API raw outputs remained 670.3817138671875 kWh and 43.74503707885742 sessions. All three services were healthy; remaining model endpoints returned 503; Digital Twin/simulator regression and stack smoke checks passed. New file payload is approximately 1.20 MB before Git compression, including the derived history asset; no raw transaction file is committed.
