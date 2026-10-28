# Final historical demo validation

**Date:** 2026-10-28 (Africa/Cairo). **Candidate runtime commit:** `338c977e061df646717d4156182d2af3c399b003` on `feature/faris-integration`. This was a fresh temporary clone of the remote branch, with a new Python 3.12 virtual environment installed using `pip install --no-cache-dir -r requirements/dev.txt`. The documentation-only release gate commit follows this validation; no serving code or model artifact changed.

## Reproduction commands and outcome

| Step | Command / check | Result |
|---|---|
| Clean clone and dependencies | `git clone --branch feature/faris-integration --single-branch <remote> <temporary directory>`; Python 3.12 venv; `pip install --no-cache-dir -r requirements/dev.txt` | PASS. No existing project virtualenv used. |
| Isolated practical suite | `python -m pytest tests -q` | PASS: 133 passed, 22 Docker-gated tests skipped before stack startup. |
| Fresh Docker images | `docker compose build --no-cache` | PASS: API, UI and MLflow images built. An initial attempt stalled when Docker Desktop ran low on host space; after recovering Docker and clearing unused build artifacts, the new no-cache build completed. |
| Start and health | `MLFLOW_HOST_PORT=5050 docker compose up -d`; `docker compose ps`; `curl -f` for `:8000/health`, `:8501/_stcore/health`, `:5050/health` | PASS: all three containers healthy and all health endpoints HTTP 200. |
| Stack smoke | `DOCKER_MLFLOW_URL=http://localhost:5050 python deployment/smoke_test.py` | PASS: health, OpenAPI, UI import/logs and API connection, demand, both solar horizons, session risk, grid-load interval and Digital Twin. |
| Docker-enabled suite | `RUN_DOCKER_TESTS=1 MLFLOW_HOST_PORT=5050 DOCKER_MLFLOW_URL=http://localhost:5050 python -m pytest tests -q --disable-warnings` | PASS: 155 passed, one warning, zero failures. |
| Restored demand owner suite | `python -m pytest team/waseem_demand/tests -q --disable-warnings` in its separate training-dependency environment | PASS: 34 passed, zero failures. |
| Evidence-only MLflow registration | `MLFLOW_TRACKING_URI=http://localhost:5050 python docs/experiment_evidence/register_metadata.py` from the working branch | PASS: four receipt runs; a second invocation reused all four without duplicates. |

## Runbook sequence

All POSTs used `Content-Type: application/json` and the payloads in `docs/DEMO_RUNBOOK.md`. The table records the actual clean-clone responses; it does not replace the source reports or claim live production input.

| Order | Endpoint / input | Observed result |
|---|---|---|
| 1–2 | `POST /api/v1/solar/forecast`, origin `2024-06-15T00:00:00+00:00`, horizons 24 and 48 | HTTP 200; 24 points, `solar-seasonal-persistence-24h-v1`; 48 points, `solar-ridge-regression-48h-v1`. |
| 3 | `POST /api/v1/demand/forecast`, origin `2021-10-01T00:00:00` | HTTP 200; `demand-completion-v1`; 670.3817 kWh and 43.7450 expected sessions. |
| 4 | `POST /api/v1/anomaly/predict`, charger `12`, Financial Industrial Park / Nanhu, start `2021-10-16T12:02:44` | HTTP 200; `anomaly-v1`; score 0.50585, threshold 0.40, SHAP-based reasons. |
| 5 | `POST /api/v1/anomaly/grid-load`, `2021-10-02T13:00:00` | HTTP 200; `grid-load-seasonal-rule-v1`; anomaly true, score 8.2143 versus threshold 5.8272, 1165.13 kWh/125 sessions, `is_simulated=false`. This is a recorded charging-load interval, not a confirmed fault. |
| 6 | `POST /api/v1/digital-twin/simulate`, runbook scenario (60 kW solar, 100 kW demand, multipliers 0.5/1.3, one offline charger) | HTTP 200; `is_simulation=true`; 30 kW scenario solar, 130 kW demand, −100 kW balance, HIGH severity. |
| 7 | `POST /api/v1/simulator/reset` `{}`, add `EV-DEMO` (20% to 80%, 60 kWh, 11 kW), advance 15 minutes, reset again | All HTTP 200; EV entered charging, received 2.75 kWh, reached 24.5833% SOC; final reset succeeded. |
| 8 | `http://localhost:5050/health` and MLflow experiment `chargesentinel-release-evidence` | HTTP 200; the browser run list visibly showed four matching runs. Each has a receipt artifact, `run_kind=artifact/evidence registration` and `original_training_run=false`. These are **not** original training runs. |
| 9 | `POST /api/v1/chargers/availability` and `/recommend` with valid schema inputs | Both HTTP 503, `model_not_ready`, as intended. |

The UI health endpoint passed; Docker-enabled tests also exercised Streamlit rendering and API-client flows. The MLflow server belongs to this temporary Compose project and does not make the original local training stores portable. `docs/experiment_evidence/` remains the committed evidence source.

## Presentation limits

PVGIS is a modelled New Cairo solar reference. Jiaxing completed-session energy and arrivals are charging-demand proxies; the grid-load screen has no feeder telemetry, maintenance-confirmed fault labels or Egyptian grid validation. Injected anomalies are sensitivity cases, not observed production faults. Session risk estimates a later abnormal end cause, not an equipment fault. Solar, demand and anomaly inference use supported historical origins. Availability and recommendation remain intentionally unavailable. The grid-load UI card does not yet provide an in-UI historical time-series chart; `docs/SAMSUNG_COMPLIANCE_MATRIX.md` records this as partial.
