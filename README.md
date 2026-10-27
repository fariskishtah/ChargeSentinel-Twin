# ChargeSentinel Twin

Samsung Innovation Campus Smart Energy historical demonstration: renewable PV forecasting, EV completed-session energy demand forecasting, session-start abnormal-ending risk, and a separate hourly load-proxy anomaly screen. FastAPI serves strict Pydantic contracts to Streamlit; Docker Compose also runs MLflow. The Digital Twin and Charging Station Simulator are deterministic what-if tools, not live controllers.

## Current functional status

| Component | Status | Version / limitation |
|---|---|---|
| Solar 24h | Integrated | Seasonal Persistence; 2024 H1 PVGIS modelled/reference PV, not measured station output. |
| Solar 48h | Integrated | Ridge Regression; same historical reference. |
| EV demand | Integrated | `demand-completion-v1`; Jiaxing completed-session energy proxy, not physical instantaneous grid load. |
| EV Session Risk | Integrated | `anomaly-v1`; chronological session-start model, 0.40 threshold, real SHAP. Not a confirmed equipment fault. |
| Grid-Load Interval Anomaly | Integrated historical screen | `grid-load-seasonal-rule-v1`; unusual Jiaxing hourly demand-proxy intervals, no real grid-fault labels. |
| Digital Twin / Charging Station Simulator | Working simulations | Explicit scenario inputs, deterministic outputs. |
| Availability / Recommendation | Not integrated | Safe `ModelNotReady` / 503; no fabricated results. |

The project is **not production validated in Egypt**. See [Samsung compliance](docs/SAMSUNG_COMPLIANCE_MATRIX.md) for evidence and remaining limitations, and [validation](docs/VALIDATION.md) for the final test gate.

## Setup and launch

Python **3.12** matches both API and UI Docker base images. For local work, create a fresh Python 3.12 virtual environment, install `requirements/dev.txt`, then run `python -m pytest tests -q`. Training-only demand dependencies are in `team/waseem_demand/requirements-training.txt`; the grid-load comparison needs scikit-learn. Raw source data is not needed for portable runtime inference.

```bash
docker compose build
MLFLOW_HOST_PORT=5050 docker compose up -d
docker compose ps
DOCKER_MLFLOW_URL=http://localhost:5050 python deployment/smoke_test.py
```

API: <http://localhost:8000/docs>; Streamlit: <http://localhost:8501>; MLflow: <http://localhost:5050>. Use `docker compose down` when finished. For a deterministic walk-through with valid historical dates and exact payloads, follow [the demo runbook](docs/DEMO_RUNBOOK.md). Docker MLflow storage is fresh unless runs are imported or reproduced; committed receipts and reports remain the evidence index.

## Architecture and reproducibility

The actual data → pipeline → portable bundle → adapter → API/UI flow is in [ARCHITECTURE.md](docs/ARCHITECTURE.md) and editable [Mermaid source](docs/architecture.mmd). `app/` holds API and UI; `src/` holds integration, grid-load, shared and simulation logic; `team/` holds owner modules, bundles and evidence; `tests/` covers contracts and regressions; `deployment/` contains Docker and smoke checks.

Data references and checksums: [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). Field meanings: [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md). Chronological boundaries: [SPLITS.md](docs/SPLITS.md). Leakage controls: [PREPROCESSING.md](docs/PREPROCESSING.md). The raw 441,077-row Jiaxing CSV, raw PVGIS/Open-Meteo files, local MLflow stores, and experiment binaries are excluded from Git. Small derived runtime assets and selected portable bundles are committed and checksum checked.

Model comparison, tuning and held-out evidence live in `team/waseem_demand/reports/`, `team/mazen_solar/reports/integration_hardening/`, `team/fatima_anomaly/reports/integration_hardening/`, and `team/fatima_anomaly/reports/grid_load/`. Use `python -m src.grid_load.train` in a Python environment with NumPy/scikit-learn to reproduce the grid-load comparison. Do not interpret injected sensitivity cases as real fault labels.

## Team and contribution history

Waseem owns EV demand; Mazen owns solar; Fatima owns the original anomaly pipeline and XAI; Faris owns backend/integration, leakage-safe anomaly retraining, Digital Twin, simulator, Docker and release engineering. Mariam's availability/recommendation scope is not integrated. See [TEAM_CONTRIBUTIONS.md](docs/TEAM_CONTRIBUTIONS.md) and [TEAM_STATUS.md](docs/TEAM_STATUS.md). Fatima's old PR #1 was closed without merging its completed-session model; her branch and contribution history remain intact.
