# ChargeSentinel Twin

AI-powered EV Charging Station Intelligence Platform for the Samsung Innovation Campus Smart Energy graduation project.

## Team

- Mazen — Solar Forecasting (`feature/mazen-solar`) — [tasks](team/mazen_solar/TASKS.md)
- Waseem — EV Demand Forecasting (`feature/waseem-demand`) — [tasks](team/waseem_demand/TASKS.md)
- Fatima — Anomaly Detection + XAI (`feature/fatima-anomaly`) — [tasks](team/fatima_anomaly/TASKS.md)
- Mariam — Charger Availability + Smart Recommendation (`feature/mariam-availability`) — [tasks](team/mariam_availability/TASKS.md)
- Faris — Backend, Integration, Digital Twin, Deployment (`feature/faris-integration`); integration owner

## Architecture

```text
PVGIS + Open-Meteo
        |
        v
Solar Forecast

Jiaxing
   |-- Demand Forecast
   |-- Anomaly Detection
   `-- Availability Prediction

All outputs
        |
        v
Integration / Decision Layer
        |
        v
Digital Twin -> FastAPI -> Dashboard
```

Shared code lives under `app/` and `src/`; owner modules remain under `team/`. Model handoffs enter the platform only through stable adapters in `src/integration/adapters/`.

See [branch rules](docs/BRANCH_STRATEGY.md), the [model handoff contract](docs/MODEL_HANDOFF_CONTRACT.md), [team status](docs/TEAM_STATUS.md), the [MLflow guide](docs/MLFLOW_GUIDE.md), the [Digital Twin definition](docs/DIGITAL_TWIN.md), and the [EV Charging Station Simulator guide](docs/CHARGING_STATION_SIMULATOR.md). Datasets and large model artifacts remain local and are excluded from Git.

## Run the integration shell

```bash
# Terminal 1
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
streamlit run app/ui/streamlit_app.py
```

The dashboard calls FastAPI through `CHARGESENTINEL_API_URL` and shows unavailable components without generating placeholder predictions.

## Docker Quick Start

```bash
docker compose build
docker compose up -d
docker compose ps
```

- API: <http://localhost:8000>
- API Docs: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>
- MLflow: <http://localhost:5000>

```bash
docker compose logs -f
docker compose down
```
