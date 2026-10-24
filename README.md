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

See [branch rules](docs/BRANCH_STRATEGY.md), the [model handoff contract](docs/MODEL_HANDOFF_CONTRACT.md), [team status](docs/TEAM_STATUS.md), and the [MLflow guide](docs/MLFLOW_GUIDE.md). Datasets and large model artifacts remain local and are excluded from Git.
