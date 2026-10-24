# ChargeSentinel Twin

AI-powered EV Charging Station Intelligence Platform for the Samsung Innovation Campus Smart Energy graduation project.

## Team

- Mazen - Solar Forecasting
- Waseem - EV Demand Forecasting
- Fatima - Anomaly Detection + XAI
- Mariam - Charger Availability + Smart Recommendation
- Faris - Backend, Integration, Digital Twin, Deployment

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

Working branches and pull-request rules are documented in `docs/BRANCH_STRATEGY.md`. Datasets and large model artifacts remain local and are excluded from Git.
