# Deployment

Docker Compose runs FastAPI, Streamlit, and MLflow on one private bridge network. All services run as non-root users, and MLflow metadata and artifacts persist in the `mlflow_data` named volume.

After `docker compose up -d` reports healthy services, run:

```bash
python deployment/smoke_test.py
RUN_DOCKER_TESTS=1 pytest -m docker
```

The smoke scenario is explicit simulated input. It does not create model predictions, MLflow experiments, or runs.

Compose exposes MLflow on port 5000 by default. If that host port is already occupied, set `MLFLOW_HOST_PORT` to another local port and set `DOCKER_MLFLOW_URL` to the matching URL when running smoke checks; MLflow remains on port 5000 inside the Compose network.
