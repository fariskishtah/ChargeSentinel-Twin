"""Shared MLflow experiment naming and setup."""

from src.common.config import get_settings

EXPERIMENT_NAMES = {
    "solar": "chargesentinel-solar",
    "demand": "chargesentinel-demand",
    "anomaly": "chargesentinel-anomaly",
    "availability": "chargesentinel-availability",
}


def configure_mlflow(module: str) -> str:
    """Select a documented experiment and return its name."""

    if module not in EXPERIMENT_NAMES:
        raise ValueError(f"Unknown model module: {module}")

    import mlflow

    experiment_name = EXPERIMENT_NAMES[module]
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    return experiment_name
