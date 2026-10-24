"""Shared, environment-driven MLflow configuration."""

import os
from typing import Final, Literal, TypeAlias

ModuleName: TypeAlias = Literal["solar", "demand", "anomaly", "availability"]

DEFAULT_TRACKING_URI: Final = "http://127.0.0.1:5000"
EXPERIMENT_NAMES: Final[dict[ModuleName, str]] = {
    "solar": "chargesentinel-solar",
    "demand": "chargesentinel-demand",
    "anomaly": "chargesentinel-anomaly",
    "availability": "chargesentinel-availability",
}


def get_tracking_uri() -> str:
    """Return the configured tracking URI or the documented local default."""

    configured_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI).strip()
    return configured_uri or DEFAULT_TRACKING_URI


def get_experiment_name(module: ModuleName) -> str:
    """Return the canonical experiment name for a model module."""

    try:
        return EXPERIMENT_NAMES[module]
    except KeyError as exc:
        raise ValueError(f"Unknown model module: {module}") from exc


def get_or_create_experiment(module: ModuleName) -> str:
    """Configure MLflow, get/create the module experiment, and return its ID."""

    import mlflow

    mlflow.set_tracking_uri(get_tracking_uri())
    experiment = mlflow.set_experiment(get_experiment_name(module))
    return str(experiment.experiment_id)
