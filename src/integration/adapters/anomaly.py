"""Anomaly model adapter contract."""

from typing import Any, Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_anomaly(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an anomaly score after the validated model is connected."""

    del payload
    raise ModelNotReadyError("Anomaly detection model is not ready.")
