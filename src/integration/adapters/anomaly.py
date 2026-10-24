"""Anomaly model adapter contract."""

from collections.abc import Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_anomaly(payload: Mapping[str, object]) -> dict[str, object]:
    """Return an anomaly score after the validated model is connected."""

    del payload
    raise ModelNotReadyError("anomaly")
