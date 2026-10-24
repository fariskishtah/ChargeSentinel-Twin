"""Availability and recommendation adapter contracts."""

from typing import Any, Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_availability(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return availability after the validated model is connected."""

    del payload
    raise ModelNotReadyError("Charger availability model is not ready.")


def recommend(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recommendation after the validated ranker is connected."""

    del payload
    raise ModelNotReadyError("Charger recommendation service is not ready.")
