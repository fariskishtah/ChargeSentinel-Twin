"""Demand model adapter contract."""

from typing import Any, Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_demand(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return an EV demand forecast after the validated model is connected."""

    del payload
    raise ModelNotReadyError("EV demand forecasting model is not ready.")
