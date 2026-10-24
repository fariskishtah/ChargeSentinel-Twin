"""Demand model adapter contract."""

from collections.abc import Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_demand(payload: Mapping[str, object]) -> dict[str, object]:
    """Return an EV demand forecast after the validated model is connected."""

    del payload
    raise ModelNotReadyError("demand")
