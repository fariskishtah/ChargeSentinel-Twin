"""Availability and recommendation adapter contracts."""

from collections.abc import Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_availability(payload: Mapping[str, object]) -> dict[str, object]:
    """Return availability after the validated model is connected."""

    del payload
    raise ModelNotReadyError("availability")


def recommend(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a recommendation after the validated ranker is connected."""

    del payload
    raise ModelNotReadyError("recommendation")
