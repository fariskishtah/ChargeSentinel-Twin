"""Solar model adapter contract."""

from typing import Any, Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_solar(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a solar forecast after the validated model is connected."""

    del payload
    raise ModelNotReadyError("Solar forecasting model is not ready.")
