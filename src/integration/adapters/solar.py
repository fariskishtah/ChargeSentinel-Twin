"""Solar model adapter contract."""

from collections.abc import Mapping

from src.integration.adapters.base import ModelNotReadyError


def predict_solar(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a solar forecast after the validated model is connected."""

    del payload
    raise ModelNotReadyError("solar")
