"""Validated portable Mazen solar handoff adapter."""

import logging
from collections.abc import Mapping

from team.mazen_solar.integration.inference import SolarInputError, predict_solar as real_predict_solar

logger = logging.getLogger(__name__)


class SolarUnavailableError(RuntimeError):
    pass


def predict_solar(payload: Mapping[str, object]) -> dict[str, object]:
    try:
        result = real_predict_solar(dict(payload))
        from app.api.schemas import SolarForecastResponse
        checked = SolarForecastResponse(**result)
        if len(checked.points) != checked.horizon_hours:
            raise ValueError("Solar output length mismatch.")
        return checked.model_dump(mode="json")
    except SolarInputError:
        raise
    except Exception as exc:
        logger.error("Solar inference unavailable (%s)", type(exc).__name__)
        raise SolarUnavailableError("Solar component unavailable.") from exc
