"""Portable Mariam availability model and deterministic recommendation adapter."""

from collections.abc import Mapping
from functools import lru_cache

from src.integration.adapters.base import ModelNotReadyError
from team.mariam_availability.src.availability.inference import AvailabilityPredictor
from team.mariam_availability.src.recommendation.risk import load_risk_table
from team.mariam_availability.src.recommendation.service import recommend as owner_recommend


@lru_cache(maxsize=1)
def _predictor() -> AvailabilityPredictor:
    try:
        return AvailabilityPredictor()
    except (FileNotFoundError, OSError, EOFError, KeyError, ValueError) as exc:
        raise ModelNotReadyError("availability") from exc


def predict_availability(payload: Mapping[str, object]) -> dict[str, object]:
    """Estimate free time from explicit session-start context, never End Time."""

    return _predictor().predict(payload)


def recommend(payload: Mapping[str, object]) -> dict[str, object]:
    """Rank caller-supplied historical candidate states; exclude fault/offline."""

    predictor = _predictor()
    try:
        load_risk_table()
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        raise ModelNotReadyError("recommendation") from exc
    result = owner_recommend(payload, predictor=predictor)
    if result.get("recommended_charger") is None:
        raise ValueError("No eligible charger at the requested location")
    return result
