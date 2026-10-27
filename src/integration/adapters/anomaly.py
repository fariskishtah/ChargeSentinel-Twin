"""Validated portable Fatima anomaly handoff adapter."""

import logging
from collections.abc import Mapping

from team.fatima_anomaly.integration.features import feature_record
from team.fatima_anomaly.integration.inference import predict_anomaly as real_predict_anomaly

logger = logging.getLogger(__name__)


class AnomalyUnavailableError(RuntimeError):
    pass


def predict_anomaly(payload: Mapping[str, object]) -> dict[str, object]:
    feature_record(str(payload["charger_id"]), dict(payload["features"]))
    try:
        result = real_predict_anomaly(dict(payload))
        from app.api.schemas import AnomalyResponse
        return AnomalyResponse(**result).model_dump(mode="json")
    except Exception as exc:
        logger.error("Anomaly inference unavailable (%s)", type(exc).__name__)
        raise AnomalyUnavailableError("Anomaly component unavailable.") from exc
