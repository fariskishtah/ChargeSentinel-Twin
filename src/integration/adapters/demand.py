"""Real demand model with cached artifacts and explicit real historical context."""

from collections.abc import Mapping

from functools import lru_cache
import json
import logging

from src.integration.demand_history import (LocalJiaxingHistoryProvider, DemandHistoryError,
    DemandUnavailableError, operational_origin, digest)

logger = logging.getLogger(__name__)


class DemandService:
    def __init__(self, history_provider=None):
        from team.waseem_demand.demand.inference import load_bundle, load_xgboost
        from team.waseem_demand.demand.features import FEATURES
        try:
            path, metadata = load_bundle()
            checksums = json.loads((path / "checksums.json").read_text())
            for name in ["metadata.json", "features.json", "xgboost_energy_kwh.json", "xgboost_sessions.json"]:
                if digest(path / name) != checksums[name]:
                    raise ValueError("Bundle checksum mismatch")
            if (metadata["model_version"] != "demand-completion-v1"
                    or metadata["feature_version"] != "causal-completion-v1"
                    or metadata["features"] != FEATURES
                    or metadata["champions"] != {"energy_kwh": "XGBoost", "sessions": "XGBoost"}):
                raise ValueError("Unapproved demand bundle")
            self.history_provider = history_provider or LocalJiaxingHistoryProvider()
            if metadata["dataset_version"] != "sha256:" + self.history_provider.metadata["source_sha256"]:
                raise ValueError("History/model source mismatch")
            for c in ["energy_kwh", "sessions"]:
                load_xgboost(path / f"xgboost_{c}.json")
            self.model_version = metadata["model_version"]
        except Exception as exc:
            logger.error("Demand bundle unavailable (%s)", type(exc).__name__)
            raise DemandUnavailableError("Demand component unavailable.") from exc
        logger.info("Demand model loaded: version=%s feature_version=%s training_commit=%s",
                    self.model_version, metadata["feature_version"], metadata["training_code_commit"])

    def predict(self, payload):
        from team.waseem_demand.demand.inference import predict_demand as model_predict
        from app.api.schemas import DemandForecastResponse
        origin = operational_origin(payload.get("forecast_origin"))
        history = self.history_provider.get_history(origin.to_pydatetime())
        try:
            output = model_predict({"forecast_origin": origin.isoformat(), "history": history})
            validated = DemandForecastResponse(**output)
            if validated.timestamp != origin.to_pydatetime() or validated.model_version != self.model_version:
                raise ValueError("Unexpected model output")
            logger.info("Demand inference success: origin=%s model=%s", origin.isoformat(), self.model_version)
            return validated.model_dump()
        except Exception as exc:
            logger.error("Demand inference failed (%s)", type(exc).__name__)
            raise DemandUnavailableError("Demand inference unavailable.") from exc


@lru_cache(maxsize=1)
def get_demand_service():
    return DemandService()


def predict_demand(payload: Mapping[str, object]) -> dict[str, object]:
    """No synthetic input, fixture fallback, or kWh-to-kW conversion."""
    operational_origin(payload.get("forecast_origin"))
    return get_demand_service().predict(payload)
