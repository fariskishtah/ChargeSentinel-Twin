"""Stable model adapter entry points."""

from src.integration.adapters.anomaly import predict_anomaly
from src.integration.adapters.availability import predict_availability, recommend
from src.integration.adapters.demand import predict_demand
from src.integration.adapters.solar import predict_solar

__all__ = [
    "predict_anomaly",
    "predict_availability",
    "predict_demand",
    "predict_solar",
    "recommend",
]
