"""Pydantic API contracts."""

from app.api.schemas.models import (
    AnomalyRequest,
    AnomalyResponse,
    AvailabilityRequest,
    AvailabilityResponse,
    DemandForecastRequest,
    DemandForecastResponse,
    DigitalTwinRequest,
    DigitalTwinResponse,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    SolarForecastRequest,
    SolarForecastResponse,
)

__all__ = [
    "AnomalyRequest",
    "AnomalyResponse",
    "AvailabilityRequest",
    "AvailabilityResponse",
    "DemandForecastRequest",
    "DemandForecastResponse",
    "DigitalTwinRequest",
    "DigitalTwinResponse",
    "HealthResponse",
    "RecommendationRequest",
    "RecommendationResponse",
    "SolarForecastRequest",
    "SolarForecastResponse",
]
