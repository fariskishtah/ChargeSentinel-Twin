"""Minimal API schema placeholders pending model-owner handoffs."""

from pydantic import BaseModel


class SolarForecastRequest(BaseModel):
    # TODO: confirm exact forecast-origin, horizon, weather, and feature field names.
    pass


class SolarForecastResponse(BaseModel):
    timestamp: str
    horizon_hours: int
    forecast_kw: float
    model_version: str


class DemandForecastRequest(BaseModel):
    # TODO: confirm exact forecast-origin, history, and horizon field names.
    pass


class DemandForecastResponse(BaseModel):
    timestamp: str
    expected_energy_kwh: float
    expected_sessions: int
    model_version: str


class AnomalyRequest(BaseModel):
    # TODO: confirm the leakage-audited online feature names.
    pass


class AnomalyResponse(BaseModel):
    charger_id: str
    anomaly: bool
    score: float
    reason: list[str]
    model_version: str


class AvailabilityRequest(BaseModel):
    # TODO: confirm safe session/history inputs; End Time must never be included.
    pass


class AvailabilityResponse(BaseModel):
    # TODO: confirm the standalone availability response with the model owner.
    pass


class RecommendationRequest(BaseModel):
    # TODO: confirm ETA and upstream model-output field names.
    pass


class RecommendationResponse(BaseModel):
    recommended_charger: str
    expected_free_time: str
    expected_wait_minutes: float
    recommended_slot: str
    model_version: str


class DigitalTwinRequest(BaseModel):
    # TODO: confirm scenario and station-state fields before implementation.
    pass


class DigitalTwinResponse(BaseModel):
    # TODO: confirm simulated KPI and assumption fields before implementation.
    pass
