"""Version 1 API contracts shared by the integration layer and model owners."""

from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
    NaiveDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

Identifier: TypeAlias = Annotated[str, Field(min_length=1, max_length=128)]
TextValue: TypeAlias = Annotated[str, Field(min_length=1)]
FeatureValue: TypeAlias = str | int | float | bool | None


class APIModel(BaseModel):
    """Strict base model for the public v1 contract."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        str_strip_whitespace=True,
    )


class SolarForecastRequest(APIModel):
    forecast_origin: AwareDatetime
    horizon_hours: Literal[24, 48]

    @field_validator("forecast_origin")
    @classmethod
    def require_utc_whole_hour(cls, value):
        if value.utcoffset().total_seconds() != 0 or value.minute or value.second or value.microsecond:
            raise ValueError("Solar forecast_origin must be a whole UTC hour")
        return value


class SolarForecastPoint(APIModel):
    timestamp: AwareDatetime
    forecast_kw: float = Field(ge=0, le=100)


class SolarForecastResponse(APIModel):
    timestamp: AwareDatetime
    horizon_hours: Literal[24, 48]
    forecast_kw: float = Field(ge=0)
    model_version: Identifier
    model_family: Identifier | None = None
    points: list[SolarForecastPoint] = Field(default_factory=list)


class DemandForecastRequest(APIModel):
    forecast_origin: NaiveDatetime = Field(description="Whole-hour source operational time; timezone unverified, not UTC")

    @field_validator("forecast_origin")
    @classmethod
    def require_whole_hour(cls, value):
        if value.minute or value.second or value.microsecond:
            raise ValueError("Demand forecast_origin must be a whole hour")
        return value


class DemandForecastResponse(APIModel):
    timestamp: NaiveDatetime
    expected_energy_kwh: float = Field(ge=0)
    expected_sessions: float = Field(ge=0, description="Statistical expected arrival count, not rounded")
    model_version: Identifier


class AnomalyStartFeatures(APIModel):
    location: Identifier
    district: Identifier
    start_time: NaiveDatetime = Field(description="Session start in source operational time; timezone unverified")


class AnomalyRequest(APIModel):
    charger_id: Identifier
    features: AnomalyStartFeatures


class AnomalyResponse(APIModel):
    charger_id: Identifier
    anomaly: bool
    score: float = Field(ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)
    reason: list[TextValue]
    model_version: Identifier


class GridLoadRequest(APIModel):
    timestamp: NaiveDatetime = Field(description="Historical Jiaxing whole hour; source timezone unverified")


class GridLoadResponse(APIModel):
    timestamp: NaiveDatetime
    anomaly: bool
    score: float = Field(ge=0)
    threshold: float = Field(ge=0)
    reason: list[TextValue]
    detector_version: Identifier
    energy_kwh: float = Field(ge=0)
    sessions: int = Field(ge=0)
    is_simulated: Literal[False] = False


class AvailabilityHistory(APIModel):
    """Caller-supplied context known when the current session started."""

    idle_before_min: float | None = Field(ge=0)
    prev_duration_min: float | None = Field(ge=0)
    hist_median_normal: float | None = Field(ge=0)
    hist_mean_normal: float | None = Field(ge=0)
    hist_abnormal_rate: float | None = Field(ge=0, le=1)
    sessions_before: int = Field(ge=0)
    has_history: Literal[0, 1]
    site_n_chargers: int = Field(ge=1)
    site_busy_others: float = Field(ge=0)
    site_busy_share: float = Field(ge=0, le=1)
    site_starts_last_60min: float = Field(ge=0)


class AvailabilityRequest(APIModel):
    charger_id: Identifier
    location: Identifier
    district: Identifier
    session_start: NaiveDatetime = Field(description="Jiaxing source operational time; timezone unverified")
    as_of: NaiveDatetime = Field(description="Historical prediction time; no future End Time supplied")
    history_features: AvailabilityHistory
    temp_c: float | None = Field(default=None, ge=-100, le=100)
    humidity: float | None = Field(default=None, ge=0, le=100)
    precip_mm: float | None = Field(default=None, ge=0)
    explain: bool = False

    @model_validator(mode="after")
    def require_supported_historical_interval(self) -> "AvailabilityRequest":
        if self.as_of < self.session_start:
            raise ValueError("as_of cannot precede session_start")
        if not datetime(2019, 12, 31) <= self.session_start < datetime(2022, 1, 1):
            raise ValueError("session_start is outside the Jiaxing historical support window")
        if self.as_of >= datetime(2022, 1, 1):
            raise ValueError("as_of is outside the Jiaxing historical support window")
        return self


class AvailabilityResponse(APIModel):
    charger_id: Identifier
    predicted_duration_minutes: float = Field(ge=0)
    expected_free_time: NaiveDatetime
    expected_wait_minutes: float = Field(ge=0)
    anomaly_risk: float | None = Field(ge=0, le=1)
    model_version: Identifier
    bundle_version: Identifier
    config_version: Identifier
    schema_version: Identifier
    limitations: list[TextValue]
    explanation: dict[str, object] | None = None
    explanation_note: str | None = None


class RecommendationCandidate(APIModel):
    charger_id: Identifier
    location: Identifier
    district: Identifier | None = None
    busy: bool
    status: Literal["AVAILABLE", "CHARGING", "FAULT", "OFFLINE", "UNAVAILABLE", "RESERVED"]
    session_start: NaiveDatetime | None = None
    history_features: AvailabilityHistory | None = None
    temp_c: float | None = Field(default=None, ge=-100, le=100)
    humidity: float | None = Field(default=None, ge=0, le=100)
    precip_mm: float | None = Field(default=None, ge=0)
    demand_pressure: float | None = Field(default=None, ge=0, le=1)
    solar_benefit: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def require_consistent_state(self) -> "RecommendationCandidate":
        if self.status == "AVAILABLE" and self.busy:
            raise ValueError("AVAILABLE charger cannot be busy")
        if self.status == "CHARGING":
            if not self.busy or self.session_start is None or self.history_features is None:
                raise ValueError("CHARGING charger needs busy=true, session_start and history_features")
        return self


class RecommendationRequest(APIModel):
    location: Identifier
    user_eta: NaiveDatetime = Field(description="Jiaxing source operational time; historical replay")
    candidates: list[RecommendationCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def require_supported_historical_time(self) -> "RecommendationRequest":
        if not datetime(2019, 12, 31) <= self.user_eta < datetime(2022, 1, 1):
            raise ValueError("user_eta is outside the Jiaxing historical support window")
        return self


class RecommendationResponse(APIModel):
    recommended_charger: Identifier
    location: Identifier
    expected_free_time: NaiveDatetime
    recommended_slot: NaiveDatetime
    expected_wait_minutes: float = Field(ge=0)
    model_version: Identifier
    bundle_version: Identifier
    config_version: Identifier
    schema_version: Identifier
    risk_source: Identifier
    score: float
    score_components: dict[str, float]
    explanation: dict[str, object]
    mock_signals_used: bool
    free_pool_restricted: bool
    reason: list[TextValue]
    alternatives: list[dict[str, object]]
    limitations: list[TextValue]
    comparison: dict[str, object] | None = None


class DigitalTwinRequest(APIModel):
    baseline_solar_kw: float = Field(ge=0)
    baseline_demand_kw: float = Field(ge=0)
    total_chargers: int = Field(gt=0)
    active_chargers: int = Field(ge=0)
    average_session_minutes: float = Field(gt=0)
    demand_multiplier: float = Field(default=1.0, gt=0)
    solar_multiplier: float = Field(default=1.0, ge=0)
    offline_chargers: list[Identifier] = Field(default_factory=list)

    @field_validator("offline_chargers")
    @classmethod
    def require_unique_chargers(cls, chargers: list[str]) -> list[str]:
        if len(chargers) != len(set(chargers)):
            raise ValueError("offline_chargers must contain unique identifiers")
        return chargers

    @model_validator(mode="after")
    def validate_charger_counts(self) -> "DigitalTwinRequest":
        if self.active_chargers > self.total_chargers:
            raise ValueError("active_chargers must not exceed total_chargers")
        if len(self.offline_chargers) > self.active_chargers:
            raise ValueError(
                "offline_chargers count must not exceed active_chargers"
            )
        return self


class DigitalTwinResponse(APIModel):
    scenario_solar_kw: float = Field(ge=0)
    scenario_demand_kw: float = Field(ge=0)
    supply_demand_balance_kw: float
    renewable_coverage_ratio: float | None = Field(default=None, ge=0)
    scenario_active_chargers: int = Field(ge=0)
    estimated_utilization: float = Field(ge=0, le=1)
    estimated_wait_impact_minutes: float = Field(ge=0)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommendation: TextValue
    simulation_version: Identifier
    is_simulation: Literal[True] = True


class SimulatorResetRequest(APIModel):
    charger_count: int = Field(default=6, ge=1, le=100)
    charger_power_kw: float = Field(default=22.0, gt=0, le=1000)
    current_solar_kw: float = Field(default=0.0, ge=0)
    current_station_demand_kw: float = Field(default=0.0, ge=0)


class SimulatorEVRequest(APIModel):
    ev_id: Identifier
    initial_soc_percent: float = Field(ge=0, lt=100)
    target_soc_percent: float = Field(gt=0, le=100)
    battery_capacity_kwh: float = Field(gt=0)
    max_charge_power_kw: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_soc_range(self) -> "SimulatorEVRequest":
        if self.target_soc_percent <= self.initial_soc_percent:
            raise ValueError("target_soc_percent must exceed initial_soc_percent")
        return self


class SimulatorAdvanceRequest(APIModel):
    minutes: int = Field(gt=0, le=1440)


class SimulatorSessionState(APIModel):
    ev_id: Identifier
    arrival_time: float = Field(ge=0)
    initial_soc_percent: float = Field(ge=0, lt=100)
    target_soc_percent: float = Field(gt=0, le=100)
    current_soc_percent: float = Field(ge=0, le=100)
    battery_capacity_kwh: float = Field(gt=0)
    max_charge_power_kw: float = Field(gt=0)
    assigned_charger_id: Identifier | None = None
    session_start_time: float | None = Field(default=None, ge=0)
    session_end_time: float | None = Field(default=None, ge=0)
    status: Literal["WAITING", "CHARGING", "COMPLETED", "CANCELLED"]
    energy_delivered_kwh: float = Field(ge=0)


class SimulatorChargerState(APIModel):
    charger_id: Identifier
    status: Literal["AVAILABLE", "RESERVED", "CHARGING", "FAULT", "OFFLINE"]
    max_power_kw: float = Field(gt=0)
    current_ev_id: Identifier | None = None
    current_power_kw: float = Field(ge=0)
    energy_delivered_kwh: float = Field(ge=0)
    session_elapsed_minutes: float = Field(ge=0)
    soc_percent: float | None = Field(default=None, ge=0, le=100)
    estimated_minutes_remaining: float | None = Field(default=None, ge=0)
    anomaly_active: bool


class SimulatorStationSummary(APIModel):
    total_chargers: int = Field(ge=1)
    available_chargers: int = Field(ge=0)
    charging_chargers: int = Field(ge=0)
    reserved_chargers: int = Field(ge=0)
    faulted_chargers: int = Field(ge=0)
    offline_chargers: int = Field(ge=0)
    queue_length: int = Field(ge=0)
    current_station_power_kw: float = Field(ge=0)
    current_solar_kw: float = Field(ge=0)
    current_station_demand_kw: float = Field(ge=0)


class SimulatorStateResponse(APIModel):
    simulation_time: float = Field(ge=0)
    chargers: list[SimulatorChargerState]
    queue: list[Identifier]
    sessions: list[SimulatorSessionState]
    completed_sessions: list[SimulatorSessionState]
    station_summary: SimulatorStationSummary
    simulation_version: Identifier
    is_simulation: Literal[True] = True


class HealthResponse(APIModel):
    status: Literal["ok"] = "ok"
    service: Literal["ChargeSentinel API"] = "ChargeSentinel API"
    api_version: Literal["v1"] = "v1"
