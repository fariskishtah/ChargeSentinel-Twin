"""Version 1 API contracts shared by the integration layer and model owners."""

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    AwareDatetime,
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
    # TODO(Mazen): confirm any client-supplied weather fields after feature review.


class SolarForecastResponse(APIModel):
    timestamp: AwareDatetime
    horizon_hours: Literal[24, 48]
    forecast_kw: float = Field(ge=0)
    model_version: Identifier


class DemandForecastRequest(APIModel):
    forecast_origin: AwareDatetime
    # TODO(Waseem): confirm whether the client needs any additional inputs.


class DemandForecastResponse(APIModel):
    timestamp: AwareDatetime
    expected_energy_kwh: float = Field(ge=0)
    expected_sessions: int = Field(ge=0)
    model_version: Identifier


class AnomalyRequest(APIModel):
    charger_id: Identifier
    features: dict[str, FeatureValue] = Field(min_length=1)
    # TODO(Fatima): replace this extensible map with the leakage-audited safe schema.

    @field_validator("features")
    @classmethod
    def reject_outcome_fields(
        cls, features: dict[str, FeatureValue]
    ) -> dict[str, FeatureValue]:
        forbidden = {"end_cause", "label", "is_abnormal"}
        supplied = {key.strip().lower().replace(" ", "_") for key in features}
        if "" in supplied:
            raise ValueError("Feature names must not be blank")
        leaked = sorted(forbidden & supplied)
        if leaked:
            raise ValueError(f"Outcome fields are not allowed: {', '.join(leaked)}")
        return features


class AnomalyResponse(APIModel):
    charger_id: Identifier
    anomaly: bool
    score: float = Field(ge=0, le=1)
    reason: list[TextValue]
    model_version: Identifier


class AvailabilityRequest(APIModel):
    charger_id: Identifier
    query_time: AwareDatetime
    # End Time is deliberately excluded because it is target-derived information.


class AvailabilityResponse(APIModel):
    charger_id: Identifier
    expected_free_time: AwareDatetime
    expected_wait_minutes: float = Field(ge=0)
    model_version: Identifier


class RecommendationRequest(APIModel):
    user_eta: AwareDatetime
    preferred_charger_id: Identifier | None = None
    # TODO(Mariam): confirm which versioned upstream outputs the ranker accepts.


class RecommendationResponse(APIModel):
    recommended_charger: Identifier
    recommended_slot: TextValue
    expected_wait_minutes: float = Field(ge=0)
    alternatives: list[Identifier]
    model_version: Identifier


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
