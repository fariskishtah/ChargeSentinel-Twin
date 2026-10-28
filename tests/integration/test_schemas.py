"""Validation tests for the public v1 Pydantic contracts."""

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    AnomalyRequest,
    AnomalyResponse,
    AvailabilityRequest,
    AvailabilityResponse,
    DemandForecastRequest,
    DemandForecastResponse,
    DigitalTwinRequest,
    HealthResponse,
    RecommendationRequest,
    SolarForecastRequest,
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
FIXTURES = Path(__file__).resolve().parents[2] / "team/mariam_availability/tests/fixtures"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_valid_requests() -> None:
    assert SolarForecastRequest(forecast_origin=NOW, horizon_hours=24)
    assert SolarForecastRequest(forecast_origin=NOW, horizon_hours=48)
    assert DemandForecastRequest(forecast_origin=datetime(2021, 10, 1))
    assert AnomalyRequest(charger_id="charger-1", features={
        "location": "Financial Industrial Park", "district": "Nanhu",
        "start_time": "2021-10-16T12:02:44"})
    assert AvailabilityRequest.model_validate(_fixture("real_jiaxing_availability_request.json"))
    assert RecommendationRequest.model_validate(_fixture("real_jiaxing_recommendation_request.json"))
    assert DigitalTwinRequest(
        baseline_solar_kw=60,
        baseline_demand_kw=100,
        total_chargers=10,
        active_chargers=10,
        average_session_minutes=60,
        demand_multiplier=1.2,
        solar_multiplier=0.7,
        offline_chargers=["charger-1"],
    )
    assert HealthResponse().api_version == "v1"


def test_invalid_solar_horizon_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SolarForecastRequest(forecast_origin=NOW, horizon_hours=12)


def test_negative_demand_output_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DemandForecastResponse(
            timestamp=NOW.replace(tzinfo=None),
            expected_energy_kwh=-0.1,
            expected_sessions=1,
            model_version="demand-v1",
        )

    with pytest.raises(ValidationError):
        DemandForecastResponse(
            timestamp=NOW.replace(tzinfo=None),
            expected_energy_kwh=1,
            expected_sessions=-1,
            model_version="demand-v1",
        )


def test_anomaly_score_above_one_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AnomalyResponse(
            charger_id="charger-1",
            anomaly=True,
            score=1.01,
            reason=["threshold exceeded"],
            model_version="anomaly-v1",
        )


def test_negative_wait_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AvailabilityResponse(
            charger_id="charger-1",
            expected_free_time=NOW,
            expected_wait_minutes=-1,
            model_version="availability-v1",
        )


@pytest.mark.parametrize(
    ("demand_multiplier", "solar_multiplier"),
    [(0, 1), (-1, 1), (1, -0.1)],
)
def test_invalid_digital_twin_multipliers_are_rejected(
    demand_multiplier: float, solar_multiplier: float
) -> None:
    with pytest.raises(ValidationError):
        DigitalTwinRequest(
            baseline_solar_kw=60,
            baseline_demand_kw=100,
            total_chargers=10,
            active_chargers=10,
            average_session_minutes=60,
            demand_multiplier=demand_multiplier,
            solar_multiplier=solar_multiplier,
        )


@pytest.mark.parametrize("field", ["end_cause", "label", "is_abnormal"])
def test_anomaly_outcome_fields_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        AnomalyRequest(charger_id="charger-1", features={field: 1})


def test_end_time_is_not_an_availability_input() -> None:
    payload = _fixture("real_jiaxing_availability_request.json")
    payload["end_time"] = "2021-10-15T12:33:47"
    with pytest.raises(ValidationError):
        AvailabilityRequest.model_validate(payload)
