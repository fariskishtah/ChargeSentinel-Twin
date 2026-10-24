"""Deterministic Digital Twin calculations and API integration."""

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.main import app
from src.digital_twin.config import DIGITAL_TWIN_VERSION
from src.digital_twin.simulator import simulate

client = TestClient(app)

BASELINE: dict[str, object] = {
    "baseline_solar_kw": 60.0,
    "baseline_demand_kw": 100.0,
    "total_chargers": 10,
    "active_chargers": 10,
    "average_session_minutes": 60.0,
    "demand_multiplier": 1.0,
    "solar_multiplier": 1.0,
    "offline_chargers": [],
}


def scenario(**changes: object) -> dict[str, object]:
    payload = deepcopy(BASELINE)
    payload.update(changes)
    return payload


def test_baseline_scenario() -> None:
    result = simulate(BASELINE)

    assert result["scenario_solar_kw"] == 60.0
    assert result["scenario_demand_kw"] == 100.0
    assert result["supply_demand_balance_kw"] == -40.0
    assert result["renewable_coverage_ratio"] == 0.6
    assert result["scenario_active_chargers"] == 10
    assert result["estimated_utilization"] == 0.5
    assert result["estimated_wait_impact_minutes"] == 0.0
    assert result["severity"] == "LOW"


def test_demand_increase_by_thirty_percent() -> None:
    result = simulate(scenario(demand_multiplier=1.3))

    assert result["scenario_demand_kw"] == 130.0
    assert result["supply_demand_balance_kw"] == -70.0
    assert result["estimated_utilization"] == 0.65
    assert result["estimated_wait_impact_minutes"] == 18.0
    assert result["severity"] == "MEDIUM"


def test_solar_reduction_by_fifty_percent() -> None:
    result = simulate(scenario(solar_multiplier=0.5))

    assert result["scenario_solar_kw"] == 30.0
    assert result["renewable_coverage_ratio"] == 0.3
    assert result["severity"] == "MEDIUM"


def test_one_charger_offline() -> None:
    result = simulate(scenario(offline_chargers=["charger-1"]))

    assert result["scenario_active_chargers"] == 9
    assert result["estimated_utilization"] == 0.5556
    assert result["estimated_wait_impact_minutes"] == 6.0
    assert result["severity"] == "MEDIUM"


def test_multiple_chargers_offline() -> None:
    result = simulate(
        scenario(
            offline_chargers=[
                "charger-1",
                "charger-2",
                "charger-3",
                "charger-4",
                "charger-5",
            ]
        )
    )

    assert result["scenario_active_chargers"] == 5
    assert result["estimated_utilization"] == 1.0
    assert result["estimated_wait_impact_minutes"] == 30.0
    assert result["severity"] == "CRITICAL"


def test_zero_demand_is_safe_and_coverage_is_undefined() -> None:
    result = simulate(scenario(baseline_demand_kw=0.0))

    assert result["scenario_demand_kw"] == 0.0
    assert result["supply_demand_balance_kw"] == 60.0
    assert result["renewable_coverage_ratio"] is None
    assert result["estimated_utilization"] == 0.0
    assert result["estimated_wait_impact_minutes"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("baseline_solar_kw", -1.0), ("baseline_demand_kw", -1.0)],
)
def test_negative_baseline_energy_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        simulate(scenario(**{field: value}))


def test_offline_count_above_active_chargers_is_rejected() -> None:
    with pytest.raises(ValidationError):
        simulate(
            scenario(
                active_chargers=1,
                offline_chargers=["charger-1", "charger-2"],
            )
        )


def test_active_count_above_total_chargers_is_rejected() -> None:
    with pytest.raises(ValidationError):
        simulate(scenario(total_chargers=5, active_chargers=6))


@pytest.mark.parametrize(
    "payload",
    [
        scenario(demand_multiplier=0.1, solar_multiplier=5.0),
        scenario(demand_multiplier=10.0, offline_chargers=["charger-1"]),
        scenario(active_chargers=0, baseline_demand_kw=100.0),
    ],
)
def test_utilization_is_bounded_and_wait_impact_is_nonnegative(
    payload: dict[str, object],
) -> None:
    result = simulate(payload)

    assert 0 <= result["estimated_utilization"] <= 1
    assert result["estimated_wait_impact_minutes"] >= 0


def test_same_input_produces_same_output() -> None:
    assert simulate(BASELINE) == simulate(BASELINE)


def test_simulation_label_and_version_are_always_present() -> None:
    result = simulate(BASELINE)

    assert result["is_simulation"] is True
    assert result["simulation_version"] == DIGITAL_TWIN_VERSION


def test_api_returns_simulation_for_valid_input() -> None:
    response = client.post("/api/v1/digital-twin/simulate", json=BASELINE)

    assert response.status_code == 200
    assert response.json() == simulate(BASELINE)


def test_api_rejects_invalid_simulation_input() -> None:
    response = client.post(
        "/api/v1/digital-twin/simulate",
        json=scenario(active_chargers=11),
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_payload"
