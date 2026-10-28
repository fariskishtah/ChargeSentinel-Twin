"""API contract tests for the charging station simulator namespace."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app, raise_server_exceptions=False)

RESET_PAYLOAD = {
    "charger_count": 2,
    "charger_power_kw": 22,
    "current_solar_kw": 15,
    "current_station_demand_kw": 30,
}
EV_PAYLOAD = {
    "ev_id": "API-EV-1",
    "initial_soc_percent": 20,
    "target_soc_percent": 80,
    "battery_capacity_kwh": 60,
    "max_charge_power_kw": 11,
}


@pytest.fixture(autouse=True)
def reset_simulator() -> None:
    response = client.post("/api/v1/simulator/reset", json=RESET_PAYLOAD)
    assert response.status_code == 200


def test_state_endpoint_returns_200() -> None:
    response = client.get("/api/v1/simulator/state")

    assert response.status_code == 200
    assert response.json()["is_simulation"] is True
    assert response.json()["station_summary"]["total_chargers"] == 2


def test_reset_endpoint_applies_explicit_configuration() -> None:
    response = client.post(
        "/api/v1/simulator/reset",
        json={
            "charger_count": 4,
            "charger_power_kw": 50,
            "current_solar_kw": 20,
            "current_station_demand_kw": 45,
        },
    )

    assert response.status_code == 200
    state = response.json()
    assert state["station_summary"]["total_chargers"] == 4
    assert state["station_summary"]["current_solar_kw"] == 20


def test_add_ev_and_advance_endpoints_update_state() -> None:
    added = client.post("/api/v1/simulator/ev", json=EV_PAYLOAD)
    advanced = client.post("/api/v1/simulator/advance", json={"minutes": 15})

    assert added.status_code == 200
    assert added.json()["sessions"][0]["status"] == "CHARGING"
    assert advanced.status_code == 200
    assert advanced.json()["simulation_time"] == 15
    assert advanced.json()["sessions"][0]["current_soc_percent"] > 20


def test_invalid_charger_id_returns_404() -> None:
    response = client.post("/api/v1/simulator/chargers/UNKNOWN/offline")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_invalid_transition_returns_409() -> None:
    client.post("/api/v1/simulator/ev", json=EV_PAYLOAD)

    response = client.post("/api/v1/simulator/chargers/CHG-01/offline")

    assert response.status_code == 409
    assert "Cannot transition" in response.json()["detail"]


def test_invalid_soc_payload_returns_422() -> None:
    payload = {**EV_PAYLOAD, "target_soc_percent": 20}

    response = client.post("/api/v1/simulator/ev", json=payload)

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_payload"
