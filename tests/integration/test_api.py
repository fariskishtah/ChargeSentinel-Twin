"""Integration contract tests for the API skeleton."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)
FORECAST_ORIGIN = "2026-10-24T12:00:00+02:00"
FIXTURES = Path(__file__).resolve().parents[2] / "team/mariam_availability/tests/fixtures"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ChargeSentinel API",
        "api_version": "v1",
    }


def test_real_jiaxing_availability_and_recommendation() -> None:
    availability = client.post(
        "/api/v1/chargers/availability",
        json=_fixture("real_jiaxing_availability_request.json"),
    )
    assert availability.status_code == 200
    result = availability.json()
    assert result["charger_id"] == "2"
    assert result["predicted_duration_minutes"] == pytest.approx(31.74, abs=0.02)
    assert result["expected_wait_minutes"] == pytest.approx(16.74, abs=0.02)
    assert result["expected_free_time"].startswith("2021-10-15T12:44:10")
    assert result["bundle_version"].startswith("availability-duration-xgboost-default-v1@87ec74f0fbee")

    recommendation = client.post(
        "/api/v1/chargers/recommend",
        json=_fixture("real_jiaxing_recommendation_request.json"),
    )
    assert recommendation.status_code == 200
    ranked = recommendation.json()
    assert ranked["recommended_charger"] == "8"
    assert ranked["mock_signals_used"] is True
    assert ranked["expected_wait_minutes"] >= 0


def test_availability_rejects_future_outcome_and_live_date() -> None:
    payload = _fixture("real_jiaxing_availability_request.json")
    payload["end_time"] = "2021-10-15T12:33:47"
    response = client.post("/api/v1/chargers/availability", json=payload)
    assert response.status_code == 422
    payload.pop("end_time")
    payload["as_of"] = "2026-10-24T12:00:00"
    assert client.post("/api/v1/chargers/availability", json=payload).status_code == 422


def test_recommendation_excludes_fault_and_offline() -> None:
    payload = _fixture("real_jiaxing_recommendation_request.json")
    for candidate in payload["candidates"]:
        if candidate["charger_id"] == "8":
            candidate["status"] = "FAULT"
    response = client.post("/api/v1/chargers/recommend", json=payload)
    assert response.status_code == 200
    assert response.json()["recommended_charger"] != "8"
    for candidate in payload["candidates"]:
        candidate["status"] = "OFFLINE"
    assert client.post("/api/v1/chargers/recommend", json=payload).status_code == 422


def test_missing_availability_artifact_returns_sanitized_503(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.integration.adapters import availability

    def missing_bundle() -> object:
        raise availability.ModelNotReadyError("availability")

    monkeypatch.setattr(availability, "_predictor", missing_bundle)
    response = client.post(
        "/api/v1/chargers/availability",
        json=_fixture("real_jiaxing_availability_request.json"),
    )
    assert response.status_code == 503
    assert response.json() == {
        "error": "model_not_ready",
        "message": "Model or component is not ready.",
        "component": "availability",
    }


def test_solar_horizon_validation() -> None:
    valid_response = client.post(
        "/api/v1/solar/forecast",
        json={"forecast_origin": "2024-06-15T00:00:00+00:00", "horizon_hours": 48},
    )
    invalid_response = client.post(
        "/api/v1/solar/forecast",
        json={"forecast_origin": FORECAST_ORIGIN, "horizon_hours": 12},
    )

    assert valid_response.status_code == 200
    assert invalid_response.status_code == 422
    assert invalid_response.json()["error"] == "invalid_payload"


def test_anomaly_outcome_field_is_rejected() -> None:
    response = client.post(
        "/api/v1/anomaly/predict",
        json={"charger_id": "charger-1", "features": {"is_abnormal": True}},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_payload"


@pytest.mark.parametrize(
    "payload",
    [
        {"demand_multiplier": 0, "solar_multiplier": 1},
        {"demand_multiplier": 1, "solar_multiplier": -0.1},
    ],
)
def test_digital_twin_multiplier_validation(payload: dict[str, float]) -> None:
    response = client.post(
        "/api/v1/digital-twin/simulate",
        json={
            "baseline_solar_kw": 60,
            "baseline_demand_kw": 100,
            "total_chargers": 10,
            "active_chargers": 10,
            "average_session_minutes": 60,
            "offline_chargers": [],
            **payload,
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_payload"


def test_openapi_contract_and_docs() -> None:
    openapi_response = client.get("/openapi.json")
    docs_response = client.get("/docs")

    assert openapi_response.status_code == 200
    assert docs_response.status_code == 200
    assert set(openapi_response.json()["paths"]) == {
        "/health",
        "/api/v1/solar/forecast",
        "/api/v1/demand/forecast",
        "/api/v1/anomaly/predict",
        "/api/v1/anomaly/grid-load",
        "/api/v1/chargers/availability",
        "/api/v1/chargers/recommend",
        "/api/v1/digital-twin/simulate",
        "/api/v1/simulator/reset",
        "/api/v1/simulator/state",
        "/api/v1/simulator/ev",
        "/api/v1/simulator/advance",
        "/api/v1/simulator/chargers/{charger_id}/fault",
        "/api/v1/simulator/chargers/{charger_id}/clear-fault",
        "/api/v1/simulator/chargers/{charger_id}/offline",
        "/api/v1/simulator/chargers/{charger_id}/restore",
    }
    assert {
        "SolarForecastRequest",
        "DemandForecastRequest",
        "AnomalyRequest",
        "AvailabilityRequest",
        "RecommendationRequest",
        "DigitalTwinRequest",
        "HealthResponse",
    }.issubset(openapi_response.json()["components"]["schemas"])


def test_unexpected_errors_are_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api.routers import solar

    def fail_without_leaking(_payload: object) -> dict[str, object]:
        raise RuntimeError("private internal detail")

    monkeypatch.setattr(solar, "predict_solar", fail_without_leaking)
    safe_client = TestClient(app, raise_server_exceptions=False)
    response = safe_client.post(
        "/api/v1/solar/forecast",
        json={"forecast_origin": "2024-06-15T00:00:00+00:00", "horizon_hours": 24},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_server_error",
        "message": "Unexpected internal error.",
    }
    assert "private internal detail" not in response.text
