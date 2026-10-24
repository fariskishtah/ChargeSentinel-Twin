"""Contract tests for the Streamlit-to-FastAPI HTTP client."""

import httpx
import pytest

from app.ui.api_client import (
    APIResult,
    ChargeSentinelAPIClient,
    ResultStatus,
)


def _client(handler: httpx.MockTransport) -> ChargeSentinelAPIClient:
    return ChargeSentinelAPIClient(
        base_url="http://api.test",
        transport=handler,
    )


def test_health_parses_successful_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "service": "ChargeSentinel API",
                "api_version": "v1",
            },
        )

    result = _client(httpx.MockTransport(handler)).health()

    assert result.status is ResultStatus.READY
    assert result.data == {
        "status": "ok",
        "service": "ChargeSentinel API",
        "api_version": "v1",
    }


def test_client_handles_model_not_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/solar/forecast"
        return httpx.Response(
            503,
            json={
                "error": "model_not_ready",
                "message": "Model or component is not ready.",
                "component": "solar",
            },
        )

    result = _client(httpx.MockTransport(handler)).get_solar_forecast(
        {
            "forecast_origin": "2026-10-24T12:00:00+02:00",
            "horizon_hours": 24,
        }
    )

    assert result == APIResult(
        status=ResultStatus.MODEL_NOT_READY,
        component="solar",
    )
    assert result.data is None


def test_client_methods_target_the_versioned_api_routes() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(503, json={"component": "pending"})

    client = _client(httpx.MockTransport(handler))
    origin = "2026-10-24T12:00:00+02:00"
    client.get_solar_forecast({"forecast_origin": origin, "horizon_hours": 24})
    client.get_demand_forecast({"forecast_origin": origin})
    client.predict_anomaly(
        {"charger_id": "charger-1", "features": {"feature_1": 1.0}}
    )
    client.get_availability({"charger_id": "charger-1", "query_time": origin})
    client.get_recommendation({"user_eta": origin})
    client.simulate_digital_twin(
        {
            "demand_multiplier": 1,
            "solar_multiplier": 1,
            "offline_chargers": [],
        }
    )

    assert paths == [
        "/api/v1/solar/forecast",
        "/api/v1/demand/forecast",
        "/api/v1/anomaly/predict",
        "/api/v1/chargers/availability",
        "/api/v1/chargers/recommend",
        "/api/v1/digital-twin/simulate",
    ]


def test_client_handles_validation_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"error": "invalid_payload", "issues": []},
        )

    result = _client(httpx.MockTransport(handler)).predict_anomaly(
        {"charger_id": "charger-1", "features": {}}
    )

    assert result.status is ResultStatus.INVALID_INPUT
    assert result.data is None


def test_client_handles_api_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = _client(httpx.MockTransport(handler)).health()

    assert result.status is ResultStatus.API_OFFLINE
    assert result.data is None


def test_client_handles_invalid_base_url_without_exposing_exception() -> None:
    result = ChargeSentinelAPIClient(base_url="://invalid").health()

    assert result.status is ResultStatus.API_OFFLINE
    assert result.data is None


@pytest.mark.parametrize("status_code", [500, 502, 404])
def test_client_sanitizes_other_backend_failures(status_code: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"detail": "internal value that must not reach the UI"},
        )

    result = _client(httpx.MockTransport(handler)).health()

    assert result == APIResult(status=ResultStatus.BACKEND_ERROR)
    assert result.data is None


def test_invalid_success_payload_is_not_presented_as_ready() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"forecast_kw": 999999})

    result = _client(httpx.MockTransport(handler)).health()

    assert result.status is ResultStatus.BACKEND_ERROR
    assert result.data is None


def test_base_url_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHARGESENTINEL_API_URL", "http://api.internal:9000/")

    client = ChargeSentinelAPIClient()

    assert client.base_url == "http://api.internal:9000"
