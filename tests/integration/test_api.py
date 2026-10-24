"""Integration contract tests for the API skeleton."""

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unconnected_model_endpoints_are_explicitly_unavailable() -> None:
    requests = [
        ("GET", "/solar/forecast", None),
        ("GET", "/demand/forecast", None),
        ("POST", "/anomaly/predict", {}),
        ("GET", "/chargers/availability", None),
        ("POST", "/chargers/recommend", {}),
        ("POST", "/digital-twin/simulate", {}),
    ]

    for method, path, payload in requests:
        response = client.request(method, path, json=payload)
        assert response.status_code == 503
        assert response.json()["status"] == "model_not_ready"
