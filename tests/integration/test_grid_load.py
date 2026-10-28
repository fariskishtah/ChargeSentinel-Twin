"""Causal historical interval detector and public contract checks."""

from datetime import datetime

from fastapi.testclient import TestClient

from app.api.main import app
from src.grid_load.detector import detect, interval_features, load_history

client = TestClient(app)


def test_historical_grid_load_route_uses_real_recorded_interval():
    response = client.post("/api/v1/anomaly/grid-load", json={"timestamp": "2021-10-01T00:00:00"})
    assert response.status_code == 200
    body = response.json()
    assert body["detector_version"] == "grid-load-seasonal-rule-v1"
    assert body["timestamp"] == "2021-10-01T00:00:00"
    assert body["is_simulated"] is False
    assert body["energy_kwh"] >= 0 and body["sessions"] >= 0
    assert body["reason"]


def test_future_changes_cannot_change_interval_features():
    _, rows = load_history()
    index = next(i for i, row in enumerate(rows) if row["timestamp"] == datetime(2021, 10, 1))
    before = interval_features(rows, index)
    changed = [dict(row) for row in rows]
    changed[index + 1]["energy_kwh"] = 10**9
    changed[index + 168]["sessions"] = 10**9
    assert interval_features(changed, index) == before


def test_invalid_or_future_interval_is_rejected():
    for value in ("2022-01-01T00:00:00", "2021-10-01T00:30:00", "2021-10-01T00:00:00Z"):
        assert client.post("/api/v1/anomaly/grid-load", json={"timestamp": value}).status_code == 422


def test_corrupt_detector_config_returns_safe_503(tmp_path, monkeypatch):
    from src.grid_load import detector
    config = tmp_path / "grid-load-v1.json"
    config.write_text('{"history_sha256":"incorrect","threshold":1}')
    monkeypatch.setattr(detector, "CONFIG", config)
    detector.load_history.cache_clear()
    try:
        response = client.post("/api/v1/anomaly/grid-load", json={"timestamp": "2021-10-01T00:00:00"})
        assert response.status_code == 503
        assert "checksum" not in response.text.lower()
    finally:
        detector.load_history.cache_clear()


def test_integrated_session_risk_contract_still_works():
    from team.fatima_anomaly.integration.inference import BUNDLE
    import json
    fixture = json.loads((BUNDLE / "real_test_fixture.json").read_text())
    response = client.post("/api/v1/anomaly/predict", json=fixture)
    assert response.status_code == 200
    assert response.json()["model_version"] == "anomaly-v1"
