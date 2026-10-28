"""Real portable handoff and public API regression checks."""

import hashlib
import json
import shutil
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from team.fatima_anomaly.integration import inference as anomaly_handoff
from team.mazen_solar.integration import inference as solar_handoff
from team.mazen_solar.integration.source_features import NWP_WEATHER_FEATURES, SolarFeaturePipeline

client = TestClient(app)
ORIGIN = "2024-06-15T00:00:00+00:00"
FIXTURE = json.loads((anomaly_handoff.BUNDLE / "real_test_fixture.json").read_text())


@pytest.mark.parametrize("horizon,family", [(24, "Seasonal Persistence"), (48, "Ridge Regression")])
def test_real_solar_artifact_routing(horizon, family):
    response = client.post("/api/v1/solar/forecast", json={"forecast_origin": ORIGIN, "horizon_hours": horizon})
    assert response.status_code == 200
    body = response.json()
    assert body["model_family"] == family
    assert str(horizon) in body["model_version"]
    assert len(body["points"]) == horizon
    assert body["forecast_kw"] == body["points"][0]["forecast_kw"]
    assert all(0 <= point["forecast_kw"] <= 100 for point in body["points"])
    assert pd.Timestamp(body["points"][0]["timestamp"]) == pd.Timestamp("2024-06-15T01:00:00Z")


def test_solar_artifacts_are_distinct_and_verified():
    checks = json.loads((solar_handoff.BUNDLE / "checksums.json").read_text())
    assert checks["solar_24h.json"] != checks["solar_48h.json"]
    for name, digest in checks.items():
        assert hashlib.sha256((solar_handoff.BUNDLE / name).read_bytes()).hexdigest() == digest
    a, _, _ = solar_handoff.load_bundle(24)
    b, _, _ = solar_handoff.load_bundle(48)
    assert a["model_version"] != b["model_version"]


def test_solar_origin_and_missing_source_are_honest():
    for origin in ("2026-10-27T00:00:00+00:00", "2024-06-29T00:00:00+00:00"):
        assert client.post("/api/v1/solar/forecast", json={"forecast_origin": origin, "horizon_hours": 48}).status_code == 422
    assert client.post("/api/v1/solar/forecast", json={"forecast_origin": "2024-06-15T00:00:00+02:00", "horizon_hours": 24}).status_code == 422


def test_solar_48_hour_weather_run_is_causal():
    origin = pd.Timestamp("2024-06-15")
    targets = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
    weather = pd.DataFrame(index=targets)
    for field in NWP_WEATHER_FEATURES:
        weather[field] = 100.0 if field.endswith("day1") else 10.0
    hist = pd.Series(10.0, index=pd.date_range(origin - pd.Timedelta(hours=48), origin, freq="h"))
    hist.loc[origin] = 99.0  # PVGIS HH:09 reading is unavailable at HH:00 origin.
    frame = SolarFeaturePipeline().build_forecast_features(origin, 48, weather, hist)
    assert (frame.iloc[:23]["shortwave_radiation_previous_day1"] == 100.0).all()
    assert (frame.iloc[23:]["shortwave_radiation_previous_day1"] == 10.0).all()
    assert (frame["pv_origin_power"] == 10.0).all()
    assert frame.iloc[23]["pv_seasonal_persistence"] == 10.0
    assert frame.iloc[47]["pv_seasonal_persistence"] == 10.0


def test_anomaly_real_model_and_explanation():
    response = client.post("/api/v1/anomaly/predict", json=FIXTURE)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "anomaly-v1"
    assert body["threshold"] == 0.4
    assert 0 <= body["score"] <= 1
    assert body["reason"] and "unavailable" not in str(body["reason"]).lower()
    assert any(name in " ".join(body["reason"]) for name in ("location", "charger_id", "start_month"))


def test_anomaly_chronological_split_and_frozen_threshold():
    config = json.loads((anomaly_handoff.BUNDLE / "config.json").read_text())
    report = json.loads((anomaly_handoff.BUNDLE.parents[1] / "reports" / "integration_hardening" / "metrics.json").read_text())
    splits = config["splits"]
    assert splits["train"]["last"] < splits["validation"]["first"]
    assert splits["validation"]["last"] < splits["test"]["first"]
    assert sum(s["rows"] for s in splits.values()) == 441077
    assert config["threshold"] == report["validation_selected"]["threshold"]
    assert set(report["test"]) >= {"precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"}
    assert report["test"]["tn"] + report["test"]["fp"] + report["test"]["fn"] + report["test"]["tp"] == splits["test"]["rows"]


def test_solar_selection_report_preserves_missing_targets():
    report = json.loads((solar_handoff.BUNDLE.parents[1] / "reports" / "integration_hardening" / "selection.json").read_text())
    assert len(report["missing_target_hours"]) == 2
    assert report["horizons"]["24"]["champion"] == "Seasonal Persistence"
    assert report["horizons"]["48"]["champion"] == "Ridge Regression"
    for item in report["horizons"].values():
        assert item["rows"]["train"] > item["rows"]["validation"] > 0
        assert item["rows"]["test"] > 0
        assert all(item["origins"][k] > 0 for k in ("train", "validation", "test"))


@pytest.mark.parametrize("field", ["End Time", "session_duration_minutes", "energy_per_hour", "end_cause", "is_system_issue", "Transaction power/kwh"])
def test_anomaly_post_event_fields_rejected(field):
    payload = json.loads(json.dumps(FIXTURE))
    payload["features"][field] = 1
    assert client.post("/api/v1/anomaly/predict", json=payload).status_code == 422


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_corrupt_anomaly_artifact_returns_safe_503(tmp_path, monkeypatch, mode):
    from src.integration.adapters import anomaly
    bundle = tmp_path / "bundle"
    shutil.copytree(anomaly_handoff.BUNDLE, bundle)
    if mode == "missing":
        (bundle / "catboost.cbm").unlink()
    else:
        (bundle / "catboost.cbm").write_bytes(b"corrupt")
    monkeypatch.setattr(anomaly, "real_predict_anomaly", lambda p: anomaly_handoff.predict_anomaly(p, str(bundle)))
    response = client.post("/api/v1/anomaly/predict", json=FIXTURE)
    assert response.status_code == 503
    assert "checksum" not in response.text.lower()


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_corrupt_solar_artifact_returns_safe_503(tmp_path, monkeypatch, mode):
    from src.integration.adapters import solar
    bundle = tmp_path / "bundle"
    shutil.copytree(solar_handoff.BUNDLE, bundle)
    if mode == "missing":
        (bundle / "solar_48h.json").unlink()
    else:
        (bundle / "solar_48h.json").write_text("{}")
    monkeypatch.setattr(solar, "real_predict_solar", lambda p: solar_handoff.predict_solar(p, str(bundle)))
    response = client.post("/api/v1/solar/forecast", json={"forecast_origin": ORIGIN, "horizon_hours": 48})
    assert response.status_code == 503
    assert "checksum" not in response.text.lower()
