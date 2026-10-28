"""Real demand asset, portable inference, serving contract, and safe errors."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.ui.api_client import ChargeSentinelAPIClient, ResultStatus
from src.integration import demand_history
from src.integration.adapters import demand
from src.integration.demand_history import (HISTORY_DIR, LocalJiaxingHistoryProvider,
    DemandHistoryError, DemandUnavailableError, digest)

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = "2021-10-01T00:00:00"
client = TestClient(app)


def test_asset_provenance_and_values():
    meta = json.loads((HISTORY_DIR / "metadata.json").read_text())
    path = HISTORY_DIR / "hourly_history.csv"
    assert digest(path) == meta["derived_sha256"]
    assert meta["source_rows"] == 441077 and meta["output_rows"] == 17547
    assert meta["aggregation_source_commit"] == "64eadd73175f1a063e134ecc7e346b628b8309cf"
    assert meta["data_version"] == "jiaxing-hourly-demand-v1"
    assert path.stat().st_size < 1024 * 1024
    d = pd.read_csv(path)
    assert list(d) == ["timestamp", "energy_kwh", "sessions"]
    t = pd.to_datetime(d.timestamp)
    assert t.is_unique and t.is_monotonic_increasing and t.dt.tz is None
    assert t.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    assert np.isfinite(d[["energy_kwh", "sessions"]]).all().all()
    assert (d[["energy_kwh", "sessions"]] >= 0).all().all()
    assert t.min().isoformat() == meta["first_hour"] and t.max().isoformat() == meta["last_hour"]


@pytest.mark.parametrize("origin", ["2020-01-08T21:00:00", ORIGIN, "2022-01-01T00:00:00"])
def test_exact_192_past_real_hours(origin):
    provider = LocalJiaxingHistoryProvider()
    rows = provider.get_history(origin)
    t = pd.to_datetime([r["timestamp"] for r in rows])
    assert len(rows) == 192 and (t < pd.Timestamp(origin)).all()
    assert t[-1] == pd.Timestamp(origin) - pd.Timedelta(hours=1)
    assert t[0] == pd.Timestamp(origin) - pd.Timedelta(hours=192)
    original = pd.read_csv(HISTORY_DIR / "hourly_history.csv", float_precision="round_trip")
    expected = original[original.timestamp.isin([r["timestamp"] for r in rows])].to_dict("records")
    assert rows == expected


@pytest.mark.parametrize("origin", ["2020-01-08T20:00:00", "2022-01-01T01:00:00", "2026-01-01T00:00:00"])
def test_unsupported_origins_no_fallback(origin):
    with pytest.raises(DemandHistoryError): LocalJiaxingHistoryProvider().get_history(origin)
    response = client.post("/api/v1/demand/forecast", json={"forecast_origin": origin})
    assert response.status_code == 422


@pytest.mark.parametrize("origin", ["bad", "2021-10-01T00:30:00", "2021-10-01T00:00:00Z", None])
def test_invalid_origin_422(origin):
    assert client.post("/api/v1/demand/forecast", json={"forecast_origin": origin}).status_code == 422


def test_real_api_contract_and_fractional_sessions():
    response = client.post("/api/v1/demand/forecast", json={"forecast_origin": ORIGIN})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "demand-completion-v1" and body["timestamp"] == ORIGIN
    assert body["expected_energy_kwh"] == pytest.approx(670.3817138671875)
    assert body["expected_sessions"] == pytest.approx(43.74503707885742)
    assert isinstance(body["expected_sessions"], float) and body["expected_sessions"] >= 0
    assert body["expected_energy_kwh"] >= 0
    schema = client.get("/openapi.json").json()["components"]["schemas"]["DemandForecastResponse"]
    assert schema["properties"]["expected_sessions"]["type"] == "number"


def test_client_parses_real_api_response():
    def handler(request):
        response = client.post(request.url.path, json=json.loads(request.content))
        return httpx.Response(response.status_code, json=response.json())
    ui = ChargeSentinelAPIClient(transport=httpx.MockTransport(handler))
    result = ui.get_demand_forecast({"forecast_origin": ORIGIN})
    assert result.status == ResultStatus.READY
    assert isinstance(result.data["expected_sessions"], float)
    assert result.data["model_version"] == "demand-completion-v1"


def test_no_history_override_or_missing_origin():
    for p in [{}, {"forecast_origin": ORIGIN, "history": []}]:
        assert client.post("/api/v1/demand/forecast", json=p).status_code == 422


def test_missing_history_sanitized_503(tmp_path, monkeypatch):
    monkeypatch.setattr(demand, "get_demand_service", lambda: demand.DemandService(
        LocalJiaxingHistoryProvider(tmp_path / "private-location")))
    response = client.post("/api/v1/demand/forecast", json={"forecast_origin": ORIGIN})
    assert response.status_code == 503
    assert "private-location" not in response.text and "Traceback" not in response.text


def test_corrupt_history_fails_without_filling(tmp_path):
    shutil.copytree(HISTORY_DIR, tmp_path / "history")
    path = tmp_path / "history/hourly_history.csv"
    path.write_text(path.read_text().replace("2020-01-01T00:00:00", "2020-01-01T01:00:00"))
    with pytest.raises(DemandUnavailableError): LocalJiaxingHistoryProvider(tmp_path / "history")


def test_corrupt_bundle_sanitized_503(monkeypatch):
    monkeypatch.setattr(demand, "digest", lambda path: "corrupt")
    with pytest.raises(DemandUnavailableError): demand.DemandService()
    monkeypatch.setattr(demand, "get_demand_service", lambda: demand.DemandService())
    response = client.post("/api/v1/demand/forecast", json={"forecast_origin": ORIGIN})
    assert response.status_code == 503 and "checksum" not in response.text.lower()


def test_inference_failure_sanitized(monkeypatch):
    from team.waseem_demand.demand import inference
    def fail(payload): raise RuntimeError("secret private path")
    monkeypatch.setattr(inference, "predict_demand", fail)
    response = client.post("/api/v1/demand/forecast", json={"forecast_origin": ORIGIN})
    assert response.status_code == 503 and "private" not in response.text


def test_artifacts_cached_between_calls():
    from team.waseem_demand.demand.inference import load_xgboost
    service = demand.get_demand_service()
    service.predict({"forecast_origin": ORIGIN})
    misses = load_xgboost.cache_info().misses
    service.predict({"forecast_origin": ORIGIN})
    assert load_xgboost.cache_info().misses == misses


def test_inference_outside_cwd(tmp_path):
    code = f"import sys;sys.path.insert(0,{str(ROOT)!r});from src.integration.adapters.demand import predict_demand;assert predict_demand({{'forecast_origin':{ORIGIN!r}}})['model_version']=='demand-completion-v1'"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
