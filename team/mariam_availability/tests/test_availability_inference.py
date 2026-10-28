import json
import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator
from pathlib import Path

from src.availability import inference
from src.availability.features import NUMERIC, FeatureEncoder


class FixedDurationModel:
    def predict(self, frame):
        return np.full(len(frame), np.log1p(60.0))


def fitted_encoder():
    frame = pd.DataFrame({
        "charger_id": ["c1", "c2", "c3"],
        "location": ["site-a"] * 3,
        "district": ["district-a"] * 3,
    })
    for number, name in enumerate(NUMERIC, start=1):
        frame[name] = float(number)
    return FeatureEncoder().fit(frame)


def payload():
    return {
        "charger_id": "c1",
        "location": "site-a",
        "district": "district-a",
        "session_start": "2021-10-15T12:00:00",
        "as_of": "2021-10-15T12:30:00",
        "history_features": {
            "hist_abnormal_rate": 0.12,
            "sessions_before": 20,
            "has_history": 1,
        },
    }


@pytest.fixture
def predictor_bundle(monkeypatch):
    bundle = {
        "model": FixedDurationModel(),
        "encoder_state": fitted_encoder().get_state(),
        "duration_cap_min": 600.0,
        "target_mode": "log1p",
        "model_name": "xgboost_default",
        "data_version": {"data_sha256_12": "fixture"},
    }
    monkeypatch.setattr(inference.joblib, "load", lambda _: bundle)
    return inference.AvailabilityPredictor(inference.DEFAULT_BUNDLE_PATH)


def test_predict_returns_duration_end_remaining_wait_and_risk(predictor_bundle):
    repo = Path(__file__).resolve().parents[3]
    input_schema = json.loads((repo / "schemas" / "availability_input.schema.json").read_text())
    Draft202012Validator(input_schema).validate(payload())
    result = predictor_bundle.predict(payload())
    assert result["charger_id"] == "c1"
    assert result["predicted_duration_minutes"] == pytest.approx(60.0, abs=0.01)
    assert result["expected_free_time"] == "2021-10-15T13:00:00"
    assert result["expected_wait_minutes"] == pytest.approx(30.0)
    assert result["anomaly_risk"] == pytest.approx(0.12)
    assert result["model_version"] == "xgboost_default@fixture"
    schema = json.loads((repo / "schemas" / "availability_output.schema.json").read_text())
    Draft202012Validator(schema).validate(result)


@pytest.mark.parametrize("leakage", ["End Time", "duration_min", "is_abnormal", "Payment time"])
def test_inference_rejects_post_event_fields(predictor_bundle, leakage):
    request = payload()
    request[leakage] = "forbidden"
    with pytest.raises(ValueError, match="post-event"):
        predictor_bundle.predict(request)


def test_inference_requires_causal_history_risk_in_unit_range(predictor_bundle):
    request = payload()
    request["history_features"]["hist_abnormal_rate"] = 1.1
    with pytest.raises(ValueError, match="hist_abnormal_rate"):
        predictor_bundle.predict(request)


def test_inference_rejects_mixed_timezone_awareness(predictor_bundle):
    request = payload()
    request["as_of"] = "2021-10-15T12:30:00+08:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        predictor_bundle.predict(request)


def test_missing_bundle_error_is_actionable():
    missing = inference.TEAM_DIR / "artifacts" / "definitely-missing-bundle.joblib"
    with pytest.raises(FileNotFoundError, match="Run the availability training workflow"):
        inference.AvailabilityPredictor(missing)
