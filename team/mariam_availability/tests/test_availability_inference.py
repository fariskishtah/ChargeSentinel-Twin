import json
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from jsonschema import Draft202012Validator
from pathlib import Path

from src.availability import inference
from src.availability.features import NUMERIC, FeatureEncoder


class FixedDurationModel:
    def predict(self, frame):
        return np.full(len(frame), np.log1p(60.0))


class FakeContribBooster:
    def __init__(self, contributions):
        self.contributions = np.asarray(contributions, dtype=float)

    def predict(self, matrix, pred_contribs=False):
        assert pred_contribs is True
        return np.tile(self.contributions, (matrix.num_row(), 1))


class FakeXGBoostModel(xgb.XGBRegressor):
    def __init__(self, contributions):
        super().__init__()
        self.contributions = np.asarray(contributions, dtype=float)

    def get_booster(self):
        return FakeContribBooster(self.contributions)

    def predict(self, frame, **kwargs):
        margin = self.contributions[-1] + self.contributions[:-1].sum()
        return np.full(len(frame), margin)


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
    repo = Path(__file__).resolve().parents[1]
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


def test_explain_false_is_byte_identical_to_legacy_request(predictor_bundle):
    import json
    legacy = predictor_bundle.predict(payload())
    request = payload()
    request["explain"] = False
    disabled = predictor_bundle.predict(request)
    assert json.dumps(disabled, separators=(",", ":")) == json.dumps(legacy, separators=(",", ":"))


def test_explain_true_returns_additive_sorted_xgboost_contributions(monkeypatch):
    n_features = len(fitted_encoder().feature_names)
    values = np.linspace(-0.5, 0.5, n_features)
    contributions = np.append(values, 4.0)
    bundle = {
        "model": FakeXGBoostModel(contributions),
        "encoder_state": fitted_encoder().get_state(),
        "duration_cap_min": 600.0,
        "target_mode": "log1p",
        "model_name": "fake_xgboost",
        "data_version": {"data_sha256_12": "xai-test"},
    }
    monkeypatch.setattr(inference.joblib, "load", lambda _: bundle)
    predictor = inference.AvailabilityPredictor(inference.DEFAULT_BUNDLE_PATH)
    request = payload()
    request["explain"] = True
    result = predictor.predict(request)
    explanation = result["explanation"]
    encoded, _, _ = inference._prepare_features(request)
    encoded = predictor.encoder.transform(encoded)
    margin = float(predictor.model.predict(encoded)[0])
    assert explanation["base_value"] + values.sum() == pytest.approx(margin, abs=1e-9)
    assert len(explanation["top_contributors"]) == 5
    magnitudes = [abs(item["contribution"]) for item in explanation["top_contributors"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "availability_output.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text())).validate(result)


def test_explain_true_for_non_xgboost_returns_null_with_note(predictor_bundle):
    request = payload()
    request["explain"] = True
    result = predictor_bundle.predict(request)
    assert result["explanation"] is None
    assert "only for XGBoost" in result["explanation_note"]


def test_inference_works_from_outside_project_directory(predictor_bundle, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = inference.predict_availability(payload())
    assert result["expected_free_time"] == "2021-10-15T13:00:00"


def test_missing_charger_id_bad_timestamp_and_negative_elapsed_are_rejected(predictor_bundle):
    request = payload()
    del request["charger_id"]
    with pytest.raises(ValueError, match="charger_id"):
        predictor_bundle.predict(request)
    request = payload()
    request["session_start"] = "not-a-time"
    with pytest.raises((ValueError, TypeError), match="valid timestamp|ISO-8601"):
        predictor_bundle.predict(request)
    request = payload()
    request["as_of"] = "2021-10-15T11:59:00"
    with pytest.raises(ValueError, match="earlier than session_start"):
        predictor_bundle.predict(request)


def test_expected_availability_fixture_validates_against_output_schema():
    team_dir = Path(__file__).resolve().parents[1]
    schema = json.loads((team_dir / "schemas" / "availability_output.schema.json").read_text())
    output = json.loads((team_dir / "tests" / "fixtures" / "availability_response.json").read_text())
    Draft202012Validator(schema).validate(output)


def test_explain_availability_fixture_validates_against_output_schema():
    team_dir = Path(__file__).resolve().parents[1]
    schema = json.loads((team_dir / "schemas" / "availability_output.schema.json").read_text())
    output = json.loads((team_dir / "tests" / "fixtures" / "availability_explain_response.json").read_text())
    Draft202012Validator(schema).validate(output)
    assert output["explanation"]["unit"] == "log_minutes"


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
