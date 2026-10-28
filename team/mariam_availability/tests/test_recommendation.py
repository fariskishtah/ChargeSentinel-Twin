import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.recommendation import recommend
from src.recommendation.ranker import recommend as rank_candidates


class FakePredictor:
    model_version = "fixture-duration-model"

    def predict(self, payload):
        return {
            "charger_id": str(payload["charger_id"]),
            "predicted_duration_minutes": 20.0,
            "expected_free_time": "2021-10-15T12:20:00",
            "expected_wait_minutes": 20.0,
            "anomaly_risk": payload["history_features"]["hist_abnormal_rate"],
            "model_version": self.model_version,
        }


def test_ranker_prefers_free_candidates_before_scoring_and_returns_alternatives():
    result = rank_candidates({
        "location": "site-a",
        "user_eta": "2021-10-15T12:00:00",
        "candidates": [
            {"charger_id": "c2", "location": "site-a", "expected_free_time": "2021-10-15T12:30:00",
             "anomaly_risk": 0.1, "demand_pressure": 0.2, "solar_benefit": 0.8},
            {"charger_id": "c1", "location": "site-a", "expected_free_time": "2021-10-15T12:00:00",
             "anomaly_risk": 0.3, "demand_pressure": 0.8, "solar_benefit": 0.1},
            {"charger_id": "c3", "location": "site-a", "expected_free_time": "2021-10-15T12:00:00",
             "anomaly_risk": 0.1, "demand_pressure": 0.1, "solar_benefit": 0.5},
        ],
    })
    assert result["recommended_charger"] == "c3"
    assert result["expected_wait_minutes"] == 0.0
    assert result["free_pool_restricted"] is True
    assert len(result["alternatives"]) == 1


def test_service_predicts_busy_candidates_filters_other_locations_and_uses_placeholders():
    result = recommend({
        "location": "site-a",
        "user_eta": "2021-10-15T12:00:00",
        "candidates": [
            {"charger_id": "busy", "busy": True, "session_start": "2021-10-15T12:00:00",
             "history_features": {"hist_abnormal_rate": 0.1}},
            {"charger_id": "free", "busy": False, "history_features": {"hist_abnormal_rate": 0.2}},
            {"charger_id": "free-2", "busy": False, "history_features": {"hist_abnormal_rate": 0.15}},
            {"charger_id": "other-site", "location": "site-b", "busy": False,
             "history_features": {"hist_abnormal_rate": 0.0}},
        ],
    }, predictor=FakePredictor())
    assert result["recommended_charger"] in {"free", "free-2"}
    assert len(result["alternatives"]) == 1
    assert result["mock_signals_used"] is True
    assert all(row["location"] == "site-a" for row in result["alternatives"])
    repo = Path(__file__).resolve().parents[3]
    schema = json.loads((repo / "schemas" / "recommendation_output.schema.json").read_text())
    Draft202012Validator(schema).validate(result)


def test_all_free_candidates_do_not_require_duration_bundle():
    result = recommend({
        "location": "site-a",
        "user_eta": "2021-10-15T12:00:00",
        "candidates": [{"charger_id": "free", "busy": False,
                        "history_features": {"hist_abnormal_rate": 0.1}}],
    }, bundle_path="missing-bundle.joblib")
    assert result["recommended_charger"] == "free"
    assert result["expected_wait_minutes"] == 0.0


def test_busy_candidates_need_session_start_and_history_anomaly_rate():
    body = {"location": "site-a", "user_eta": "2021-10-15T12:00:00",
            "candidates": [{"charger_id": "busy", "busy": True}]}
    with pytest.raises(ValueError, match="training-history anomaly_risk"):
        recommend(body, predictor=FakePredictor())


def test_same_location_and_timestamp_requirements_are_enforced():
    body = {"location": "site-a", "user_eta": "2021-10-15T12:00:00+08:00",
            "candidates": [{"charger_id": "busy", "busy": True,
                            "session_start": "2021-10-15T11:00:00",
                            "history_features": {"hist_abnormal_rate": 0.1}}]}
    with pytest.raises(ValueError, match="timezone"):
        recommend(body)


def test_json_schemas_are_valid_json_documents():
    repo = Path(__file__).resolve().parents[3]
    names = ["availability_input.schema.json", "availability_output.schema.json",
             "recommendation_input.schema.json", "recommendation_output.schema.json"]
    for name in names:
        schema = json.loads((repo / "schemas" / name).read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
    fixture_dir = Path(__file__).parent / "fixtures"
    availability_request = json.loads((fixture_dir / "availability_request.json").read_text())
    recommendation_request = json.loads((fixture_dir / "recommendation_request.json").read_text())
    availability_schema = json.loads((repo / "schemas" / "availability_input.schema.json").read_text())
    recommendation_schema = json.loads((repo / "schemas" / "recommendation_input.schema.json").read_text())
    Draft202012Validator(availability_schema).validate(availability_request)
    Draft202012Validator(recommendation_schema).validate(recommendation_request)
