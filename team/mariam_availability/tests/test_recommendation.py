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


def test_real_recommendation_fixture_explains_every_ranked_candidate():
    fixture_dir = Path(__file__).parent / "fixtures"
    request = json.loads((fixture_dir / "recommendation_request.json").read_text(encoding="utf-8"))
    result = recommend(request, predictor=FakePredictor())
    for candidate in [result, *result["alternatives"]]:
        explanation = candidate["explanation"]
        assert sum(explanation["weighted_terms"].values()) == candidate["score"]
        expected_dominant = max(
            explanation["weighted_terms"],
            key=lambda term: abs(explanation["weighted_terms"][term]),
        )
        assert explanation["dominant_factor"] == expected_dominant
    assert result["comparison"]["second_charger_id"] == result["alternatives"][0]["charger_id"]


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
    repo = Path(__file__).resolve().parents[1]
    schema = json.loads((repo / "schemas" / "recommendation_output.schema.json").read_text())
    Draft202012Validator(schema).validate(result)
    candidates = [result, *result["alternatives"]]
    for candidate in candidates:
        explanation = candidate["explanation"]
        assert sum(explanation["weighted_terms"].values()) == candidate["score"]
        dominant = max(explanation["weighted_terms"], key=lambda term: abs(explanation["weighted_terms"][term]))
        assert explanation["dominant_factor"] == dominant
    assert "comparison" in result


def test_all_free_candidates_do_not_require_duration_bundle():
    result = recommend({
        "location": "site-a",
        "user_eta": "2021-10-15T12:00:00",
        "candidates": [{"charger_id": "free", "busy": False,
                        "history_features": {"hist_abnormal_rate": 0.1}}],
    }, bundle_path="missing-bundle.joblib")
    assert result["recommended_charger"] == "free"
    assert result["expected_wait_minutes"] == 0.0


def test_busy_candidate_without_person3_uses_historical_table():
    body = {"location": "site-a", "user_eta": "2021-10-15T12:00:00",
            "candidates": [{"charger_id": "busy", "busy": False}]}
    result = recommend(body)
    assert result["risk_source"] == "historical_abnormal_rate"


def test_recommend_succeeds_without_any_risk_or_person3_payload():
    result = recommend({"location": "site-a", "user_eta": "2021-10-15T12:00:00",
                        "candidates": [{"charger_id": "never-seen", "busy": False}]})
    assert result["recommended_charger"] == "never-seen"
    assert result["risk_source"] == "historical_abnormal_rate"
    assert result["score_components"]["anomaly_risk"] >= 0


def test_unseen_charger_uses_global_training_rate():
    result = recommend({"location": "site-a", "user_eta": "2021-10-15T12:00:00",
                        "candidates": [{"charger_id": "unseen-id", "busy": False}]})
    from src.recommendation.risk import load_risk_table
    global_rate = load_risk_table()["global_rate"]
    assert result["score_components"]["anomaly_risk"] == pytest.approx(global_rate)


def test_missing_risk_table_is_built_from_training_period_only():
    import joblib
    import pandas as pd
    from src.recommendation.risk import build_risk_table
    features = pd.DataFrame({
        "charger_id": ["a", "a", "b"],
        "start": pd.to_datetime(["2021-06-30", "2021-06-29", "2021-07-01"]),
        "is_abnormal": [0, 1, 1],
    })
    temp_dir = Path(__file__).resolve().parents[1] / "artifacts"
    feature_path = temp_dir / "risk_table_test_features.pkl"
    table_path = temp_dir / "risk_table_test_output.json"
    try:
        joblib.dump(features, feature_path)
        table = build_risk_table(feature_path, table_path)
        assert table["built_from_end"] == "2021-07-01"
        assert table["global_rate"] == pytest.approx(0.5)
        assert table["per_charger"] == {"a": {"rate": 0.5, "n_sessions": 2}}
        assert table_path.is_file()
    finally:
        feature_path.unlink(missing_ok=True)
        table_path.unlink(missing_ok=True)


def test_disabled_person3_score_does_not_change_ranking():
    base = {"location": "site-a", "user_eta": "2021-10-15T12:00:00", "candidates": [
        {"charger_id": "201001", "busy": False},
        {"charger_id": "unseen-x", "busy": False},
    ]}
    with_score = json.loads(json.dumps(base))
    with_score["candidates"][0]["anomaly_prediction"] = {"score": 0.999}
    assert recommend(with_score) == recommend(base)


def test_enabled_person3_score_is_used_when_explicitly_configured(monkeypatch):
    from src.recommendation import service
    config = service.load_config()
    config["use_person3_score"] = True
    monkeypatch.setattr(service, "load_config", lambda: config)
    result = service.recommend({"location": "site-a", "user_eta": "2021-10-15T12:00:00",
        "candidates": [{"charger_id": "201001", "busy": False,
                        "anomaly_prediction": {"score": 0.93}}]})
    assert result["risk_source"] == "person3_score"
    assert result["score_components"]["anomaly_risk"] == pytest.approx(0.93)


def test_empty_candidate_list_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        recommend({"location": "site-a", "user_eta": "2021-10-15T12:00:00", "candidates": []})


def test_missing_charger_id_and_bad_timestamp_are_rejected():
    with pytest.raises(ValueError, match="charger_id is required"):
        recommend({"location": "site-a", "user_eta": "2021-10-15T12:00:00",
                   "candidates": [{"busy": False}]})
    with pytest.raises(ValueError, match="ISO-8601 timestamp"):
        recommend({"location": "site-a", "user_eta": "bad-time",
                   "candidates": [{"charger_id": "c1", "busy": False}]})


def test_same_location_and_timestamp_requirements_are_enforced():
    body = {"location": "site-a", "user_eta": "2021-10-15T12:00:00+08:00",
            "candidates": [{"charger_id": "busy", "busy": True,
                            "session_start": "2021-10-15T11:00:00",
                            "history_features": {"hist_abnormal_rate": 0.1}}]}
    with pytest.raises(ValueError, match="timezone"):
        recommend(body)


def test_json_schemas_are_valid_json_documents():
    repo = Path(__file__).resolve().parents[1]
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
    recommendation_request["candidates"][0]["anomaly_prediction"] = {"score": 0.99}
    Draft202012Validator(recommendation_schema).validate(recommendation_request)
    no_risk_request = {"location": "site-a", "user_eta": "2021-10-15T12:00:00",
                       "candidates": [{"charger_id": "unseen", "busy": False}]}
    Draft202012Validator(recommendation_schema).validate(no_risk_request)


def test_expected_recommendation_fixture_validates_against_output_schema():
    team_dir = Path(__file__).resolve().parents[1]
    schema = json.loads((team_dir / "schemas" / "recommendation_output.schema.json").read_text())
    output = json.loads((team_dir / "tests" / "fixtures" / "recommendation_response.json").read_text())
    Draft202012Validator(schema).validate(output)
    assert sum(output["explanation"]["weighted_terms"].values()) == output["score"]
