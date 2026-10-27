
"""Tests for the anomaly inference interface."""

import pytest

from team.fatima_anomaly.src import inference


def test_invalid_empty_payload_is_rejected():
    with pytest.raises(ValueError):
        inference.predict_anomaly([])


def test_invalid_payload_record_is_rejected():
    with pytest.raises(TypeError):
        inference.predict_anomaly([123])


def test_missing_model_raises_clear_error(tmp_path):
    missing_model = tmp_path / "missing_model.cbm"
    missing_config = tmp_path / "missing_config.json"

    with pytest.raises(FileNotFoundError):
        inference.predict_anomaly(
            {"charger_id": "C01", "temperature": 25},
            model_path=missing_model,
            config_path=missing_config,
        )


def test_output_contract_with_mocked_model(monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return __import__("numpy").array([[0.1, 0.9] for _ in range(len(X))])

        def get_feature_importance(self, data, type):
            return [[0.5, 0.2, 0.1]]

    fake_config = {
        "decision_threshold": 0.41,
        "features": ["temperature", "energy", "start_hour"],
        "categorical_features": [],
        "model_version": "test-version-1",
    }

    monkeypatch.setattr(
        inference,
        "_get_model",
        lambda *args, **kwargs: (FakeModel(), fake_config),
    )

    monkeypatch.setattr(
        inference,
        "prepare_features",
        lambda records, **kwargs: __import__("pandas").DataFrame(
            [[25.0, 10.0, 8.0]],
            columns=["temperature", "energy", "start_hour"],
        ),
    )

    result = inference.predict_anomaly(
        {
            "charger_id": "C01",
            "temperature": 25,
            "energy": 10,
            "start_hour": 8,
        }
    )

    assert result["charger_id"] == "C01"
    assert result["anomaly"] is True
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["reason"], list)
    assert result["model_version"] == "test-version-1"


def test_batch_payload_returns_list(monkeypatch):
    class FakeModel:
        def predict_proba(self, X):
            return __import__("numpy").array([[0.1, 0.9] for _ in range(len(X))])

        def get_feature_importance(self, data, type):
            return [[0.5, 0.2, 0.1]]

    fake_config = {
        "decision_threshold": 0.41,
        "features": ["temperature", "energy", "start_hour"],
        "categorical_features": [],
        "model_version": "test-version-1",
    }

    monkeypatch.setattr(
        inference,
        "_get_model",
        lambda *args, **kwargs: (FakeModel(), fake_config),
    )

    monkeypatch.setattr(
        inference,
        "prepare_features",
        lambda records, **kwargs: __import__("pandas").DataFrame(
            [[25.0, 10.0, 8.0]] * len(records),
            columns=["temperature", "energy", "start_hour"],
        ),
    )

    result = inference.predict_anomaly(
        [
            {"charger_id": "C01"},
            {"charger_id": "C02"},
        ]
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all("score" in item for item in result)
