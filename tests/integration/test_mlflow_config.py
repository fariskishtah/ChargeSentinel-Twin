"""Tests for shared MLflow configuration without starting a server."""

import sys
from types import SimpleNamespace

import pytest

from src.common.mlflow_config import (
    EXPERIMENT_NAMES,
    get_experiment_name,
    get_or_create_experiment,
    get_tracking_uri,
)


def test_experiment_names_are_centralized() -> None:
    assert EXPERIMENT_NAMES == {
        "solar": "chargesentinel-solar",
        "demand": "chargesentinel-demand",
        "anomaly": "chargesentinel-anomaly",
        "availability": "chargesentinel-availability",
    }


def test_tracking_uri_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.test:5000")
    assert get_tracking_uri() == "http://mlflow.example.test:5000"


def test_get_or_create_experiment_configures_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, str] = {}

    def set_tracking_uri(uri: str) -> None:
        calls["tracking_uri"] = uri

    def set_experiment(name: str) -> SimpleNamespace:
        calls["experiment"] = name
        return SimpleNamespace(experiment_id="42")

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=set_tracking_uri,
        set_experiment=set_experiment,
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    assert get_or_create_experiment("solar") == "42"
    assert calls == {
        "tracking_uri": "http://localhost:5000",
        "experiment": "chargesentinel-solar",
    }


def test_unknown_experiment_module_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_experiment_name("unknown")  # type: ignore[arg-type]
