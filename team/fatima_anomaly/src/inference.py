
"""Inference interface for EV charging anomaly detection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from team.fatima_anomaly.src.features import prepare_features

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

DEFAULT_MODEL_PATH = ARTIFACT_DIR / "catboost_anomaly_model.cbm"
DEFAULT_CONFIG_PATH = ARTIFACT_DIR / "model_config.json"


_model = None
_config = None


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load training metadata needed for consistent inference."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = [
        "target_column",
        "decision_threshold",
        "features",
        "categorical_features",
        "model_version",
    ]

    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"Configuration is missing required keys: {missing}"
        )

    return config


def _get_model(
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Load the model and configuration once per process."""
    global _model, _config

    resolved_model_path = Path(
        model_path or os.getenv(
            "ANOMALY_MODEL_PATH", str(DEFAULT_MODEL_PATH)
        )
    )
    resolved_config_path = Path(
        config_path or os.getenv(
            "ANOMALY_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)
        )
    )

    # Reload when callers specify paths different from cached defaults.
    cache_matches = (
        _model is not None
        and _config is not None
        and _config.get("_model_path") == str(resolved_model_path)
        and _config.get("_config_path") == str(resolved_config_path)
    )

    if cache_matches:
        return _model, _config

    if not resolved_model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {resolved_model_path}. "
            "Train the model first."
        )

    config = _load_config(resolved_config_path)

    model = CatBoostClassifier()
    model.load_model(str(resolved_model_path))

    config["_model_path"] = str(resolved_model_path)
    config["_config_path"] = str(resolved_config_path)

    _model = model
    _config = config

    return _model, _config


def _to_dataframe(payload: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Convert one record or a list of records into a DataFrame."""
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list) and payload:
        records = payload
    else:
        raise ValueError(
            "payload must be a record dictionary or a non-empty list "
            "of record dictionaries."
        )

    if not all(isinstance(record, dict) for record in records):
        raise TypeError("Every payload record must be a dictionary.")

    return pd.DataFrame(records)


def _build_reasons(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    prediction: int,
    top_k: int = 3,
) -> list[str]:
    """Return the strongest model feature contributions for one record."""
    try:
        contributions = model.get_feature_importance(
            data=X,
            type="ShapValues",
        )
        values = np.asarray(contributions)[0, :-1]

        if prediction == 1:
            indices = np.argsort(values)[::-1]
            selected = [
                i for i in indices
                if values[i] > 0
            ][:top_k]
        else:
            indices = np.argsort(values)
            selected = [
                i for i in indices
                if values[i] < 0
            ][:top_k]

        return [
            f"{X.columns[i]} contributed toward "
            f"{'anomaly' if prediction == 1 else 'normal'} prediction"
            for i in selected
        ]
    except Exception:
        return ["Feature-level explanation unavailable"]


def predict_anomaly(
    payload: dict[str, Any] | list[dict[str, Any]],
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Predict whether one or more charging sessions are anomalous.

    Returns one dictionary for a single record, or a list for a batch.
    """
    records = _to_dataframe(payload)
    model, config = _get_model(model_path, config_path)

    id_candidates = [
        col for col in records.columns
        if col.lower().strip() in {
            "charger_id",
            "charger id",
            "charging_station_id",
            "station_id",
            "session_id",
            "transaction_id",
        }
    ]
    id_col = id_candidates[0] if id_candidates else None

    X = prepare_features(
        records,
        feature_names=config["features"],
        categorical_features=config["categorical_features"],
    )

    # CatBoost expects missing numeric values as NaN and categorical
    # values as strings, matching the training preprocessing.
    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(config["decision_threshold"])
    predictions = (probabilities >= threshold).astype(int)

    results = []

    for i, prediction in enumerate(predictions):
        record_id = (
            records.iloc[i][id_col]
            if id_col is not None
            else None
        )

        reasons = _build_reasons(
            model,
            X.iloc[[i]],
            int(prediction),
        )

        results.append(
            {
                "charger_id": (
                    str(record_id)
                    if pd.notna(record_id)
                    else None
                ),
                "anomaly": bool(prediction),
                "score": float(probabilities[i]),
                "reason": reasons,
                "model_version": config["model_version"],
            }
        )

    return results[0] if isinstance(payload, dict) else results
