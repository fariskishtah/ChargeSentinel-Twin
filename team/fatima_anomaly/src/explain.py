
"""SHAP-based explanations for EV charging anomaly predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier

from features import prepare_features


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
REPORT_DIR = PROJECT_DIR / "reports"

DEFAULT_MODEL_PATH = ARTIFACT_DIR / "catboost_anomaly_model.cbm"
DEFAULT_CONFIG_PATH = ARTIFACT_DIR / "model_config.json"


def load_model_and_config(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Load the trained CatBoost model and its training configuration."""
    model_path = Path(model_path)
    config_path = Path(config_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    return model, config


def explain_records(
    records: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return predictions and top SHAP contributions for input records."""
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("records must be a non-empty pandas DataFrame.")

    model, config = load_model_and_config(model_path, config_path)

    feature_names = config["features"]
    categorical_features = config.get("categorical_features", [])

    X = prepare_features(
        records,
        feature_names=feature_names,
        categorical_features=categorical_features,
    )

    shap_values = np.asarray(
        model.get_feature_importance(
            data=X,
            type="ShapValues",
        )
    )

    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(config["decision_threshold"])
    predictions = (probabilities >= threshold).astype(int)

    rows = []

    for row_index in range(len(X)):
        contributions = shap_values[row_index, :-1]

        if predictions[row_index] == 1:
            order = np.argsort(contributions)[::-1]
            direction = "increases anomaly score"
            indices = [i for i in order if contributions[i] > 0]
        else:
            order = np.argsort(contributions)
            direction = "supports normal prediction"
            indices = [i for i in order if contributions[i] < 0]

        top_indices = indices[:top_k]

        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "record_index": row_index,
                    "prediction": (
                        "anomaly" if predictions[row_index] else "normal"
                    ),
                    "anomaly_score": float(probabilities[row_index]),
                    "threshold": threshold,
                    "model_version": config.get(
                        "model_version", "unknown"
                    ),
                    "rank": rank,
                    "feature": X.columns[feature_index],
                    "feature_value": str(
                        X.iloc[row_index, feature_index]
                    ),
                    "shap_value": float(contributions[feature_index]),
                    "interpretation": direction,
                }
            )

    return pd.DataFrame(rows)


def save_global_importance(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = REPORT_DIR / "shap_global_importance.csv",
) -> pd.DataFrame:
    """Export global SHAP importance using a labeled sample of records.

    Supply representative records separately when using this function.
    """
    raise NotImplementedError(
        "Call explain_records() on representative labeled data first. "
        "Global SHAP importance should be calculated from a representative "
        "dataset, not from a single arbitrary record."
    )


if __name__ == "__main__":
    print(
        "Import explain_records(records) and pass a pandas DataFrame "
        "containing representative charging sessions."
    )
