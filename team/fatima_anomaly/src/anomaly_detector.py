
"""Inference utilities for EV charging session anomaly detection.

The trained CatBoost model is expected to be available locally.
Set ANOMALY_MODEL_PATH to its location before calling predict_anomaly().
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


DEFAULT_THRESHOLD = 0.41
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "catboost_anomaly_model.cbm"
)

TARGET = "is_abnormal"

START_CANDIDATES = [
    "start_time",
    "Start Time",
    "start_datetime",
    "Start Date Time",
    "start_date",
    "Start Date",
    "start_time/date",
    "start_time_stamp",
]

END_CANDIDATES = [
    "end_time",
    "End Time",
    "end_datetime",
    "End Date Time",
    "end_date",
    "End Date",
    "end_time_stamp",
]

ID_NAMES = {
    "charger_id",
    "charger id",
    "charging_station_id",
    "station_id",
    "session_id",
    "transaction_id",
}


def _first_existing(columns, candidates):
    """Return the first candidate column present in the input."""
    return next((c for c in candidates if c in columns), None)


def _engineer_features(data):
    """Create the time and energy features used by the training pipeline."""
    df = data.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    start_col = _first_existing(df.columns, START_CANDIDATES)
    end_col = _first_existing(df.columns, END_CANDIDATES)

    if start_col is not None:
        start_dt = pd.to_datetime(df[start_col], errors="coerce")

        if "start_hour" not in df.columns:
            df["start_hour"] = start_dt.dt.hour
        if "start_dayofweek" not in df.columns:
            df["start_dayofweek"] = start_dt.dt.dayofweek
        if "start_month" not in df.columns:
            df["start_month"] = start_dt.dt.month
        if "start_is_weekend" not in df.columns:
            df["start_is_weekend"] = (
                start_dt.dt.dayofweek >= 5
            ).astype(float)

        if "start_hour_sin" not in df.columns:
            df["start_hour_sin"] = np.sin(
                2 * np.pi * df["start_hour"] / 24
            )
        if "start_hour_cos" not in df.columns:
            df["start_hour_cos"] = np.cos(
                2 * np.pi * df["start_hour"] / 24
            )
        if "start_day_sin" not in df.columns:
            df["start_day_sin"] = np.sin(
                2 * np.pi * df["start_dayofweek"] / 7
            )
        if "start_day_cos" not in df.columns:
            df["start_day_cos"] = np.cos(
                2 * np.pi * df["start_dayofweek"] / 7
            )

    if start_col is not None and end_col is not None:
        if "session_duration_minutes" not in df.columns:
            start_dt = pd.to_datetime(df[start_col], errors="coerce")
            end_dt = pd.to_datetime(df[end_col], errors="coerce")
            df["session_duration_minutes"] = (
                end_dt - start_dt
            ).dt.total_seconds() / 60

    energy_col = next(
        (
            c for c in df.columns
            if c.lower().replace(" ", "") in {
                "transactionpower/kwh",
                "transactionpower(kwh)",
                "transaction_power_kwh",
            }
        ),
        None,
    )

    if (
        energy_col is not None
        and "session_duration_minutes" in df.columns
        and "energy_per_hour" not in df.columns
    ):
        energy = pd.to_numeric(df[energy_col], errors="coerce")
        duration_hours = df["session_duration_minutes"] / 60
        df["energy_per_hour"] = (
            energy / duration_hours.replace(0, np.nan)
        )

    return df, start_col, end_col


def _prepare_features(data, model):
    """Align incoming records to the exact feature order expected by CatBoost."""
    df, start_col, end_col = _engineer_features(data)

    charger_col = next(
        (
            c for c in df.columns
            if c.lower().strip() in ID_NAMES
            and "charger" in c.lower()
        ),
        None,
    )

    if charger_col is None:
        charger_col = next(
            (
                c for c in df.columns
                if c.lower().strip() in ID_NAMES
            ),
            None,
        )

    if charger_col is not None:
        charger_ids = df[charger_col].to_numpy()
    else:
        charger_ids = np.arange(len(df))

    # Exclude target/outcome leakage, identifiers, and raw timestamps.
    excluded = {
        c for c in df.columns
        if c.lower().strip() == "end_cause"
        or (
            c.lower().strip().startswith("is_")
            and c != TARGET
        )
    }

    excluded.update(c for c in [start_col, end_col] if c is not None)
    excluded.update(
        c for c in df.columns
        if c.lower().strip() in ID_NAMES
    )
    excluded.add(TARGET)

    X = df.drop(
        columns=[c for c in excluded if c in df.columns],
        errors="ignore",
    ).copy()

    # Use the model's saved feature names and preserve training order.
    feature_names = model.feature_names_
    if not feature_names:
        raise ValueError(
            "The model has no saved feature names. "
            "Retrain/export it with named input columns."
        )

    X = X.reindex(columns=feature_names)

    # Match categorical columns recorded in the trained CatBoost model.
    categorical_indices = model.get_cat_feature_indices()

    for index in categorical_indices:
        column = X.columns[index]
        X[column] = X[column].where(X[column].notna(), "Missing")
        X[column] = X[column].astype(str)

    # CatBoost accepts missing numeric values; normalize infinities.
    for column in X.columns:
        if column not in {
            X.columns[i] for i in categorical_indices
        }:
            X[column] = pd.to_numeric(X[column], errors="coerce")
            X[column] = X[column].replace(
                [np.inf, -np.inf], np.nan
            )

    return X, charger_ids


def _explain_predictions(model, X, anomaly_flags):
    """Return a short reason based on the strongest positive SHAP contribution."""
    if len(X) == 0:
        return []

    pool = Pool(
        X,
        cat_features=model.get_cat_feature_indices(),
    )
    shap_values = model.get_feature_importance(
        pool,
        type="ShapValues",
    )

    feature_names = list(X.columns)
    reasons = []

    for row_index, is_anomaly in enumerate(anomaly_flags):
        if not is_anomaly:
            reasons.append("Score below decision threshold")
            continue

        # Last SHAP value is the expected/base value, not a feature.
        contributions = shap_values[row_index, :-1]

        if len(contributions) == 0:
            reasons.append("No feature explanation available")
            continue

        # Positive contribution pushes the model toward its positive class.
        strongest_index = int(np.argmax(contributions))
        strongest_value = float(contributions[strongest_index])
        strongest_feature = feature_names[strongest_index]

        if strongest_value <= 0:
            reasons.append(
                "Model flagged this session; no positive SHAP "
                "feature contribution was identified"
            )
        else:
            reasons.append(
                f"{strongest_feature} (positive SHAP contribution: "
                f"{strongest_value:.3f})"
            )

    return reasons


def predict_anomaly(
    data,
    model_path=None,
    threshold=DEFAULT_THRESHOLD,
    explain=True,
):
    """Predict anomalies for one or more charging sessions.

    Args:
        data: pandas DataFrame containing raw charging-session columns.
        model_path: path to the trained .cbm model; optional if the
            ANOMALY_MODEL_PATH environment variable or default path is used.
        threshold: probability threshold for the abnormal class.
        explain: whether to generate SHAP-based reasons.

    Returns:
        DataFrame with charger_id, anomaly, score, reason, anomaly_label.
        `score` is the predicted probability for the model's positive class.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        return pd.DataFrame(
            columns=[
                "charger_id",
                "anomaly",
                "score",
                "reason",
                "anomaly_label",
            ]
        )

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    configured_path = (
        model_path
        or os.getenv("ANOMALY_MODEL_PATH")
        or DEFAULT_MODEL_PATH
    )
    configured_path = Path(configured_path)

    if not configured_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {configured_path}. "
            "Train the model first and set ANOMALY_MODEL_PATH "
            "to the saved .cbm file."
        )

    model = CatBoostClassifier()
    model.load_model(str(configured_path))

    X, charger_ids = _prepare_features(data, model)

    probabilities = model.predict_proba(X)[:, 1]
    anomaly_flags = probabilities >= threshold

    if explain:
        reasons = _explain_predictions(model, X, anomaly_flags)
    else:
        reasons = [
            "Explanation disabled" if flag
            else "Score below decision threshold"
            for flag in anomaly_flags
        ]

    return pd.DataFrame(
        {
            "charger_id": charger_ids,
            "anomaly": anomaly_flags.astype(int),
            "score": probabilities,
            "reason": reasons,
            "anomaly_label": np.where(
                anomaly_flags, "abnormal", "normal"
            ),
        }
    )
