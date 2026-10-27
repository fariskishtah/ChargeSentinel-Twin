
"""Shared feature preparation for EV charging anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create time and energy features without modifying the input."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    df = data.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    start_col = next(
        (c for c in START_CANDIDATES if c in df.columns), None
    )
    end_col = next(
        (c for c in END_CANDIDATES if c in df.columns), None
    )

    if start_col is not None:
        start_dt = pd.to_datetime(df[start_col], errors="coerce")

        df["start_hour"] = start_dt.dt.hour
        df["start_dayofweek"] = start_dt.dt.dayofweek
        df["start_month"] = start_dt.dt.month
        df["start_is_weekend"] = (
            start_dt.dt.dayofweek >= 5
        ).astype(float)

        df["start_hour_sin"] = np.sin(
            2 * np.pi * df["start_hour"] / 24
        )
        df["start_hour_cos"] = np.cos(
            2 * np.pi * df["start_hour"] / 24
        )
        df["start_day_sin"] = np.sin(
            2 * np.pi * df["start_dayofweek"] / 7
        )
        df["start_day_cos"] = np.cos(
            2 * np.pi * df["start_dayofweek"] / 7
        )

    if start_col is not None and end_col is not None:
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
    ):
        duration_hours = df["session_duration_minutes"] / 60
        energy = pd.to_numeric(df[energy_col], errors="coerce")
        df["energy_per_hour"] = (
            energy / duration_hours.replace(0, np.nan)
        )

    return df


def get_excluded_columns(data: pd.DataFrame) -> list[str]:
    """Identify targets, identifiers, raw timestamps, and outcome fields."""
    excluded = []

    for col in data.columns:
        normalized = col.lower().strip()

        if (
            normalized == TARGET
            or normalized == "end_cause"
            or normalized.startswith("is_")
            or normalized in ID_NAMES
        ):
            excluded.append(col)

    for col in START_CANDIDATES + END_CANDIDATES:
        if col in data.columns and col not in excluded:
            excluded.append(col)

    return excluded


def prepare_features(
    data: pd.DataFrame,
    feature_names: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare model inputs and optionally align to saved training metadata."""
    df = engineer_features(data)
    excluded = get_excluded_columns(df)

    X = df.drop(columns=excluded, errors="ignore").copy()
    X = X.dropna(axis=1, how="all")

    if feature_names is not None:
        # Reindex to the exact feature order recorded during training.
        X = X.reindex(columns=feature_names)

    if categorical_features is None:
        categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].where(X[col].notna(), "Missing").astype(str)

    categorical_set = set(categorical_features)

    for col in X.columns:
        if col not in categorical_set:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    return X
