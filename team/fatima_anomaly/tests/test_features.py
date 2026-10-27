
"""Tests for shared EV charging feature preparation."""

import numpy as np
import pandas as pd

from team.fatima_anomaly.src.features import (
    engineer_features,
    get_excluded_columns,
    prepare_features,
)


def sample_data():
    return pd.DataFrame(
        {
            "charger_id": ["C01", "C02", "C03"],
            "start_time": [
                "2026-01-05 08:00:00",
                "2026-01-06 14:30:00",
                "2026-01-10 22:15:00",
            ],
            "end_time": [
                "2026-01-05 09:00:00",
                "2026-01-06 15:00:00",
                "2026-01-10 23:15:00",
            ],
            "Transaction Power (kWh)": [20.0, 10.0, 30.0],
            "is_abnormal": [0, 1, 0],
            "is_fault": [0, 1, 0],
            "end_cause": ["normal", "fault", "normal"],
            "temperature": [25.0, np.nan, 30.0],
        }
    )


def test_target_and_outcome_columns_are_excluded():
    df = sample_data()
    excluded = get_excluded_columns(df)

    assert "is_abnormal" in excluded
    assert "is_fault" in excluded
    assert "end_cause" in excluded
    assert "charger_id" in excluded


def test_time_features_are_created():
    result = engineer_features(sample_data())

    expected = {
        "start_hour",
        "start_dayofweek",
        "start_month",
        "start_is_weekend",
        "start_hour_sin",
        "start_hour_cos",
        "start_day_sin",
        "start_day_cos",
        "session_duration_minutes",
    }

    assert expected.issubset(result.columns)


def test_session_duration_is_calculated_in_minutes():
    result = engineer_features(sample_data())

    assert result["session_duration_minutes"].tolist() == [
        60.0,
        30.0,
        60.0,
    ]


def test_prepare_features_excludes_leakage_columns():
    X = prepare_features(sample_data())

    assert "is_abnormal" not in X.columns
    assert "is_fault" not in X.columns
    assert "end_cause" not in X.columns
    assert "charger_id" not in X.columns


def test_feature_order_is_preserved():
    df = sample_data()
    first = prepare_features(df)
    expected_order = list(first.columns)

    second = prepare_features(
        df,
        feature_names=expected_order,
    )

    assert list(second.columns) == expected_order


def test_missing_numeric_values_are_allowed():
    X = prepare_features(sample_data())

    assert "temperature" in X.columns
    assert X["temperature"].isna().sum() == 1
