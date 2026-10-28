"""Run from team/mariam_availability: python -m pytest tests/test_occupancy.py -v"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.occupancy import (
    build_intervals,
    charger_state_at,
    hourly_busy,
    hourly_occupancy_matrix,
    load_sessions,
    validate_intervals,
)


def toy(rows):
    frame = pd.DataFrame(rows, columns=["charger_id", "start", "end"])
    frame[["start", "end"]] = frame[["start", "end"]].apply(pd.to_datetime)
    frame["location"], frame["district"], frame["is_abnormal"] = "X", "D", 0
    frame["duration_min"] = (frame["end"] - frame["start"]).dt.total_seconds() / 60
    return build_intervals(frame)


def test_checker_passes_clean_data():
    intervals = toy([(1, "2021-01-01 10:00", "2021-01-01 10:30"),
                     (1, "2021-01-01 11:00", "2021-01-01 11:20")])
    assert sum(validate_intervals(intervals).values()) == 0


def test_checker_detects_simple_and_nested_overlaps():
    simple = toy([(1, "2021-01-01 10:00", "2021-01-01 10:30"),
                  (1, "2021-01-01 10:20", "2021-01-01 11:00")])
    assert validate_intervals(simple)["overlap_with_previous"] == 1
    nested = toy([(1, "2021-01-01 08:00", "2021-01-01 12:00"),
                  (1, "2021-01-01 09:00", "2021-01-01 09:10"),
                  (1, "2021-01-01 10:00", "2021-01-01 10:30")])
    assert validate_intervals(nested)["overlap_with_any_earlier"] >= 2


def test_checker_detects_zero_length_and_duplicate_start():
    invalid = toy([(1, "2021-01-01 10:00", "2021-01-01 10:00"),
                   (1, "2021-01-01 10:00", "2021-01-01 10:30")])
    counts = validate_intervals(invalid)
    assert counts["end_not_after_start"] == 1
    assert counts["duplicate_charger_start"] == 1


def test_same_time_on_different_chargers_is_allowed():
    intervals = toy([(1, "2021-01-01 10:00", "2021-01-01 10:30"),
                     (2, "2021-01-01 10:00", "2021-01-01 10:30")])
    assert sum(validate_intervals(intervals).values()) == 0


def test_hourly_busy_splits_sessions_and_preserves_total_minutes():
    sessions = pd.DataFrame({
        "charger_id": [1, 2],
        "start": pd.to_datetime(["2021-01-01 10:45", "2021-01-01 11:00"]),
        "end": pd.to_datetime(["2021-01-01 11:15", "2021-01-01 11:30"]),
    })
    hourly = hourly_busy(sessions)
    assert hourly["busy_min"].between(0, 60 + 1e-6).all()
    assert hourly["busy_min"].sum() == pytest.approx(60)
    assert len(hourly) == 3
    matrix = hourly_occupancy_matrix(hourly)
    assert matrix.to_numpy().min() >= 0 and matrix.to_numpy().max() <= 1


def test_state_at_reports_busy_and_free_chargers():
    intervals = toy([(1, "2021-01-01 10:00", "2021-01-01 10:30"),
                     (1, "2021-01-01 11:00", "2021-01-01 11:20"),
                     (2, "2021-01-01 10:00", "2021-01-01 10:30")])
    state = charger_state_at(intervals, "2021-01-01 10:15")
    assert state.loc[1, "busy"] and state.loc[2, "busy"]
    assert state.loc[1, "true_remaining_min"] == pytest.approx(15)
    free = charger_state_at(intervals, "2021-01-01 10:45")
    assert not free.loc[1, "busy"]


def test_loader_requires_data_cleaning_standardized_schema():
    with tempfile.NamedTemporaryFile(suffix=".csv", dir=Path(__file__).parent, delete=False) as handle:
        raw_path = Path(handle.name)
    try:
        pd.DataFrame({"Charging Post ID": [1], "Start Time": ["2021-01-01"],
                      "End Time": ["2021-01-01 00:10"]}).to_csv(raw_path, index=False)
        with pytest.raises(ValueError, match="standardized columns"):
            load_sessions(raw_path)
    finally:
        raw_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def sessions():
    try:
        return load_sessions()
    except FileNotFoundError as exc:
        pytest.skip(f"clean charging dataset is unavailable: {exc}")


@pytest.fixture(scope="module")
def real_intervals(sessions):
    return build_intervals(sessions)


def test_real_data_has_no_timeline_violations(real_intervals):
    counts = validate_intervals(real_intervals)
    assert all(value == 0 for value in counts.values()), counts
    assert len(real_intervals) > 0


def test_real_hourly_busy_stays_within_capacity_and_conserves_duration(real_intervals):
    hourly = hourly_busy(real_intervals)
    assert hourly["busy_min"].between(0, 60 + 1e-6).all()
    assert hourly["busy_min"].sum() == pytest.approx(real_intervals["duration_min"].sum(), rel=1e-6)
    occupancy = hourly_occupancy_matrix(hourly)
    assert occupancy.to_numpy().min() >= 0 and occupancy.to_numpy().max() <= 1 + 1e-9


def test_real_sessions_keep_one_location_per_charger(real_intervals):
    if "location" not in real_intervals:
        pytest.skip("clean dataset has no location column")
    assert (real_intervals.groupby("charger_id")["location"].nunique() <= 1).all()
