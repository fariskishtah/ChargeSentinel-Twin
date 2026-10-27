"""Run from project root:  pytest tests/test_occupancy.py -v"""
import pandas as pd
import pytest

from src.occupancy import (load_sessions, hourly_busy, build_intervals,
                           validate_intervals, charger_state_at,
                           hourly_occupancy_matrix)


@pytest.fixture(scope="module")
def sessions():
    try:
        return load_sessions()
    except FileNotFoundError as exc:
        pytest.skip(f"clean charging dataset is unavailable: {exc}")


@pytest.fixture(scope="module")
def iv(sessions):
    return build_intervals(sessions)


def _toy(rows):
    d = pd.DataFrame(rows, columns=["charger_id", "start", "end"])
    d[["start", "end"]] = d[["start", "end"]].apply(pd.to_datetime)
    d["location"], d["district"], d["is_abnormal"] = "X", "D", 0
    d["duration_min"] = (d["end"] - d["start"]).dt.total_seconds() / 60
    return build_intervals(d)


def test_checker_passes_clean_data():
    iv_ok = _toy([(1, "2021-01-01 10:00", "2021-01-01 10:30"),
                  (1, "2021-01-01 11:00", "2021-01-01 11:20")])
    assert sum(validate_intervals(iv_ok).values()) == 0


def test_checker_detects_simple_overlap():
    bad = _toy([(1, "2021-01-01 10:00", "2021-01-01 10:30"),
                (1, "2021-01-01 10:20", "2021-01-01 11:00")])
    assert validate_intervals(bad)["overlap_with_previous"] == 1


def test_checker_detects_nested_overlap():
    # session B is inside long session A, session C overlaps A but not B
    bad = _toy([(1, "2021-01-01 08:00", "2021-01-01 12:00"),
                (1, "2021-01-01 09:00", "2021-01-01 09:10"),
                (1, "2021-01-01 10:00", "2021-01-01 10:30")])
    v = validate_intervals(bad)
    assert v["overlap_with_any_earlier"] >= 2


def test_checker_detects_zero_length():
    bad = _toy([(1, "2021-01-01 10:00", "2021-01-01 10:00")])
    assert validate_intervals(bad)["end_not_after_start"] == 1


def test_same_time_on_different_chargers_is_allowed():
    ok = _toy([(1, "2021-01-01 10:00", "2021-01-01 10:30"),
               (2, "2021-01-01 10:00", "2021-01-01 10:30")])
    assert sum(validate_intervals(ok).values()) == 0


# ---------------- real data: overlap and chronology ----------------
def test_no_violations_in_real_data(iv):
    v = validate_intervals(iv)
    assert all(x == 0 for x in v.values()), v


def test_every_session_ends_after_it_starts(iv):
    assert (iv["end"] > iv["start"]).all()


def test_sessions_sorted_in_time_within_each_charger(iv):
    assert iv.groupby("charger_id")["start"].apply(lambda s: s.is_monotonic_increasing).all()


def test_each_session_starts_after_previous_one_ends(iv):
    assert (iv["gap_before_min"].dropna() >= 0).all()


def test_no_rows_lost_or_invented(sessions, iv):
    assert len(iv) == len(sessions)


def test_every_charger_has_one_location(iv):
    if "location" not in iv.columns:
        pytest.skip("dataset has no location column")
    assert (iv.groupby("charger_id")["location"].nunique() == 1).all()


# ---------------- hourly occupancy ----------------
def test_hourly_busy_between_0_and_60_minutes(iv):
    h = hourly_busy(iv)
    assert h["busy_min"].between(0, 60 + 1e-6).all()


def test_total_busy_minutes_preserved(iv):
    h = hourly_busy(iv)
    assert h["busy_min"].sum() == pytest.approx(iv["duration_min"].sum(), rel=1e-6)


def test_occupancy_matrix_is_a_fraction(iv):
    m = hourly_occupancy_matrix(hourly_busy(iv))
    assert m.values.min() >= 0 and m.values.max() <= 1 + 1e-9


# ---------------- state query ----------------
def test_state_at_matches_a_known_session(iv):
    row = iv.iloc[1000]
    t = row["start"] + (row["end"] - row["start"]) / 2          # middle of a session
    s = charger_state_at(iv, t)
    assert bool(s.loc[row["charger_id"], "busy"])
    assert s.loc[row["charger_id"], "true_remaining_min"] == pytest.approx(row["duration_min"] / 2, rel=1e-6)


def test_state_at_free_between_sessions(iv):
    gaps = iv[iv["gap_before_min"] > 30].iloc[0]
    t = gaps["prev_end"] + pd.Timedelta(minutes=5)
    s = charger_state_at(iv, t)
    assert not bool(s.loc[gaps["charger_id"], "busy"])
