"""Integration checks for the cleaned sessions produced by notebook 01."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "Charging_Data_clean.csv"


def test_cleaned_sessions_have_valid_times_and_no_overlaps():
    sessions = pd.read_csv(CLEAN_PATH, usecols=["charger_id", "start", "end"])
    sessions["start"] = pd.to_datetime(sessions["start"], errors="coerce")
    sessions["end"] = pd.to_datetime(sessions["end"], errors="coerce")

    assert sessions[["start", "end"]].notna().all().all()

    sessions = sessions.sort_values(["charger_id", "start"])
    prior_max_end = sessions.groupby("charger_id")["end"].transform(
        lambda ends: ends.cummax().shift()
    )
    first_sessions = prior_max_end.isna()
    assert first_sessions.sum() == 92
    assert sessions.loc[first_sessions, ["start", "end"]].notna().all().all()
    assert (prior_max_end.isna() | sessions["start"].ge(prior_max_end)).all()
