"""Session loading and occupancy reconstruction for the Jiaxing data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_SESSION_COLUMNS = ["charger_id", "start", "end"]
OPTIONAL_SESSION_COLUMNS = ["location", "district", "is_abnormal", "duration_min"]


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data").is_dir() and (candidate / "team").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find repository root from {start}")


def load_sessions(path: str | Path | None = None) -> pd.DataFrame:
    """Load the already-standardized cleaned CSV; schema renaming belongs in the data-cleaning notebook."""
    path = Path(path) if path is not None else find_project_root() / "data/processed/Charging_Data_clean.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Clean sessions dataset not found: {path}")
    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="cp1252")
    missing = [column for column in REQUIRED_SESSION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            f"Clean dataset is missing standardized columns {missing}; rerun 01_data_cleaning.ipynb"
        )
    frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
    if "duration_min" not in frame:
        frame["duration_min"] = (frame["end"] - frame["start"]).dt.total_seconds() / 60
    return frame


def build_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_SESSION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Sessions DataFrame is missing columns {missing}")
    columns = REQUIRED_SESSION_COLUMNS + [c for c in OPTIONAL_SESSION_COLUMNS if c in frame]
    intervals = frame[columns].sort_values(["charger_id", "start"]).reset_index(drop=True).copy()
    groups = intervals.groupby("charger_id", sort=False)
    intervals["prev_end"] = groups["end"].shift()
    intervals["gap_before_min"] = (intervals["start"] - intervals["prev_end"]).dt.total_seconds() / 60
    intervals["seq"] = groups.cumcount()
    return intervals


def validate_intervals(intervals: pd.DataFrame) -> dict[str, int]:
    required = ["charger_id", "start", "end"]
    missing = [column for column in required if column not in intervals]
    if missing:
        raise ValueError(f"Intervals DataFrame is missing columns {missing}")
    ordered = intervals.sort_values(["charger_id", "start"])
    groups = ordered.groupby("charger_id", sort=False)
    prev_end = groups["end"].shift()
    earlier_max = groups["end"].transform(lambda values: values.cummax().shift())
    gaps = intervals["gap_before_min"] if "gap_before_min" in intervals else (
        intervals["start"] - intervals.groupby("charger_id")["end"].shift()
    ).dt.total_seconds() / 60
    return {
        "end_not_after_start": int((intervals["end"] <= intervals["start"]).sum()),
        "overlap_with_previous": int((ordered["start"] < prev_end).sum()),
        "overlap_with_any_earlier": int((ordered["start"] < earlier_max).sum()),
        "chargers_not_sorted": int(sum(not x.is_monotonic_increasing for _, x in intervals.groupby("charger_id")["start"])),
        "duplicate_charger_start": int(intervals.duplicated(["charger_id", "start"]).sum()),
        "missing_values": int(intervals[required].isna().sum().sum()),
        "negative_gaps": int((gaps < 0).sum()),
    }


def hourly_busy(frame: pd.DataFrame) -> pd.DataFrame:
    """Split positive-length sessions into charger/hour busy-minute totals."""
    missing = [column for column in REQUIRED_SESSION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Sessions DataFrame is missing columns {missing}")
    valid = frame.loc[
        frame["start"].notna() & frame["end"].notna() & (frame["end"] > frame["start"]),
        ["charger_id", "start", "end"],
    ].copy()
    if valid.empty:
        return pd.DataFrame(columns=["charger_id", "hour", "busy_min"])
    start_hour = valid["start"].dt.floor("h")
    end_hour = valid["end"].dt.floor("h")
    counts = ((end_hour - start_hour).dt.total_seconds() / 3600).astype(int).to_numpy().copy()
    counts += valid["end"].ne(end_hour).to_numpy().astype(int)
    counts = np.maximum(counts, 1)
    repeated = np.repeat(np.arange(len(valid)), counts)
    within = np.arange(len(repeated)) - np.repeat(np.cumsum(counts) - counts, counts)
    hours = pd.to_datetime(valid["start"].to_numpy()[repeated]).floor("h") + pd.to_timedelta(within, unit="h")
    next_hours = hours + pd.Timedelta(hours=1)
    starts = valid["start"].to_numpy()[repeated]
    ends = valid["end"].to_numpy()[repeated]
    busy = pd.to_timedelta(np.minimum(ends, next_hours.to_numpy()) - np.maximum(starts, hours.to_numpy()))
    pieces = pd.DataFrame({
        "charger_id": valid["charger_id"].to_numpy()[repeated],
        "hour": hours,
        "busy_min": busy.total_seconds() / 60,
    })
    return pieces.groupby(["charger_id", "hour"], as_index=False, sort=True)["busy_min"].sum()


def charger_state_at(intervals: pd.DataFrame, timestamp: str | pd.Timestamp) -> pd.DataFrame:
    """Historical replay state. Remaining-time fields are hindsight ground truth."""
    timestamp = pd.Timestamp(timestamp)
    started = intervals.loc[intervals["start"] <= timestamp]
    last = started.groupby("charger_id", sort=False).tail(1).set_index("charger_id")
    out = pd.DataFrame(index=sorted(intervals["charger_id"].dropna().unique()))
    out["busy"] = (last["end"] > timestamp).reindex(out.index).fillna(False).astype(bool)
    out["session_start"] = last["start"].where(out["busy"])
    out["elapsed_min"] = (timestamp - out["session_start"]).dt.total_seconds() / 60
    out["true_end"] = last["end"].where(out["busy"])
    out["true_remaining_min"] = (out["true_end"] - timestamp).dt.total_seconds() / 60
    return out


def hourly_occupancy_matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()
    return hourly.pivot(index="hour", columns="charger_id", values="busy_min").fillna(0) / 60
