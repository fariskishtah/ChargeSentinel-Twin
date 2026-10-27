"""Load cleaned charging sessions and reconstruct charger occupancy."""
from pathlib import Path
import os

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = Path(os.environ.get("CHARGESENTINEL_DATA_DIR", _PROJECT_ROOT / "data"))

_COLUMN_ALIASES = {
    "charger_id": ("charger_id", "Charging Post ID"),
    "start": ("start", "Start Time"),
    "end": ("end", "End Time"),
    "duration_min": ("duration_min",),
    "location": ("location", "Location Information"),
    "district": ("district", "District Name"),
    "is_abnormal": ("is_abnormal",),
}


def load_sessions(path=None) -> pd.DataFrame:
    """Load clean sessions, standardize occupancy columns, and parse timestamps.

    With no path, reads the clean CSV from ``CHARGESENTINEL_DATA_DIR`` or the
    repository's ``data`` directory. The successful text encoding is recorded
    in ``DataFrame.attrs["encoding"]``.
    """
    dataset_path = Path(path) if path is not None else (
        _DATA_DIR / "processed" / "Charging_Data_clean.csv"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Clean sessions dataset not found at: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="cp1252")
        encoding = "cp1252"

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].str.strip()

    rename = {}
    for target, aliases in _COLUMN_ALIASES.items():
        if target not in df.columns:
            source = next((name for name in aliases if name in df.columns), None)
            if source is not None:
                rename[source] = target
    df = df.rename(columns=rename)

    required = ["charger_id", "start", "end"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Sessions dataset is missing required columns: {missing}")

    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60
    df.attrs["encoding"] = encoding
    df.attrs["source_path"] = str(dataset_path.resolve())
    return df


def build_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Return busy intervals with helper columns, sorted by charger and time."""
    required = ["charger_id", "start", "end"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Sessions DataFrame is missing required columns: {missing}")

    optional = ["location", "district", "duration_min", "is_abnormal"]
    cols = required + [column for column in optional if column in df.columns]
    iv = df[cols].sort_values(["charger_id", "start"]).reset_index(drop=True).copy()
    g = iv.groupby("charger_id")
    iv["prev_end"] = g["end"].shift()
    iv["gap_before_min"] = (iv["start"] - iv["prev_end"]).dt.total_seconds() / 60
    iv["seq"] = g.cumcount()
    return iv


def validate_intervals(iv: pd.DataFrame) -> dict:
    """Count timeline violations. All values must be 0 for a valid timeline."""
    prev_end = iv.groupby("charger_id")["end"].shift()
    run_max_end = iv.groupby("charger_id")["end"].transform(lambda s: s.cummax().shift())
    sorted_ok = iv.groupby("charger_id")["start"].apply(lambda s: s.is_monotonic_increasing)
    return {
        "end_not_after_start": int((iv["end"] <= iv["start"]).sum()),
        "overlap_with_previous": int((iv["start"] < prev_end).sum()),
        "overlap_with_any_earlier": int((iv["start"] < run_max_end).sum()),
        "chargers_not_sorted": int((~sorted_ok).sum()),
        "duplicate_charger_start": int(iv.duplicated(["charger_id", "start"]).sum()),
        "missing_values": int(iv[["charger_id", "start", "end"]].isna().sum().sum()),
        "negative_gaps": int((iv["gap_before_min"] < 0).sum()),
    }


def hourly_busy(df: pd.DataFrame) -> pd.DataFrame:
    """Split each session across clock hours; return busy minutes per charger/hour."""
    required = ["charger_id", "start", "end"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Sessions DataFrame is missing required columns: {missing}")

    records = []
    for charger_id, start, end in df[required].itertuples(index=False, name=None):
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        hour = pd.Timestamp(start).floor("h")
        end = pd.Timestamp(end)
        while hour < end:
            next_hour = hour + pd.Timedelta(hours=1)
            minutes = (min(end, next_hour) - max(start, hour)).total_seconds() / 60
            if minutes > 0:
                records.append((charger_id, hour, minutes))
            hour = next_hour

    if not records:
        return pd.DataFrame(columns=["charger_id", "hour", "busy_min"])
    return (pd.DataFrame(records, columns=["charger_id", "hour", "busy_min"])
            .groupby(["charger_id", "hour"], as_index=False, sort=True)["busy_min"].sum())


def charger_state_at(iv: pd.DataFrame, t) -> pd.DataFrame:
    """For every charger, report busy state and true end for replay ground truth."""
    t = pd.Timestamp(t)
    started = iv[iv["start"] <= t]
    last = started.groupby("charger_id").tail(1).set_index("charger_id")
    out = pd.DataFrame(index=sorted(iv["charger_id"].unique()))
    out["busy"] = (last["end"] > t).reindex(out.index).fillna(False)
    out["session_start"] = last["start"].where(out["busy"])
    out["elapsed_min"] = (t - out["session_start"]).dt.total_seconds() / 60
    out["true_end"] = last["end"].where(out["busy"])
    out["true_remaining_min"] = (out["true_end"] - t).dt.total_seconds() / 60
    if "location" in iv.columns:
        out["location"] = iv.groupby("charger_id")["location"].first()
    return out


def hourly_occupancy_matrix(busy_hourly: pd.DataFrame) -> pd.DataFrame:
    """Return busy-hour fractions; rows are hours and columns are chargers."""
    matrix = busy_hourly.pivot(index="hour", columns="charger_id", values="busy_min").fillna(0) / 60
    return matrix.sort_index()
