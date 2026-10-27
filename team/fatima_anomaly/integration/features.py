"""Only values known when a charging session starts enter the anomaly model."""

from __future__ import annotations

import pandas as pd

FEATURES = ["charger_id", "location", "district", "start_hour", "start_dayofweek", "start_month", "start_is_weekend"]
CATEGORICAL = ["charger_id", "location", "district"]


def feature_record(charger_id: str, data: dict[str, object]) -> dict[str, object]:
    if not isinstance(data, dict) or set(data) != {"location", "district", "start_time"}:
        raise ValueError("Require location, district, and start_time only.")
    if not isinstance(charger_id, str) or not charger_id.strip():
        raise ValueError("charger_id is required.")
    for key in ("location", "district"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} is required.")
    ts = pd.Timestamp(data["start_time"])
    if pd.isna(ts) or ts.tzinfo is not None:
        raise ValueError("start_time must be a naive source timestamp.")
    return {"charger_id": charger_id.strip(), "location": data["location"].strip(),
            "district": data["district"].strip(), "start_hour": ts.hour,
            "start_dayofweek": ts.dayofweek, "start_month": ts.month,
            "start_is_weekend": int(ts.dayofweek >= 5)}


def raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(df["Start Time"].astype(str).str.strip(), errors="coerce")
    if start.isna().any():
        raise ValueError("Invalid Start Time in source data.")
    out = pd.DataFrame({"charger_id": df["Charging Post ID"].astype(str),
                        "location": df["Location Information"].astype(str),
                        "district": df["District Name"].astype(str),
                        "start_hour": start.dt.hour,
                        "start_dayofweek": start.dt.dayofweek,
                        "start_month": start.dt.month,
                        "start_is_weekend": (start.dt.dayofweek >= 5).astype(int)})
    return out[FEATURES]
