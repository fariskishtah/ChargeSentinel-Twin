"""Read-only raw audit and causal hourly event aggregation."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def aggregate(raw):
    """Energy recognized at completion; arrivals counted at start. Never mutate raw."""
    d = raw.copy(deep=True)
    start = pd.to_datetime(d["Start Time"].astype(str).str.strip(), errors="raise")
    end = pd.to_datetime(d["End Time"].astype(str).str.strip(), errors="raise")
    energy = pd.to_numeric(d["Transaction power/kwh"], errors="raise")
    if start.dt.tz is not None or end.dt.tz is not None:
        raise ValueError("Source timezone not independently verified; use naive operational time")
    if start.isna().any() or end.isna().any() or not np.isfinite(energy).all():
        raise ValueError("Invalid raw time/energy; audit required")
    if (end < start).any() or (energy < 0).any():
        raise ValueError("Invalid session chronology or negative energy")
    if d.duplicated().any():
        raise ValueError("Duplicate transactions require explicit source resolution")
    index = pd.date_range(min(start.min(), end.min()).floor("h"),
                          max(start.max(), end.max()).floor("h"), freq="h")
    arrivals = pd.Series(1, index=start.dt.floor("h")).groupby(level=0).sum()
    completed = pd.Series(energy.to_numpy(), index=end.dt.floor("h")).groupby(level=0).sum()
    return pd.DataFrame({"timestamp": index,
                         "energy_kwh": completed.reindex(index, fill_value=0).to_numpy(),
                         "sessions": arrivals.reindex(index, fill_value=0).to_numpy()})


def build_hourly(path):
    before = sha256(path)
    raw = pd.read_csv(path, encoding="gb18030")
    hourly = aggregate(raw)
    identifiers = {c: {"unique": int(raw[c].nunique()), "missing": int(raw[c].isna().sum())}
                   for c in ["UserID", "Charging Post ID"]}
    audit = {"raw_sha256": before, "raw_rows": len(raw), "schema": list(raw.columns),
             "dtypes": raw.dtypes.astype(str).to_dict(),
             "time_ranges": {c: {"first": str(pd.to_datetime(raw[c].str.strip()).min()),
                                  "last": str(pd.to_datetime(raw[c].str.strip()).max())}
                             for c in ["Start Time", "End Time"]},
             "missing": raw.isna().sum().astype(int).to_dict(),
             "duplicates": int(raw.duplicated().sum()), "identifiers": identifiers,
             "session_identifier": "No unique transaction ID; UserID is a repeated customer identifier",
             "hourly_rows": len(hourly), "first_hour": str(hourly.timestamp.min()),
             "last_hour": str(hourly.timestamp.max()),
             "zero_arrival_hours": int((hourly.sessions == 0).sum()),
             "zero_energy_hours": int((hourly.energy_kwh == 0).sum()),
             "energy_conserved": bool(np.isclose(hourly.energy_kwh.sum(), raw['Transaction power/kwh'].sum())),
             "sessions_conserved": int(hourly.sessions.sum()) == len(raw),
             "timezone": "Source timezone not independently verified; naive operational time",
             "zero_policy": "No recorded event => zero, conditional on complete source capture; outages cannot be distinguished"}
    if sha256(path) != before:
        raise RuntimeError("Raw source changed during read")
    return hourly, audit
