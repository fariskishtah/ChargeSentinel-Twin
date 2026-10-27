"""Target-indexed causal features. Row u predicts hour u using hours <= u-1."""
import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 24, 48, 168)
TARGETS = ["energy_kwh", "sessions"]
FEATURES = ([f"{c}_lag_{lag}" for c in TARGETS for lag in LAGS]
            + [f"{c}_{stat}_{window}" for c in TARGETS for window in (6, 24)
               for stat in ("mean", "std")]
            + ["hour", "dayofweek", "weekend", "month", "sin_hour", "cos_hour",
               "sin_dayofweek", "cos_dayofweek"])
SEQUENCE_LENGTH = 24
HISTORY_HOURS = 192
FEATURE_VERSION = "causal-completion-v1"


def validate_hourly(frame):
    d = frame[["timestamp", *TARGETS]].copy()
    d["timestamp"] = pd.to_datetime(d.timestamp, errors="raise")
    if d.timestamp.dt.tz is not None:
        raise ValueError("Use naive source operational timestamps; timezone not verified")
    if d.timestamp.isna().any() or not d.timestamp.eq(d.timestamp.dt.floor("h")).all():
        raise ValueError("Timestamps must be valid whole hours")
    if len(d) > 1 and not d.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("History must be ordered, unique, and continuous hourly data")
    values = d[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Real finite nonnegative hourly history is required")
    if not np.equal(values[:, 1], np.floor(values[:, 1])).all():
        raise ValueError("Historical sessions must be observed integer counts")
    return d


def extract_features(hourly):
    d = validate_hourly(hourly)
    for c in TARGETS:
        for lag in LAGS:
            d[f"{c}_lag_{lag}"] = d[c].shift(lag)
        past = d[c].shift(1)
        for window in (6, 24):
            d[f"{c}_mean_{window}"] = past.rolling(window).mean()
            d[f"{c}_std_{window}"] = past.rolling(window).std()
    d["hour"] = d.timestamp.dt.hour
    d["dayofweek"] = d.timestamp.dt.day_of_week
    d["weekend"] = (d.dayofweek >= 5).astype(int)
    d["month"] = d.timestamp.dt.month
    for c, period in [("hour", 24), ("dayofweek", 7)]:
        d[f"sin_{c}"] = np.sin(2 * np.pi * d[c] / period)
        d[f"cos_{c}"] = np.cos(2 * np.pi * d[c] / period)
    return d


def split_masks(timestamps):
    t = pd.to_datetime(timestamps)
    return {"train": np.asarray(t < pd.Timestamp("2021-07-01")),
            "validation": np.asarray((t >= pd.Timestamp("2021-07-01")) & (t < pd.Timestamp("2021-10-01"))),
            "test": np.asarray(t >= pd.Timestamp("2021-10-01"))}


def sequence_arrays(x, y, length=SEQUENCE_LENGTH):
    """Last feature row predicts its own target hour, with history through u-1."""
    if len(x) < length:
        raise ValueError("Insufficient sequence history")
    sequences = np.stack([x[i-length+1:i+1] for i in range(length-1, len(x))])
    return sequences, np.asarray(y)[length-1:]
