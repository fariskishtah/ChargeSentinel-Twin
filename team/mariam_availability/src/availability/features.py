"""Leakage-safe duration features and time-based splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2021-07-01")
VAL_END = pd.Timestamp("2021-10-01")
HIST_WINDOW = 20

# These columns are known only after a session has started or ended and must not
# be passed to an inference model.
FORBIDDEN = {
    "end", "duration_min", "target_min", "Payment time", "Order creation time",
    "Transaction power/kwh", "energy_kwh", "Electricity cost/Yuan", "Service charge/Yuan",
    "Transaction Amount/Yuan", "Actual Payment/Yuan", "end_cause", "is_abnormal",
    "is_full_charge", "is_user_stop", "is_nonsystem_fault", "is_system_issue",
    "is_EV_fault", "is_other_fault",
}
CATEGORICAL = ["charger_id", "location", "district"]
NUMERIC = [
    "hour", "dow", "is_weekend", "hour_sin", "hour_cos", "temp_c", "humidity", "precip_mm",
    "idle_before_min", "prev_duration_min", "hist_median_normal", "hist_mean_normal",
    "hist_abnormal_rate", "sessions_before", "has_history", "site_n_chargers",
    "site_busy_others", "site_busy_share", "site_starts_last_60min",
]
REQUIRED = ["charger_id", "start", "end", "location", "district", "is_abnormal", "duration_min"]


def build_features(sessions: pd.DataFrame) -> pd.DataFrame:
    """Return session metadata plus features available at the session start."""
    missing = [column for column in REQUIRED if column not in sessions]
    if missing:
        raise ValueError(f"build_features needs columns {missing}")
    d = sessions.sort_values(["charger_id", "start"]).reset_index(drop=True).copy()
    d["start"] = pd.to_datetime(d["start"], errors="coerce")
    d["end"] = pd.to_datetime(d["end"], errors="coerce")
    if d[["charger_id", "start", "end", "location", "district"]].isna().any().any():
        raise ValueError("Feature inputs contain missing charger, timestamp, location, or district values")

    f = d[["charger_id", "location", "district"]].copy()
    f["start"] = d["start"]
    f["target_min"] = pd.to_numeric(d["duration_min"], errors="coerce")
    f["is_abnormal"] = pd.to_numeric(d["is_abnormal"], errors="coerce").fillna(0).astype(int)

    h = d["start"].dt.hour + d["start"].dt.minute / 60
    f["hour"] = d["start"].dt.hour
    f["dow"] = d["start"].dt.dayofweek
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)

    for new, candidates in {
        "temp_c": ("temp_c", "temperature"),
        "humidity": ("humidity", "relative humidity"),
        "precip_mm": ("precip_mm", "precipitation"),
    }.items():
        old = next((c for c in d.columns if str(c).lower() in candidates or
                    any(str(c).lower().startswith(prefix) for prefix in candidates[1:])), None)
        if old is not None:
            f[new] = pd.to_numeric(d[old], errors="coerce")

    cid = d["charger_id"]
    groups = d.groupby("charger_id", sort=False)
    prev_end = groups["end"].shift()
    f["idle_before_min"] = (d["start"] - prev_end).dt.total_seconds() / 60
    f["prev_duration_min"] = groups["duration_min"].shift()
    normal = pd.to_numeric(d["duration_min"], errors="coerce").where(d["is_abnormal"].eq(0))
    shifted_normal = normal.groupby(cid, sort=False).shift()
    f["hist_median_normal"] = shifted_normal.groupby(cid, sort=False).transform(
        lambda x: x.rolling(HIST_WINDOW, min_periods=3).median()
    )
    f["hist_mean_normal"] = shifted_normal.groupby(cid, sort=False).transform(
        lambda x: x.rolling(HIST_WINDOW, min_periods=3).mean()
    )
    shifted_abnormal = f["is_abnormal"].astype(float).groupby(cid, sort=False).shift()
    f["hist_abnormal_rate"] = shifted_abnormal.groupby(cid, sort=False).transform(
        lambda x: x.rolling(HIST_WINDOW, min_periods=3).mean()
    )
    f["sessions_before"] = groups.cumcount()
    f["has_history"] = (f["sessions_before"] >= 3).astype(int)

    # At time t, count starts strictly before t and ends at or before t. This
    # excludes the session being predicted and any future session outcomes.
    busy = np.zeros(len(d), dtype=float)
    recent = np.zeros(len(d), dtype=float)
    n_chargers = np.zeros(len(d), dtype=float)
    for _, indices in d.groupby("location", sort=False).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        site = d.iloc[idx]
        starts = np.sort(site["start"].to_numpy())
        ends = np.sort(site["end"].to_numpy())
        times = site["start"].to_numpy()
        started = np.searchsorted(starts, times, side="left")
        ended = np.searchsorted(ends, times, side="right")
        busy[idx] = started - ended
        recent[idx] = started - np.searchsorted(starts, times - np.timedelta64(60, "m"), side="left")
        n_chargers[idx] = site["charger_id"].nunique()
    f["site_n_chargers"] = n_chargers
    f["site_busy_others"] = busy
    f["site_busy_share"] = np.where(n_chargers > 1, busy / np.maximum(n_chargers - 1, 1), 0).clip(0, 1)
    f["site_starts_last_60min"] = recent
    return f


def split_by_time(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "start" not in features:
        raise ValueError("split_by_time needs a start column")
    start = pd.to_datetime(features["start"], errors="coerce")
    train = features.loc[start < TRAIN_END].copy()
    val = features.loc[(start >= TRAIN_END) & (start < VAL_END)].copy()
    test = features.loc[start >= VAL_END].copy()
    return train, val, test


class FeatureEncoder:
    """Fit category maps and numeric medians on training data only."""

    def __init__(self):
        self.maps: dict[str, dict] = {}
        self.fill: dict[str, float] = {}
        self.numeric: list[str] = []
        self.feature_names: list[str] = []

    def fit(self, train: pd.DataFrame) -> "FeatureEncoder":
        missing = [c for c in CATEGORICAL if c not in train]
        if missing:
            raise ValueError(f"Encoder training data is missing categorical columns {missing}")
        self.maps = {c: {v: i for i, v in enumerate(sorted(train[c].dropna().unique()))} for c in CATEGORICAL}
        self.numeric = [c for c in NUMERIC if c in train]
        self.fill = {c: float(pd.to_numeric(train[c], errors="coerce").median()) for c in self.numeric}
        if any(not np.isfinite(value) for value in self.fill.values()):
            raise ValueError("Cannot fit encoder: at least one numeric training feature has no finite values")
        self.feature_names = [f"{c}_code" for c in CATEGORICAL] + self.numeric
        forbidden = set(self.feature_names) & FORBIDDEN
        if forbidden:
            raise AssertionError(f"Forbidden inference features: {sorted(forbidden)}")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names:
            raise RuntimeError("Call fit before transform")
        x = pd.DataFrame(index=frame.index)
        for c in CATEGORICAL:
            if c not in frame:
                raise ValueError(f"Encoder input is missing categorical column {c!r}")
            x[f"{c}_code"] = frame[c].map(self.maps[c]).fillna(-1).astype(int)
        for c in self.numeric:
            values = pd.to_numeric(frame[c], errors="coerce") if c in frame else pd.Series(np.nan, index=frame.index)
            x[c] = values.fillna(self.fill[c]).astype(float)
        return x[self.feature_names]

    def get_state(self) -> dict:
        return {"maps": self.maps, "fill": self.fill, "numeric": self.numeric,
                "feature_names": self.feature_names}

    @classmethod
    def from_state(cls, state: dict) -> "FeatureEncoder":
        encoder = cls()
        encoder.maps = state["maps"]
        encoder.fill = state["fill"]
        encoder.numeric = state["numeric"]
        encoder.feature_names = state["feature_names"]
        return encoder
