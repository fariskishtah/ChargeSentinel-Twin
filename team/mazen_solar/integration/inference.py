"""Portable, origin-causal 24h/48h PVGIS reference forecasting."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from team.mazen_solar.integration.source_features import SolarFeaturePipeline, calculate_solar_geometry

BUNDLE = Path(__file__).resolve().parents[1] / "bundle" / "solar-v1"


class SolarInputError(ValueError):
    pass


@lru_cache(maxsize=4)
def load_bundle(horizon: int, root: str = str(BUNDLE)):
    if horizon not in (24, 48):
        raise SolarInputError("Solar horizon must be 24 or 48 hours.")
    folder = Path(root)
    checksums = json.loads((folder / "checksums.json").read_text())
    for name, expected in checksums.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Solar bundle checksum mismatch.")
    artifact = json.loads((folder / f"solar_{horizon}h.json").read_text())
    if artifact["horizon_hours"] != horizon or artifact["capacity_kw"] != 100.0:
        raise ValueError("Solar horizon artifact mismatch.")
    history = pd.read_csv(folder / artifact["history_file"], parse_dates=["timestamp"],
                          float_precision="round_trip").set_index("timestamp")["power_kw"]
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("Solar history timestamp integrity failure.")
    if len(history) != 4368 or history.index[0] != pd.Timestamp("2024-01-01") or history.index[-1] != pd.Timestamp("2024-06-30 23:00"):
        raise ValueError("Solar history coverage mismatch.")
    weather = None
    if artifact["weather_required"]:
        weather = pd.read_csv(folder / artifact["weather_file"], parse_dates=["timestamp"],
                              float_precision="round_trip").set_index("timestamp")
        if not weather.index.equals(history.index):
            raise ValueError("Solar weather alignment mismatch.")
    return artifact, history, weather


def predict_solar(payload: dict[str, object], root: str = str(BUNDLE)) -> dict[str, object]:
    horizon = int(payload["horizon_hours"])
    origin = pd.Timestamp(payload["forecast_origin"])
    if origin.tzinfo is None or origin.utcoffset().total_seconds() != 0 or origin.minute or origin.second or origin.microsecond:
        raise SolarInputError("Use a whole-hour UTC solar forecast_origin.")
    origin = origin.tz_convert("UTC").tz_localize(None)
    artifact, history, weather = load_bundle(horizon, root)
    target = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
    if origin < pd.Timestamp("2024-01-22") or target[-1] > history.index[-1]:
        raise SolarInputError("Solar origin outside supported 2024 H1 coverage.")
    if artifact["family"] == "Seasonal Persistence":
        lags = [t - pd.Timedelta(hours=24 if i < 24 else (48 if i < 48 else 72)) for i, t in enumerate(target, 1)]
        if any(t >= origin or t not in history.index for t in lags):
            raise ValueError("Solar history lag unavailable.")
        values = history.loc[lags].to_numpy(dtype=float)
    elif artifact["family"] == "Ridge Regression":
        hist = history.loc[origin - pd.Timedelta(hours=48):origin - pd.Timedelta(hours=1)]
        if len(hist) != 48 or hist.isna().any():
            raise SolarInputError("Solar origin lacks 48 complete observed history hours.")
        if weather is None or not target.isin(weather.index).all() or weather.loc[target].isna().any().any():
            raise SolarInputError("Required origin-safe weather runs are unavailable.")
        frame = SolarFeaturePipeline().build_forecast_features(origin, horizon, weather.loc[target], hist)
        if list(frame.columns) != artifact["features"] or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Solar model feature contract mismatch.")
        params = artifact["ridge"]
        values = ((frame.to_numpy(dtype=float) - np.array(params["mean"])) / np.array(params["scale"])) @ np.array(params["coef"]) + params["intercept"]
    else:
        raise ValueError("Unsupported solar champion family.")
    if not np.isfinite(values).all():
        raise SolarInputError("Required solar history is incomplete.")
    daylight = calculate_solar_geometry(target)["is_daylight"].to_numpy() > 0
    values = np.where(daylight, np.clip(values, 0, artifact["capacity_kw"]), 0.0)
    points = [{"timestamp": ts.tz_localize("UTC").isoformat(), "forecast_kw":float(kw)}
              for ts, kw in zip(target, values)]
    return {"timestamp":points[0]["timestamp"], "horizon_hours":horizon,
            "forecast_kw":points[0]["forecast_kw"], "points":points,
            "model_family":artifact["family"], "model_version":artifact["model_version"]}
