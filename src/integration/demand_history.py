"""Versioned real Jiaxing history, with no interpolation or fallback at serving time."""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from team.waseem_demand.demand.features import validate_hourly

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = PROJECT_ROOT / "team/waseem_demand/runtime_data/jiaxing-hourly-demand-v1"


class DemandHistoryError(ValueError):
    """Public-safe validation problem with the requested historical context."""


class DemandUnavailableError(RuntimeError):
    """Internal demand component unavailable; details must not reach clients."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operational_origin(value):
    try:
        if not isinstance(value, (str, datetime)):
            raise ValueError()
        origin = pd.Timestamp(value)
        if pd.isna(origin) or origin.tzinfo is not None or origin != origin.floor("h"):
            raise ValueError()
        return origin
    except (ValueError, TypeError, OverflowError) as exc:
        raise DemandHistoryError("Use a naive whole-hour source operational forecast_origin.") from exc


class HistoryProvider(Protocol):
    def get_history(self, forecast_origin, required_hours=192) -> list[dict]: ...


class LocalJiaxingHistoryProvider:
    def __init__(self, directory=HISTORY_DIR):
        directory = Path(directory)
        try:
            self.metadata = json.loads((directory / "metadata.json").read_text())
            path = directory / "hourly_history.csv"
            if digest(path) != self.metadata["derived_sha256"]:
                raise ValueError("History checksum mismatch")
            raw = pd.read_csv(path, float_precision="round_trip")
            if list(raw.columns) != ["timestamp", "energy_kwh", "sessions"]:
                raise ValueError("Unexpected history columns")
            self._history = validate_hourly(raw)
            if len(raw) != self.metadata["output_rows"] or self.metadata["data_version"] != "jiaxing-hourly-demand-v1":
                raise ValueError("History version/row mismatch")
            if (self._history.timestamp.iloc[0].isoformat() != self.metadata["first_hour"]
                    or self._history.timestamp.iloc[-1].isoformat() != self.metadata["last_hour"]):
                raise ValueError("History coverage mismatch")
        except Exception as exc:
            logger.error("Demand history unavailable (%s)", type(exc).__name__)
            raise DemandUnavailableError("Demand history is unavailable.") from exc
        logger.info("Demand history loaded: version=%s rows=%d", self.metadata["data_version"], len(raw))

    def get_history(self, forecast_origin, required_hours=192):
        origin = operational_origin(forecast_origin)
        if required_hours != 192:
            raise DemandHistoryError("Demand requires exactly 192 history hours.")
        lower = origin - pd.Timedelta(hours=required_hours)
        history = self._history[(self._history.timestamp >= lower) & (self._history.timestamp < origin)].copy()
        expected = pd.date_range(lower, periods=required_hours, freq="h")
        if len(history) != required_hours or not pd.DatetimeIndex(history.timestamp).equals(expected):
            raise DemandHistoryError("Forecast origin is outside supported history coverage: "
                                     "2020-01-08T21:00:00 through 2022-01-01T00:00:00 (source operational time).")
        history["timestamp"] = history.timestamp.map(lambda v: v.isoformat())
        return history.to_dict("records")
