"""Causal, reproducible interval anomaly rule for historical Jiaxing data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "team/waseem_demand/runtime_data/jiaxing-hourly-demand-v1/hourly_history.csv"
CONFIG = Path(__file__).resolve().parent / "grid-load-v1.json"
VERSION = "grid-load-seasonal-rule-v1"


class GridLoadUnavailableError(RuntimeError):
    """Missing or corrupt portable history/configuration."""


@lru_cache(maxsize=1)
def load_history() -> tuple[dict, list[dict]]:
    try:
        config = json.loads(CONFIG.read_text())
        digest = hashlib.sha256(HISTORY.read_bytes()).hexdigest()
        if digest != config["history_sha256"]:
            raise ValueError("Grid-load history checksum mismatch")
        with HISTORY.open(newline="") as handle:
            rows = [
                {"timestamp": datetime.fromisoformat(row["timestamp"]),
                 "energy_kwh": float(row["energy_kwh"]),
                 "sessions": float(row["sessions"])}
                for row in csv.DictReader(handle)
            ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GridLoadUnavailableError("Grid-load assets unavailable") from exc
    if not rows or any(not math.isfinite(row[key]) or row[key] < 0
                       for row in rows for key in ("energy_kwh", "sessions")):
        raise GridLoadUnavailableError("Grid-load history is invalid")
    if any((b["timestamp"] - a["timestamp"]).total_seconds() != 3600
           for a, b in zip(rows, rows[1:])):
        raise GridLoadUnavailableError("Grid-load history must be hourly and continuous")
    return config, rows


def interval_features(rows: list[dict], index: int, *, energy: float | None = None,
                      sessions: float | None = None) -> dict:
    """Use the current observed hour and eight strictly earlier same-week hours."""
    if index < 8 * 168 or index >= len(rows):
        raise ValueError("Interval lacks eight earlier weekly peers")
    peers = [rows[index - week * 168] for week in range(1, 9)]
    expected_energy = median(peer["energy_kwh"] for peer in peers)
    expected_sessions = median(peer["sessions"] for peer in peers)
    energy_mad = median(abs(peer["energy_kwh"] - expected_energy) for peer in peers)
    sessions_mad = median(abs(peer["sessions"] - expected_sessions) for peer in peers)
    energy_scale = max(1.4826 * energy_mad, 0.15 * expected_energy, 25.0)
    sessions_scale = max(1.4826 * sessions_mad, 0.15 * expected_sessions, 2.0)
    actual_energy = rows[index]["energy_kwh"] if energy is None else energy
    actual_sessions = rows[index]["sessions"] if sessions is None else sessions
    return {
        "timestamp": rows[index]["timestamp"].isoformat(),
        "energy_kwh": actual_energy, "sessions": actual_sessions,
        "expected_energy_kwh": expected_energy,
        "expected_sessions": expected_sessions,
        "energy_deviation": (actual_energy - expected_energy) / energy_scale,
        "sessions_deviation": (actual_sessions - expected_sessions) / sessions_scale,
    }


def score(features: dict) -> float:
    return max(abs(features["energy_deviation"]), abs(features["sessions_deviation"]))


def detect(timestamp: datetime) -> dict:
    if timestamp.tzinfo is not None or timestamp.minute or timestamp.second or timestamp.microsecond:
        raise ValueError("Timestamp must be a whole hour in source operational time")
    config, rows = load_history()
    first = rows[0]["timestamp"]
    index = int((timestamp - first).total_seconds() // 3600)
    if index < 8 * 168 or index >= len(rows) or rows[index]["timestamp"] != timestamp:
        raise ValueError("Unsupported historical interval")
    features = interval_features(rows, index)
    value = score(features)
    threshold = float(config["threshold"])
    reasons = []
    if abs(features["energy_deviation"]) >= threshold:
        direction = "above" if features["energy_deviation"] > 0 else "below"
        reasons.append(f"Completed-session energy substantially {direction} earlier same-week hours")
    if abs(features["sessions_deviation"]) >= threshold:
        reasons.append("Unusual session-count deviation from earlier same-week hours")
    if not reasons:
        reasons.append("Within the calibrated seasonal deviation range")
    return {"timestamp": features["timestamp"], "anomaly": value >= threshold,
            "score": value, "threshold": threshold, "reason": reasons,
            "detector_version": VERSION, "energy_kwh": features["energy_kwh"],
            "sessions": int(features["sessions"]), "is_simulated": False}
