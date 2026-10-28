"""Connect occupancy state, duration inference, and deterministic ranking."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..availability.inference import AvailabilityPredictor, DEFAULT_BUNDLE_PATH
from .ranker import recommend as rank_candidates


def _time(value: Any, field: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if pd.isna(result):
        raise ValueError(f"{field} must be a valid timestamp")
    return result


def recommend(
    payload: Mapping[str, Any],
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    predictor: AvailabilityPredictor | None = None,
) -> dict[str, Any]:
    """Recommend a same-location charger from ETA and live/simulated occupancy.

    Each candidate has a boolean ``busy`` state. Busy candidates also provide
    ``session_start`` and safe feature context; free candidates are available
    at the driver's ETA. Historical risk comes from ``anomaly_risk`` or
    ``history_features.hist_abnormal_rate``. Demand and solar may be omitted,
    in which case the ranker's configured placeholders are reported.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    for field in ("location", "user_eta", "candidates"):
        if field not in payload:
            raise ValueError(f"payload.{field} is required")
    if not isinstance(payload["candidates"], list):
        raise ValueError("payload.candidates must be a list")
    eta = _time(payload["user_eta"], "payload.user_eta")
    location = str(payload["location"])
    ranked_input = []

    for index, candidate in enumerate(payload["candidates"]):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"payload.candidates[{index}] must be a mapping")
        charger_id = str(candidate.get("charger_id", "")).strip()
        if not charger_id:
            raise ValueError(f"payload.candidates[{index}].charger_id is required")
        candidate_location = str(candidate.get("location", location))
        if candidate_location != location:
            continue
        if not isinstance(candidate.get("busy"), bool):
            raise ValueError(f"candidate {charger_id}.busy must be boolean")
        history = candidate.get("history_features", {})
        if not isinstance(history, Mapping):
            raise ValueError(f"candidate {charger_id}.history_features must be a mapping")

        risk = candidate.get("anomaly_risk", history.get("hist_abnormal_rate"))
        if risk is None:
            raise ValueError(f"candidate {charger_id} needs training-history anomaly_risk")

        if candidate["busy"]:
            if predictor is None:
                predictor = AvailabilityPredictor(bundle_path)
            if "session_start" not in candidate:
                raise ValueError(f"busy candidate {charger_id} requires session_start")
            session_start = _time(candidate["session_start"], f"candidate {charger_id}.session_start")
            if (session_start.tz is None) != (eta.tz is None):
                raise ValueError("user_eta and candidate session_start must use consistent timezone awareness")
            if session_start > eta:
                raise ValueError(f"busy candidate {charger_id} has a session_start after user_eta")
            prediction_payload = {
                "charger_id": candidate["charger_id"],
                "location": candidate_location,
                "district": candidate.get("district", payload.get("district", "unknown")),
                "session_start": session_start,
                "as_of": eta,
                "history_features": dict(history),
            }
            for field in ("temp_c", "humidity", "precip_mm"):
                if field in candidate:
                    prediction_payload[field] = candidate[field]
            duration_prediction = predictor.predict(prediction_payload)
            expected_free_time = duration_prediction["expected_free_time"]
        else:
            expected_free_time = eta.isoformat()

        row = {
            "charger_id": charger_id,
            "location": candidate_location,
            "busy": candidate["busy"],
            "expected_free_time": expected_free_time,
            "anomaly_risk": risk,
        }
        for field in ("demand_pressure", "solar_benefit"):
            if field in candidate:
                row[field] = candidate[field]
        ranked_input.append(row)

    result = rank_candidates({
        "location": location,
        "user_eta": eta.isoformat(),
        "candidates": ranked_input,
    })
    return result
