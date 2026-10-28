"""Connect occupancy state, duration inference, and charger ranking."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..availability.inference import (
    AvailabilityPredictor, DEFAULT_BUNDLE_PATH, get_bundle_version,
)
from .ranker import load_config, recommend as rank_candidates
from .risk import DEFAULT_RISK_TABLE_PATH, charger_risk, load_risk_table


RECOMMENDATION_LIMITATIONS = [
    "Duration is predicted from the session start and is not adjusted for time already elapsed.",
    "Demand and solar inputs are placeholders until Persons 1 and 2 deliver.",
    "Anomaly risk is a historical abnormal rate, not a model prediction.",
    "Labels are operator recorded, not verified hardware diagnoses.",
]


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
    risk_table_path: str | Path = DEFAULT_RISK_TABLE_PATH,
) -> dict[str, Any]:
    """Recommend a same-location charger using historical risk by default.

    Person 3 scores are ignored unless ``use_person3_score`` is explicitly
    enabled in config.yaml. Missing or unseen charger IDs use the global
    abnormal rate from the training-period risk table.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    for field in ("location", "user_eta", "candidates"):
        if field not in payload:
            raise ValueError(f"payload.{field} is required")
    if not isinstance(payload["candidates"], list):
        raise ValueError("payload.candidates must be a list")
    if not payload["candidates"]:
        raise ValueError("payload.candidates must not be empty")

    eta = _time(payload["user_eta"], "payload.user_eta")
    location = str(payload["location"])
    settings = load_config()
    risk_table = load_risk_table(risk_table_path)
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

        risk = charger_risk(charger_id, risk_table)
        risk_source = "historical_abnormal_rate"
        if settings["use_person3_score"]:
            anomaly_prediction = candidate.get("anomaly_prediction")
            if isinstance(anomaly_prediction, Mapping) and anomaly_prediction.get("score") is not None:
                risk = anomaly_prediction["score"]
                risk_source = "person3_score"
            elif candidate.get("anomaly_risk") is not None:
                risk = candidate["anomaly_risk"]
            elif history.get("hist_abnormal_rate") is not None:
                risk = history["hist_abnormal_rate"]
        try:
            risk = float(risk)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate {charger_id}.anomaly risk must be numeric") from exc
        if not 0.0 <= risk <= 1.0:
            raise ValueError(f"candidate {charger_id}.anomaly risk must be between 0 and 1")

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
            "risk_source": risk_source,
        }
        for field in ("demand_pressure", "solar_benefit"):
            if field in candidate:
                row[field] = candidate[field]
        ranked_input.append(row)

    result = rank_candidates({
        "location": location,
        "user_eta": eta.isoformat(),
        "candidates": ranked_input,
    }, config=settings)
    result["bundle_version"] = (
        getattr(predictor, "bundle_version", getattr(predictor, "model_version", "unavailable"))
        if predictor is not None
        else get_bundle_version(bundle_path)
    )
    result["config_version"] = settings["version"]
    result["limitations"] = list(RECOMMENDATION_LIMITATIONS)
    return result
