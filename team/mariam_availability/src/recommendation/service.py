"""Connect occupancy state, duration inference, and charger ranking."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..availability.inference import (
    AvailabilityPredictor, DEFAULT_BUNDLE_PATH, get_bundle_version,
)
from .ranker import load_config, recommend as rank_candidates
from .risk import DEFAULT_RISK_TABLE_PATH, charger_risk, load_risk_table

REPO_ROOT = Path(__file__).resolve().parents[4]
DEMAND_HISTORY_PATH = (
    REPO_ROOT / "team" / "waseem_demand" / "runtime_data"
    / "jiaxing-hourly-demand-v1" / "hourly_history.csv"
)
DEMAND_METADATA_PATH = DEMAND_HISTORY_PATH.with_name("metadata.json")


RECOMMENDATION_LIMITATIONS = [
    "Duration is predicted from the session start and is not adjusted for time already elapsed.",
    "Candidate occupancy and station state are caller supplied or historical replay, not live telemetry.",
    "Person 2 expected-session forecasts are normalized using the configured saturation count; the bundled Jiaxing history covers 2020-2021 only.",
    "Person 1 solar output is a modelled New Cairo reference forecast and is simulated for the Jiaxing charger site, not measured local solar.",
    "Anomaly risk is a training-period abnormal-end rate, not a fault probability.",
    "Abnormal-end labels are operator recorded, not verified hardware diagnoses.",
]


@lru_cache(maxsize=1)
def _demand_history() -> pd.DataFrame:
    if not DEMAND_HISTORY_PATH.is_file():
        raise FileNotFoundError(f"Person 2 demand history not found: {DEMAND_HISTORY_PATH}")
    frame = pd.read_csv(DEMAND_HISTORY_PATH, parse_dates=["timestamp"])
    required = {"timestamp", "energy_kwh", "sessions"}
    if set(frame.columns) != required:
        raise ValueError("Person 2 demand history does not match its hourly schema")
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Person 2 demand history timestamps are duplicated or unordered")
    return frame


def _demand_pressure(
    eta: pd.Timestamp, payload: Mapping[str, Any], saturation_sessions: float,
) -> tuple[float | None, str]:
    origin = eta.floor("h")
    if origin.tz is not None:
        return None, "configured_placeholder_timezone_unresolved"

    supplied = payload.get("demand_forecast")
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError("payload.demand_forecast must be a Person 2 forecast object")
        forecast_time = _time(supplied.get("timestamp"), "payload.demand_forecast.timestamp")
        if forecast_time.tz is not None or forecast_time != origin:
            raise ValueError("Person 2 demand_forecast.timestamp must match the naive arrival hour")
        expected_sessions = supplied.get("expected_sessions")
        version = str(supplied.get("model_version", "person2_demand"))
        source = f"person2_expected_sessions:{version}"
    else:
        if not DEMAND_METADATA_PATH.is_file():
            return None, "configured_placeholder_demand_bundle_unavailable"
        metadata = json.loads(DEMAND_METADATA_PATH.read_text(encoding="utf-8"))
        earliest = pd.Timestamp(metadata["earliest_forecast_origin"])
        latest = pd.Timestamp(metadata["latest_forecast_origin"])
        if origin < earliest or origin > latest:
            return None, "configured_placeholder_outside_person2_history_coverage"
        history = _demand_history()
        past = history.loc[history["timestamp"] < origin].tail(192).copy()
        if len(past) < 192:
            return None, "configured_placeholder_insufficient_person2_history"
        records = past.assign(
            timestamp=past["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        ).to_dict("records")
        from team.waseem_demand.demand import predict_demand

        forecast = predict_demand({
            "forecast_origin": origin.strftime("%Y-%m-%dT%H:%M:%S"),
            "history": records,
        })
        expected_sessions = forecast["expected_sessions"]
        version = str(forecast["model_version"])
        source = f"person2_expected_sessions:{version}"

    try:
        expected_sessions = float(expected_sessions)
    except (TypeError, ValueError) as exc:
        raise ValueError("Person 2 expected_sessions must be numeric") from exc
    if not math.isfinite(expected_sessions) or expected_sessions < 0:
        raise ValueError("Person 2 expected_sessions must be finite and non-negative")
    return min(expected_sessions / saturation_sessions, 1.0), source


def _solar_benefit(
    payload: Mapping[str, Any], capacity_kw: float, reference_location: str,
) -> tuple[float | None, str]:
    supplied = payload.get("solar_forecast")
    if supplied is not None:
        if not isinstance(supplied, Mapping):
            raise ValueError("payload.solar_forecast must be a Person 1 forecast object")
        points = supplied.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError("Person 1 solar_forecast must contain forecast points")
        version = str(supplied.get("model_version", "person1_solar"))
    elif payload.get("solar_forecast_origin") is not None:
        from team.mazen_solar.integration.inference import predict_solar

        origin = payload["solar_forecast_origin"]
        horizon = int(payload.get("solar_horizon_hours", 24))
        forecast = predict_solar({"forecast_origin": origin, "horizon_hours": horizon})
        points = forecast["points"]
        version = str(forecast["model_version"])
    else:
        return None, "configured_placeholder_solar_forecast_not_supplied"

    target = payload.get("solar_target_timestamp")
    if target is None:
        point = points[0]
    else:
        target_time = _time(target, "payload.solar_target_timestamp")
        if target_time.tz is None or target_time.utcoffset().total_seconds() != 0:
            raise ValueError("solar_target_timestamp must be an explicit UTC timestamp")
        point = next((p for p in points if _time(p.get("timestamp"), "solar point timestamp") == target_time), None)
        if point is None:
            raise ValueError("solar_target_timestamp is not present in Person 1 forecast points")

    try:
        forecast_kw = float(point["forecast_kw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Person 1 solar point must include numeric forecast_kw") from exc
    if not math.isfinite(forecast_kw) or not 0 <= forecast_kw <= capacity_kw:
        raise ValueError(f"Person 1 forecast_kw must be finite and within [0, {capacity_kw:g}]")
    target_stamp = point.get("timestamp", "first forecast point")
    return forecast_kw / capacity_kw, (
        f"person1_modelled_reference_site_simulation:{reference_location}:{version}:{target_stamp}"
    )


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
    """Recommend a same-location charger using integrated demand and solar signals.

    Person 2 demand is forecast from supplied output or the packaged Jiaxing
    hourly history when covered. Person 1 solar may be supplied as a forecast
    or generated from an explicit supported UTC origin. Missing signals use
    configured neutral values and are disclosed in the response. Charger fault
    risk remains the training-period historical abnormal rate.
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
    matching_candidates = [
        candidate for candidate in payload["candidates"]
        if isinstance(candidate, Mapping)
        and str(candidate.get("location", location)) == location
    ]
    need_demand = any("demand_pressure" not in candidate for candidate in matching_candidates)
    need_solar = any("solar_benefit" not in candidate for candidate in matching_candidates)
    demand_value, demand_source = (None, "candidate_supplied")
    if need_demand:
        demand_value, demand_source = _demand_pressure(
            eta, payload, settings["demand_pressure_saturation_sessions"]
        )
    solar_value, solar_source = (None, "candidate_supplied")
    if need_solar:
        solar_value, solar_source = _solar_benefit(
            payload,
            settings["solar_reference_capacity_kw"],
            settings["solar_reference_location"],
        )
    ranked_input = []
    ranked_source_indices = []
    service_excluded = []

    for index, candidate in enumerate(payload["candidates"]):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"payload.candidates[{index}] must be a mapping")
        charger_id = str(candidate.get("charger_id", "")).strip()
        if not charger_id:
            raise ValueError(f"payload.candidates[{index}].charger_id is required")
        candidate_location = str(candidate.get("location", location))
        if candidate_location != location:
            service_excluded.append({
                "candidate_index": index,
                "charger_id": charger_id,
                "location": candidate_location,
                "eligible": False,
                "reason": "Excluded: different location",
            })
            continue
        if not isinstance(candidate.get("busy"), bool):
            raise ValueError(f"candidate {charger_id}.busy must be boolean")
        status = candidate.get("status")
        if status is not None:
            if not isinstance(status, str):
                raise ValueError(f"candidate {charger_id}.status must be a string")
            status = status.strip().upper()
            if status in {"FAULT", "OFFLINE", "UNAVAILABLE", "RESERVED"}:
                service_excluded.append({
                    "candidate_index": index,
                    "charger_id": charger_id,
                    "location": candidate_location,
                    "eligible": False,
                    "reason": f"Excluded: status is {status.lower()}",
                })
                continue
            if status not in {"AVAILABLE", "CHARGING"}:
                raise ValueError(f"candidate {charger_id}.status is unsupported")
            if (status == "AVAILABLE") == candidate["busy"]:
                raise ValueError(f"candidate {charger_id}.status conflicts with busy")
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
        if "demand_pressure" not in row and demand_value is not None:
            row["demand_pressure"] = demand_value
        if "solar_benefit" not in row and solar_value is not None:
            row["solar_benefit"] = solar_value
        ranked_input.append(row)
        ranked_source_indices.append(index)

    result = rank_candidates({
        "location": location,
        "user_eta": eta.isoformat(),
        "candidates": ranked_input,
    }, config=settings)
    for eligibility_row in result.get("candidate_eligibility", []):
        local_index = eligibility_row["candidate_index"]
        eligibility_row["candidate_index"] = ranked_source_indices[local_index]
    result["bundle_version"] = (
        getattr(predictor, "bundle_version", getattr(predictor, "model_version", "unavailable"))
        if predictor is not None
        else get_bundle_version(bundle_path)
    )
    result["config_version"] = settings["version"]
    result["candidate_eligibility"] = service_excluded + result.get("candidate_eligibility", [])
    if result.get("recommended_charger") is None and service_excluded:
        result["decision_explanation"]["summary"] = (
            "No candidates remained after location and operational-status filtering."
        )
    elif service_excluded:
        prior_summary = result["decision_explanation"]["summary"]
        result["decision_explanation"]["summary"] = (
            f"{len(service_excluded)} candidate(s) were excluded by location or operational status. "
            f"{prior_summary}"
        )
    demand_supplied = ["demand_pressure" in candidate for candidate in matching_candidates]
    solar_supplied = ["solar_benefit" in candidate for candidate in matching_candidates]
    if demand_supplied and all(demand_supplied):
        demand_source = "candidate_supplied"
    elif any(demand_supplied):
        demand_source = f"candidate_values_and_{demand_source}"
    if solar_supplied and all(solar_supplied):
        solar_source = "candidate_supplied"
    elif any(solar_supplied):
        solar_source = f"candidate_values_and_{solar_source}"
    result["signal_sources"] = {
        "demand_pressure": demand_source,
        "solar_benefit": solar_source,
        "anomaly_risk": "training_period_historical_abnormal_end_rate",
    }
    result["limitations"] = list(RECOMMENDATION_LIMITATIONS)
    if "configured_placeholder" in demand_source:
        reason = demand_source.split("configured_placeholder_", 1)[-1].replace("_", " ")
        result["limitations"].append(
            f"Demand pressure used the configured neutral placeholder because {reason}."
        )
    if "configured_placeholder" in solar_source:
        result["limitations"].append(
            "Solar benefit used the configured neutral placeholder; pass Person 1's supported forecast to use its simulated New Cairo signal."
        )
    return result
