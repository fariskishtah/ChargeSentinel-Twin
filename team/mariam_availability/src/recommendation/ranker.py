"""Deterministic charger ranking for a single charging location.

Each candidate supplies a predicted free time, historical anomaly risk, and
normalized demand and solar signals. The score is lower for better candidates.
The caller supplies normalized values in [0, 1]; raw forecast units are never
silently rescaled here.
"""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any, Mapping


RANKER_VERSION = "recommendation-score-v3"
RECOMMENDATION_SCHEMA_VERSION = "recommendation-output-v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
_WEIGHT_KEYS = ("wait", "anomaly_risk", "demand_pressure", "solar_benefit")
_SIGNAL_KEYS = ("anomaly_risk", "demand_pressure", "solar_benefit")


def _timestamp(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be an ISO-8601 timestamp")
    return parsed


def _unit_interval(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return number


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    weights = config.get("weights")
    if not isinstance(weights, Mapping) or set(weights) != set(_WEIGHT_KEYS):
        raise ValueError(f"weights must define exactly: {', '.join(_WEIGHT_KEYS)}")
    parsed_weights = {}
    for key in _WEIGHT_KEYS:
        try:
            value = float(weights[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"weights.{key} must be a number") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"weights.{key} must be finite and non-negative")
        parsed_weights[key] = value
    if not math.isclose(sum(parsed_weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("weights must sum to 1.0")

    max_wait = float(config.get("max_wait_minutes", 120))
    alternatives_limit = int(config.get("alternatives_limit", 3))
    prefer_free = config.get("prefer_free_if_available", True)
    use_person3_score = config.get("use_person3_score", False)
    mock = config.get("mock_signals", {})
    if not math.isfinite(max_wait) or max_wait <= 0:
        raise ValueError("max_wait_minutes must be finite and positive")
    if alternatives_limit < 0:
        raise ValueError("alternatives_limit must be non-negative")
    if not isinstance(prefer_free, bool):
        raise ValueError("prefer_free_if_available must be boolean")
    if not isinstance(use_person3_score, bool):
        raise ValueError("use_person3_score must be boolean")
    if not isinstance(mock, Mapping):
        raise ValueError("mock_signals must be a mapping")
    mock_signals = {}
    for key in ("demand_pressure", "solar_benefit"):
        if key in mock:
            mock_signals[key] = _unit_interval(mock[key], f"mock_signals.{key}")
    return {
        "version": str(config.get("version", RANKER_VERSION)),
        "weights": parsed_weights,
        "max_wait_minutes": max_wait,
        "alternatives_limit": alternatives_limit,
        "prefer_free_if_available": prefer_free,
        "use_person3_score": use_person3_score,
        "mock_signals": mock_signals,
        "mock_source": str(mock.get("source", "configured_mock")),
    }


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the YAML settings. Install PyYAML from the team requirements."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("config.yaml loading requires PyYAML (see team requirements.txt)") from exc
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError(f"Invalid recommendation config: {path}")
    return _validate_config(config)


def recommend(payload: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the best same-location charger and ranked alternatives.

    Payload fields:

    - ``user_eta``: ISO-8601 local timestamp (preserve the input's timezone).
    - ``candidates``: charger records, all from the driver's location. Each
      record needs ``charger_id``, ``location``, ``expected_free_time`` and
      ``anomaly_risk``. ``demand_pressure`` and ``solar_benefit`` may be
      omitted while upstream forecasts are unavailable; config mock values
      then apply and the output records that they were used.

    Wait is derived from ``max(0, expected_free_time - user_eta)``. Ranking
    score: ``w_wait*wait + w_risk*risk + w_demand*demand - w_solar*solar``.
    Ties resolve by shorter wait, then charger ID.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    if "user_eta" not in payload:
        raise ValueError("payload.user_eta is required")
    eta = _timestamp(payload["user_eta"], "payload.user_eta")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("payload.candidates must be a list")
    settings = _validate_config(config) if config is not None else load_config()

    location = payload.get("location")
    if location is None and candidates:
        locations = {str(item.get("location", "")) for item in candidates if isinstance(item, Mapping)}
        if len(locations) > 1:
            raise ValueError("candidates must be from one location; provide payload.location to filter them")
        location = next(iter(locations), None)
    eligible = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise ValueError("each candidate must be a mapping")
        if location is None or str(item.get("location", "")) == str(location):
            eligible.append(item)

    free_pool_restricted = False
    if eligible and settings["prefer_free_if_available"]:
        free_candidates = []
        for item in eligible:
            if "expected_free_time" not in item:
                raise ValueError(f"candidate {item.get('charger_id', '<unknown>')} requires expected_free_time")
            free_time = _timestamp(item["expected_free_time"],
                                   f"candidate {item.get('charger_id', '<unknown>')}.expected_free_time")
            if (eta.tzinfo is None) != (free_time.tzinfo is None):
                raise ValueError("user_eta and expected_free_time must both include a timezone or both omit it")
            if "busy" in item:
                if not isinstance(item["busy"], bool):
                    raise ValueError(f"candidate {item.get('charger_id', '<unknown>')}.busy must be boolean")
                is_free = not item["busy"]
            else:
                is_free = free_time <= eta
            if is_free:
                free_candidates.append(item)
        if free_candidates:
            eligible = free_candidates
            free_pool_restricted = True

    if not eligible:
        return {
            "recommended_charger": None,
            "location": location,
            "expected_free_time": None,
            "expected_wait_minutes": 0.0,
            "recommended_slot": None,
            "model_version": settings["version"],
            "config_version": settings["version"],
            "schema_version": RECOMMENDATION_SCHEMA_VERSION,
            "risk_source": "historical_abnormal_rate",
            "limitations": [],
            "score": None,
            "score_components": None,
            "mock_signals_used": False,
            "free_pool_restricted": False,
            "reason": ["no_candidates_at_location"],
            "alternatives": [],
        }

    seen_ids: set[str] = set()
    ranked = []
    for candidate in eligible:
        charger_id = str(candidate.get("charger_id", "")).strip()
        if not charger_id:
            raise ValueError("every candidate requires charger_id")
        if charger_id in seen_ids:
            raise ValueError(f"duplicate charger_id at location {location}: {charger_id}")
        seen_ids.add(charger_id)
        if "expected_free_time" not in candidate:
            raise ValueError(f"candidate {charger_id} requires expected_free_time")
        if "anomaly_risk" not in candidate:
            raise ValueError(f"candidate {charger_id} requires historical anomaly_risk")

        free_time = _timestamp(candidate["expected_free_time"], f"{charger_id}.expected_free_time")
        if (eta.tzinfo is None) != (free_time.tzinfo is None):
            raise ValueError("user_eta and expected_free_time must both include a timezone or both omit it")
        wait_minutes = max(0.0, (free_time - eta).total_seconds() / 60.0)
        mock_used = False
        signals = {}
        for key in _SIGNAL_KEYS:
            if key in candidate:
                value = candidate[key]
            elif key in settings["mock_signals"]:
                value = settings["mock_signals"][key]
                if key != "anomaly_risk":
                    mock_used = True
            else:
                raise ValueError(f"candidate {charger_id} requires {key}; no configured mock value exists")
            signals[key] = _unit_interval(value, f"{charger_id}.{key}")

        wait_norm = min(wait_minutes / settings["max_wait_minutes"], 1.0)
        components = {
            "wait": wait_norm,
            "anomaly_risk": signals["anomaly_risk"],
            "demand_pressure": signals["demand_pressure"],
            "solar_benefit": signals["solar_benefit"],
        }
        weights = settings["weights"]
        weighted_terms = {
            "wait": weights["wait"] * components["wait"],
            "anomaly_risk": weights["anomaly_risk"] * components["anomaly_risk"],
            "demand_pressure": weights["demand_pressure"] * components["demand_pressure"],
            "solar_benefit": -weights["solar_benefit"] * components["solar_benefit"],
        }
        score = sum(weighted_terms.values())
        reasons = []
        if free_pool_restricted:
            reasons.append("free_chargers_preferred")
        if wait_minutes == 0:
            reasons.append("available_at_driver_eta")
        elif wait_minutes <= 15:
            reasons.append("short_wait")
        if signals["anomaly_risk"] <= 0.1:
            reasons.append("low_historical_anomaly_risk")
        if signals["demand_pressure"] <= 0.3:
            reasons.append("low_forecast_demand_pressure")
        if signals["solar_benefit"] >= 0.7:
            reasons.append("high_solar_benefit")
        if not reasons:
            reasons.append("best_weighted_score_at_location")

        slot = max(eta, free_time)
        ranked.append({
            "charger_id": charger_id,
            "location": candidate.get("location", location),
            "expected_free_time": free_time.isoformat(),
            "expected_wait_minutes": round(wait_minutes, 2),
            "recommended_slot": slot.isoformat(),
            "score": score,
            "score_components": {key: round(value, 6) for key, value in components.items()},
            "risk_source": candidate.get("risk_source", "historical_abnormal_rate"),
            "_terms": components,
            "_weighted_terms": weighted_terms,
            "mock_signals_used": mock_used,
            "free_pool_restricted": free_pool_restricted,
            "reason": reasons,
            "_wait_sort": wait_minutes,
        })

    ranked.sort(key=lambda row: (round(row["score"], 12), row["_wait_sort"], row["charger_id"]))
    lowest_risk = min(row["_terms"]["anomaly_risk"] for row in ranked)
    highest_solar = max(row["_terms"]["solar_benefit"] for row in ranked)
    for row in ranked:
        weighted_terms = row.pop("_weighted_terms")
        terms = row.pop("_terms")
        dominant_factor = max(
            _WEIGHT_KEYS,
            key=lambda key: (abs(weighted_terms[key]), -_WEIGHT_KEYS.index(key)),
        )
        reasons = []
        if terms["wait"] == 0:
            reasons.append("Free now")
        if terms["anomaly_risk"] == lowest_risk:
            reasons.append("Lowest historical abnormal-end rate at this site")
        if terms["solar_benefit"] == highest_solar:
            reasons.append("Highest solar benefit")
        dominant_reasons = {
            "wait": "Wait has the largest weighted contribution",
            "anomaly_risk": "Historical abnormal-end rate has the largest weighted contribution",
            "demand_pressure": "Demand pressure has the largest weighted contribution",
            "solar_benefit": "Solar benefit has the largest weighted contribution",
        }
        reasons.append(dominant_reasons[dominant_factor])
        score = sum(weighted_terms.values())
        assert math.isclose(score, row["score"], rel_tol=0.0, abs_tol=1e-9)
        assert math.isclose(sum(weighted_terms.values()), score, rel_tol=0.0, abs_tol=1e-9)
        row["explanation"] = {
            "terms": terms,
            "weights": dict(weights),
            "weighted_terms": weighted_terms,
            "score": score,
            "dominant_factor": dominant_factor,
            "reason": "; ".join(reasons) + ".",
        }
        row.pop("_wait_sort")
    winner = ranked[0]
    result = {
        "recommended_charger": winner["charger_id"],
        "location": winner["location"],
        "expected_free_time": winner["expected_free_time"],
        "expected_wait_minutes": winner["expected_wait_minutes"],
        "recommended_slot": winner["recommended_slot"],
        "model_version": settings["version"],
        "config_version": settings["version"],
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "risk_source": winner["risk_source"],
        "limitations": [],
        "score": winner["score"],
        "score_components": winner["score_components"],
        "explanation": winner["explanation"],
        "mock_signals_used": winner["mock_signals_used"],
        "free_pool_restricted": free_pool_restricted,
        "reason": winner["reason"],
        "alternatives": ranked[1:1 + settings["alternatives_limit"]],
    }
    if len(ranked) > 1:
        second = ranked[1]
        term_differences = {
            key: round(
                second["explanation"]["weighted_terms"][key]
                - winner["explanation"]["weighted_terms"][key],
                12,
            )
            for key in _WEIGHT_KEYS
        }
        largest_difference = max(abs(value) for value in term_differences.values())
        decisive_terms = [
            key for key in _WEIGHT_KEYS
            if math.isclose(abs(term_differences[key]), largest_difference, abs_tol=1e-12)
        ]
        decisive_term = decisive_terms[0]
        if largest_difference <= 1e-12:
            comparison_reason = "Their weighted terms are tied; the ranking tie-break selected the top charger."
        elif len(decisive_terms) > 1:
            labels = {"anomaly_risk": "historical abnormal-end rate"}
            factors = [labels.get(key, key.replace("_", " ")) for key in decisive_terms]
            comparison_reason = (
                f"{' and '.join(factors)} contributed equally to the difference versus the second charger."
            )
        else:
            comparison_reason = {
                "wait": "Wait made the largest difference versus the second charger.",
                "anomaly_risk": "Historical abnormal-end rate made the largest difference versus the second charger.",
                "demand_pressure": "Demand pressure made the largest difference versus the second charger.",
                "solar_benefit": "Solar benefit made the largest difference versus the second charger.",
            }[decisive_term]
        result["comparison"] = {
            "second_charger_id": second["charger_id"],
            "decisive_term": decisive_term,
            "reason": comparison_reason,
        }
    return result
