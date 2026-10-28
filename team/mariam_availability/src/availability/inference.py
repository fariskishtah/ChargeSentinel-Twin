"""Notebook-independent charger session-duration inference."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FORBIDDEN, NUMERIC, FeatureEncoder


TEAM_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_PATH = TEAM_DIR / "artifacts" / "availability_duration_bundle_optuna_mlflow.joblib"
AVAILABILITY_SCHEMA_VERSION = "availability-output-v1"
_LIMITATIONS = [
    "Duration is predicted from the session start and is not adjusted for time already elapsed.",
    "Demand and solar inputs are placeholders until Persons 1 and 2 deliver.",
    "Anomaly risk is a historical abnormal rate, not a model prediction.",
    "Labels are operator recorded, not verified hardware diagnoses.",
]
_CALENDAR_WEATHER = {
    "hour", "dow", "is_weekend", "hour_sin", "hour_cos",
    "temp_c", "humidity", "precip_mm",
}
_FORBIDDEN_INPUTS = {str(name).strip().casefold() for name in FORBIDDEN}
_FORBIDDEN_INPUTS |= {"end_time", "end time", "duration", "duration_minutes"}


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    if isinstance(value, (str, datetime, pd.Timestamp)):
        try:
            stamp = pd.Timestamp(value)
        except Exception as exc:
            raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be an ISO-8601 local timestamp")
    if pd.isna(stamp):
        raise ValueError(f"{field} must be a valid timestamp")
    return stamp


def _reject_outcomes(value: Mapping[str, Any], prefix: str = "payload") -> None:
    for key, child in value.items():
        normalized = str(key).strip().casefold()
        if normalized in _FORBIDDEN_INPUTS or normalized.startswith("is_"):
            raise ValueError(f"{prefix}.{key} is post-event or outcome data and cannot be used for inference")
        if isinstance(child, Mapping):
            _reject_outcomes(child, f"{prefix}.{key}")


def _prepare_features(payload: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    _reject_outcomes(payload)
    required = ("charger_id", "location", "district", "session_start", "as_of")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Missing required inference fields: {missing}")
    if any(payload[name] is None for name in ("charger_id", "location", "district")):
        raise ValueError("charger_id, location, and district must not be null")

    start = _timestamp(payload["session_start"], "session_start")
    as_of = _timestamp(payload["as_of"], "as_of")
    if (start.tz is None) != (as_of.tz is None):
        raise ValueError("session_start and as_of must both be timezone-aware or both be local-naive")
    if as_of < start:
        raise ValueError("as_of cannot be earlier than session_start")

    hour = start.hour + start.minute / 60 + start.second / 3600
    row: dict[str, Any] = {
        "charger_id": payload["charger_id"],
        "location": payload["location"],
        "district": payload["district"],
        "hour": start.hour,
        "dow": start.dayofweek,
        "is_weekend": int(start.dayofweek >= 5),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
    }
    for name in ("temp_c", "humidity", "precip_mm"):
        if name in payload:
            row[name] = payload[name]

    history = payload.get("history_features", {})
    if not isinstance(history, Mapping):
        raise ValueError("history_features must be a mapping")
    _reject_outcomes(history, "payload.history_features")
    allowed_history = set(NUMERIC) - _CALENDAR_WEATHER
    unknown = set(history) - allowed_history
    if unknown:
        raise ValueError(f"Unsupported history feature(s): {sorted(unknown)}")
    row.update(history)

    risk = history.get("hist_abnormal_rate")
    if risk is not None:
        try:
            risk_value = float(risk)
        except (TypeError, ValueError) as exc:
            raise ValueError("hist_abnormal_rate must be a normalized number in [0, 1]") from exc
        if not math.isfinite(risk_value) or not 0 <= risk_value <= 1:
            raise ValueError("hist_abnormal_rate must be a normalized number in [0, 1]")
    return pd.DataFrame([row]), start, as_of


class AvailabilityPredictor:
    """Load an exported champion once and serve prediction calls."""

    def __init__(self, bundle_path: str | Path = DEFAULT_BUNDLE_PATH):
        path = Path(bundle_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Availability bundle not found at {path}. Run the availability training workflow first."
            )
        bundle = joblib.load(path)
        required = {"model", "encoder_state", "data_version"}
        missing = required - set(bundle)
        if missing:
            raise ValueError(f"Availability bundle is missing fields: {sorted(missing)}")
        self.model = bundle["model"]
        self.encoder = FeatureEncoder.from_state(bundle["encoder_state"])
        self.cap = float(bundle.get("duration_cap_min", bundle.get("CAP", 0)))
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError("Availability bundle has an invalid duration cap")
        self.target_mode = bundle.get("target_mode", "log1p")
        self.model_version = (
            f"{bundle.get('model_name', 'availability_duration')}@"
            f"{bundle['data_version'].get('data_sha256_12', 'unknown')}"
        )
        self.bundle_version = str(bundle.get("bundle_version", self.model_version))

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        features, start, as_of = _prepare_features(payload)
        encoded = self.encoder.transform(features)
        transformed = float(np.asarray(self.model.predict(encoded), dtype=float)[0])
        if self.target_mode == "log1p":
            transformed = float(np.expm1(transformed))
        duration = float(np.clip(transformed, 0.0, self.cap))
        free_time = start + pd.to_timedelta(duration, unit="m")
        wait = max(0.0, (free_time - as_of).total_seconds() / 60.0)
        history = payload.get("history_features", {})
        risk = history.get("hist_abnormal_rate")
        from ..recommendation.ranker import load_config
        result = {
            "charger_id": str(payload["charger_id"]),
            "predicted_duration_minutes": round(duration, 2),
            "expected_free_time": free_time.isoformat(),
            "expected_wait_minutes": round(wait, 2),
            "anomaly_risk": None if risk is None else float(risk),
            "model_version": self.model_version,
            "bundle_version": self.bundle_version,
            "config_version": load_config()["version"],
            "schema_version": AVAILABILITY_SCHEMA_VERSION,
            "limitations": list(_LIMITATIONS),
        }
        explain = payload.get("explain", False)
        if not isinstance(explain, bool):
            raise ValueError("explain must be a boolean")
        if explain:
            if isinstance(self.model, xgb.XGBModel):
                matrix = xgb.DMatrix(encoded, feature_names=list(encoded.columns))
                # sklearn wrappers with early stopping predict through
                # best_iteration, while Booster.predict otherwise uses every
                # tree. Match the wrapper's iteration range for additivity.
                best_iteration = getattr(self.model, "best_iteration", None)
                contrib_kwargs = (
                    {"iteration_range": (0, int(best_iteration) + 1)}
                    if best_iteration is not None
                    else {}
                )
                contributions = np.asarray(
                    self.model.get_booster().predict(
                        matrix, pred_contribs=True, **contrib_kwargs
                    ),
                    dtype=float,
                )[0]
                base_value = float(contributions[-1])
                feature_contributions = contributions[:-1]
                raw_margin = float(np.asarray(
                    self.model.predict(encoded, output_margin=True), dtype=float
                )[0])
                if not math.isclose(
                    base_value + float(feature_contributions.sum()),
                    raw_margin,
                    rel_tol=0.0,
                    abs_tol=1e-3,
                ):
                    raise AssertionError("XGBoost local contributions do not add up to the raw model margin")
                order = np.argsort(np.abs(feature_contributions))[::-1][:5]
                result["explanation"] = {
                    "unit": "log_minutes",
                    "base_value": base_value,
                    "top_contributors": [
                        {
                            "feature": str(encoded.columns[index]),
                            "value": float(encoded.iloc[0, index]),
                            "contribution": float(feature_contributions[index]),
                            "direction": "increases" if feature_contributions[index] >= 0 else "decreases",
                        }
                        for index in order
                    ],
                    "note": "Contributions are in log minutes and add up to the model output before conversion to minutes.",
                }
            else:
                result["explanation"] = None
                result["explanation_note"] = "Local explanation is available only for XGBoost bundles."
        return result


def predict_availability(
    payload: Mapping[str, Any], bundle_path: str | Path = DEFAULT_BUNDLE_PATH
) -> dict[str, Any]:
    """Load the saved champion and return duration, expected free time, and wait."""
    return AvailabilityPredictor(bundle_path).predict(payload)


def get_bundle_version(bundle_path: str | Path = DEFAULT_BUNDLE_PATH) -> str:
    """Read a bundle's version without requiring a prediction call."""
    path = Path(bundle_path)
    if not path.is_file():
        return "unavailable"
    bundle = joblib.load(path)
    explicit = bundle.get("bundle_version")
    if explicit:
        return str(explicit)
    data_version = bundle.get("data_version", {})
    digest = data_version.get("data_sha256_12", "unknown")
    return f"{bundle.get('model_name', 'availability_duration')}@{digest}"
