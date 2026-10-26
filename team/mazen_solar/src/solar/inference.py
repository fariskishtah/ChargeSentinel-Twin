"""
Solar Power Forecasting Inference Engine.

Delivers Person 1 Integration Interface for Person 5:
- Loadable pipeline from local artifact or MLflow model URI
- Strict Pydantic and JSON Schema validation
- Multi-horizon support (24h and 48h)
- predict_solar(payload) top-level function
- Zero-notebook dependency
"""

import os
import json
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.solar.features import (
    SolarFeaturePipeline,
    calculate_solar_geometry,
    NEW_CAIRO_LAT,
    NEW_CAIRO_LON,
    PV_SYSTEM_CAPACITY_KW,
    ALL_FEATURE_COLUMNS,
)

# Default model artifact path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_MODEL_PATH = os.path.join(
    DEFAULT_MODEL_DIR,
    "solar_xgboost_pipeline.joblib",
)
# ============================================================================
# Pydantic Schemas for Runtime Validation
# ============================================================================

class HistoricalPowerItem(BaseModel):
    timestamp: str
    power_kw: float = Field(ge=0.0)


class WeatherForecastItem(BaseModel):
    timestamp: str
    temperature_2m_previous_day1: float
    temperature_2m_previous_day2: Optional[float] = None
    cloud_cover_previous_day1: Optional[float] = 0.0
    cloud_cover_previous_day2: Optional[float] = None
    direct_normal_irradiance_previous_day1: Optional[float] = 0.0
    direct_normal_irradiance_previous_day2: Optional[float] = None
    shortwave_radiation_previous_day1: float
    shortwave_radiation_previous_day2: Optional[float] = None
    wind_speed_10m_previous_day1: Optional[float] = 0.0
    wind_speed_10m_previous_day2: Optional[float] = None


class SolarInputPayload(BaseModel):
    forecast_origin: str
    horizon: int = Field(default=24)
    historical_power: Optional[List[HistoricalPowerItem]] = None
    weather_forecast: List[WeatherForecastItem]

    @field_validator("horizon")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in (24, 48):
            raise ValueError(f"Horizon must be 24 or 48 hours, got {v}")
        return v


class ForecastOutputItem(BaseModel):
    timestamp: str
    horizon: int = Field(ge=1, le=48)
    forecast_kw: float = Field(ge=0.0)


# ============================================================================
# Solar Forecaster Pipeline Wrapper
# ============================================================================

class SolarForecaster:
    """
    Production Solar Power Forecaster for New Cairo.
    Encapsulates feature extraction, model prediction, and physical post-processing.
    """

    def __init__(
        self,
        model: Any,
        feature_columns: List[str] = ALL_FEATURE_COLUMNS,
        system_capacity_kw: float = PV_SYSTEM_CAPACITY_KW,
        model_name: str = "XGBoost",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.system_capacity_kw = system_capacity_kw
        self.model_name = model_name
        self.metadata = metadata or {}
        self.pipeline = SolarFeaturePipeline(system_capacity_kw=system_capacity_kw)

    def predict(self, payload: Union[Dict[str, Any], SolarInputPayload]) -> List[Dict[str, Any]]:
        """
        Execute multi-horizon solar forecast from input payload.

        Returns:
        --------
        List[Dict[str, Any]]: List of {'timestamp': ..., 'horizon': ..., 'forecast_kw': ...}
        """
        if isinstance(payload, dict):
            validated = SolarInputPayload(**payload)
        else:
            validated = payload

        origin_ts = pd.to_datetime(validated.forecast_origin)
        horizon = validated.horizon

        # Parse weather forecast into DataFrame
        weather_records = [item.model_dump() for item in validated.weather_forecast]
        weather_df = pd.DataFrame(weather_records)
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        weather_df = weather_df.set_index("timestamp").sort_index()

        # Parse historical power into Series if provided
        hist_power_series = None
        if validated.historical_power:
            hist_records = [item.model_dump() for item in validated.historical_power]
            df_hist = pd.DataFrame(hist_records)
            df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
            hist_power_series = df_hist.set_index("timestamp")["power_kw"].sort_index()

        # Build feature matrix
        X_df = self.pipeline.build_forecast_features(
            forecast_origin=origin_ts,
            horizon=horizon,
            weather_df=weather_df,
            historical_power_series=hist_power_series,
        )

        # Ensure correct column ordering
        X_mat = X_df[self.feature_columns]

        # Model inference
        if hasattr(self.model, "predict"):
            raw_preds = self.model.predict(X_mat)
        elif callable(self.model):
            raw_preds = self.model(X_mat)
        else:
            raise TypeError(f"Loaded model of type {type(self.model)} is not callable.")

        raw_preds = np.asarray(raw_preds, dtype=float).ravel()

        # Physical post-processing:
        # 1. Non-negative clipping
        preds = np.maximum(0.0, raw_preds)
        # 2. Capacity clipping (cannot exceed 100 kWp nameplate)
        preds = np.minimum(self.system_capacity_kw, preds)
        # 3. Nighttime zero-forcing using astronomical solar geometry
        # If solar elevation <= 0 or extraterrestrial irradiance == 0, output is exactly 0.0 kW
        is_daylight = X_df["is_daylight"].to_numpy()
        elevation = X_df["solar_elevation_deg"].to_numpy()
        preds = np.where((is_daylight <= 0.0) | (elevation <= 0.0), 0.0, preds)

        # Assemble output payload
        output_records = []
        for i, (ts, kw) in enumerate(zip(X_df.index, preds), start=1):
            item = ForecastOutputItem(
                timestamp=ts.isoformat(),
                horizon=i,
                forecast_kw=round(float(kw), 4),
            )
            output_records.append(item.model_dump())

        return output_records

    def save(self, file_path: str) -> None:
        """Save forecaster pipeline bundle to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        bundle = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "system_capacity_kw": self.system_capacity_kw,
            "model_name": self.model_name,
            "metadata": self.metadata,
        }
        joblib.dump(bundle, file_path)

    @classmethod
    def load(cls, model_path_or_uri: Optional[str] = None) -> "SolarForecaster":
        """
        Load forecaster from local path or MLflow model URI.
        """
        path = model_path_or_uri or DEFAULT_MODEL_PATH

        # Check if MLflow model URI
        if path.startswith("runs:/") or path.startswith("models:/"):
            import mlflow
            pyfunc_model = mlflow.pyfunc.load_model(path)
            return cls(model=pyfunc_model, model_name="MLflow_PyFunc")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at '{path}'. Please train the model first by running src/solar/train.py"
            )

        bundle = joblib.load(path)
        if isinstance(bundle, dict) and "model" in bundle:
            return cls(
                model=bundle["model"],
                feature_columns=bundle.get("feature_columns", ALL_FEATURE_COLUMNS),
                system_capacity_kw=bundle.get("system_capacity_kw", PV_SYSTEM_CAPACITY_KW),
                model_name=bundle.get("model_name", "LoadedModel"),
                metadata=bundle.get("metadata", {}),
            )
        else:
            # Direct estimator
            return cls(model=bundle)


# ============================================================================
# Singleton Cached Forecaster for Rapid API Serving
# ============================================================================

_CACHED_FORECASTER: Optional[SolarForecaster] = None


def get_forecaster(model_path: Optional[str] = None) -> SolarForecaster:
    """Return cached forecaster instance or load if needed."""
    global _CACHED_FORECASTER
    if _CACHED_FORECASTER is None or model_path is not None:
        _CACHED_FORECASTER = SolarForecaster.load(model_path)
    return _CACHED_FORECASTER


def predict_solar(
    payload: Dict[str, Any],
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Main Person 1 Integration Entry Point for Person 5.

    Parameters:
    -----------
    payload: dict
        Dictionary matching schemas/solar_input.json:
        {
            "forecast_origin": "2024-06-15T00:00:00",
            "horizon": 24, # or 48
            "historical_power": [{"timestamp": ..., "power_kw": ...}, ...], # optional
            "weather_forecast": [{"timestamp": ..., "temperature_2m_previous_day1": ..., ...}, ...]
        }

    Returns:
    --------
    List[dict]:
        [
            {"timestamp": "2024-06-15T01:00:00", "horizon": 1, "forecast_kw": 0.0},
            ...
        ]
    """
    forecaster = get_forecaster(model_path)
    return forecaster.predict(payload)
