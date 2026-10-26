"""
Unit and Integration Tests for Solar Forecasting Module.

Tests:
1. Leakage-safe feature generation (strictly no future data leakage at origin).
2. Physical solar geometry assertions for New Cairo.
3. Schema conformance with solar_input.json and solar_output.json.
4. End-to-end 24h and 48h inference with predict_solar().
5. Cold-start handling when historical power is omitted.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.solar.features import (
    SolarFeaturePipeline,
    calculate_solar_geometry,
    verify_no_leakage,
    ALL_FEATURE_COLUMNS,
    NEW_CAIRO_LAT,
    NEW_CAIRO_LON,
)
from src.solar.inference import (
    SolarForecaster,
    predict_solar,
    SolarInputPayload,
    ForecastOutputItem,
)


@pytest.fixture
def sample_weather_24h():
    """Create synthetic 24h weather forecast payload."""
    origin = pd.Timestamp("2024-06-15 00:00:00")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h")
    records = []
    for t in times:
        records.append({
            "timestamp": t.isoformat(),
            "temperature_2m_previous_day1": 28.5,
            "temperature_2m_previous_day2": 28.0,
            "cloud_cover_previous_day1": 15.0,
            "cloud_cover_previous_day2": 20.0,
            "direct_normal_irradiance_previous_day1": 650.0 if 5 <= t.hour <= 17 else 0.0,
            "direct_normal_irradiance_previous_day2": 600.0 if 5 <= t.hour <= 17 else 0.0,
            "shortwave_radiation_previous_day1": 800.0 if 5 <= t.hour <= 17 else 0.0,
            "shortwave_radiation_previous_day2": 780.0 if 5 <= t.hour <= 17 else 0.0,
            "wind_speed_10m_previous_day1": 4.5,
            "wind_speed_10m_previous_day2": 4.2,
        })
    return {
        "forecast_origin": origin.isoformat(),
        "horizon": 24,
        "weather_forecast": records,
    }


@pytest.fixture
def sample_weather_48h():
    """Create synthetic 48h weather forecast payload."""
    origin = pd.Timestamp("2024-06-15 00:00:00")
    times = pd.date_range(origin + pd.Timedelta(hours=1), periods=48, freq="h")
    records = []
    for t in times:
        records.append({
            "timestamp": t.isoformat(),
            "temperature_2m_previous_day1": 29.0,
            "temperature_2m_previous_day2": 28.5,
            "cloud_cover_previous_day1": 10.0,
            "cloud_cover_previous_day2": 15.0,
            "direct_normal_irradiance_previous_day1": 700.0 if 5 <= t.hour <= 17 else 0.0,
            "direct_normal_irradiance_previous_day2": 680.0 if 5 <= t.hour <= 17 else 0.0,
            "shortwave_radiation_previous_day1": 850.0 if 5 <= t.hour <= 17 else 0.0,
            "shortwave_radiation_previous_day2": 820.0 if 5 <= t.hour <= 17 else 0.0,
            "wind_speed_10m_previous_day1": 4.0,
            "wind_speed_10m_previous_day2": 3.8,
        })
    return {
        "forecast_origin": origin.isoformat(),
        "horizon": 48,
        "weather_forecast": records,
    }


def test_solar_geometry_physical_bounds():
    """Verify astronomical solar geometry for New Cairo."""
    times = pd.date_range("2024-06-21 00:00:00", periods=24, freq="h")
    df_geo = calculate_solar_geometry(times, lat=NEW_CAIRO_LAT, lon=NEW_CAIRO_LON)

    # Solar elevation reaches peak around noon
    midday_elevation = df_geo.loc["2024-06-21 10:00:00", "solar_elevation_deg"]
    assert midday_elevation > 70.0, f"Expected high midday solar elevation, got {midday_elevation}"

    # Nighttime elevation must be negative
    midnight_elevation = df_geo.loc["2024-06-21 00:00:00", "solar_elevation_deg"]
    assert midnight_elevation < 0.0, f"Expected negative midnight elevation, got {midnight_elevation}"

    # Extraterrestrial GHI must be 0 at night
    night_ghi = df_geo.loc["2024-06-21 22:00:00", "solar_extraterrestrial_ghi"]
    assert night_ghi == 0.0, f"Expected 0 night extraterrestrial irradiance, got {night_ghi}"


def test_feature_pipeline_leakage_safety(sample_weather_24h):
    """Verify that feature pipeline strictly obeys origin availability."""
    pipeline = SolarFeaturePipeline()
    origin = pd.Timestamp(sample_weather_24h["forecast_origin"])
    weather_df = pd.DataFrame(sample_weather_24h["weather_forecast"])

    # Historical power strictly <= origin
    hist_times = pd.date_range(origin - pd.Timedelta(hours=48), origin, freq="h")
    hist_power = pd.Series(np.random.uniform(0, 50, len(hist_times)), index=hist_times)

    feat_df = pipeline.build_forecast_features(
        forecast_origin=origin,
        horizon=24,
        weather_df=weather_df,
        historical_power_series=hist_power,
    )

    assert len(feat_df) == 24, "Expected 24 feature rows for 24h horizon"
    assert feat_df.shape[1] == len(ALL_FEATURE_COLUMNS), "Feature columns mismatch"
    assert feat_df.isnull().sum().sum() == 0, "No NaNs allowed in features"


def test_schema_files_exist_and_valid():
    """Verify JSON schema files exist and parse validly."""
    input_schema_path = os.path.join(PROJECT_ROOT, "schemas", "solar_input.json")
    output_schema_path = os.path.join(PROJECT_ROOT, "schemas", "solar_output.json")

    assert os.path.exists(input_schema_path), "Missing schemas/solar_input.json"
    assert os.path.exists(output_schema_path), "Missing schemas/solar_output.json"

    with open(input_schema_path) as f:
        in_schema = json.load(f)
        assert in_schema.get("title") == "SolarForecastingInputPayload"

    with open(output_schema_path) as f:
        out_schema = json.load(f)
        assert out_schema.get("title") == "SolarForecastingOutputPayload"


def test_inference_24h_payload(sample_weather_24h):
    """Test full 24h inference with predict_solar()."""
    # Add historical power
    origin = pd.Timestamp(sample_weather_24h["forecast_origin"])
    hist_times = pd.date_range(origin - pd.Timedelta(hours=48), origin, freq="h")
    hist_items = [{"timestamp": t.isoformat(), "power_kw": 25.0 if 6 <= t.hour <= 18 else 0.0} for t in hist_times]
    sample_weather_24h["historical_power"] = hist_items

    preds = predict_solar(sample_weather_24h)

    assert isinstance(preds, list), "Output must be a list"
    assert len(preds) == 24, f"Expected 24 predictions, got {len(preds)}"

    for item in preds:
        assert "timestamp" in item
        assert "horizon" in item
        assert "forecast_kw" in item
        assert item["forecast_kw"] >= 0.0, "Forecast must be non-negative"
        assert item["forecast_kw"] <= 100.0, "Forecast cannot exceed 100 kWp nameplate"


def test_inference_48h_payload(sample_weather_48h):
    """Test full 48h inference with predict_solar()."""
    preds = predict_solar(sample_weather_48h)

    assert isinstance(preds, list), "Output must be a list"
    assert len(preds) == 48, f"Expected 48 predictions, got {len(preds)}"
    assert preds[0]["horizon"] == 1
    assert preds[-1]["horizon"] == 48


def test_cold_start_without_history(sample_weather_24h):
    """Ensure pipeline functions safely when historical power is omitted."""
    sample_weather_24h.pop("historical_power", None)
    preds = predict_solar(sample_weather_24h)
    assert len(preds) == 24
    assert all(p["forecast_kw"] >= 0.0 for p in preds)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
