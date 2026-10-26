"""
Solar Power Forecasting Module for New Cairo Renewable Energy Pipeline.
Person 1 Hand-off for Multi-Horizon Solar Forecasting (24h and 48h).
"""

from .features import SolarFeaturePipeline, calculate_solar_geometry, NEW_CAIRO_LAT, NEW_CAIRO_LON
from .inference import SolarForecaster, predict_solar

__all__ = [
    "SolarFeaturePipeline",
    "calculate_solar_geometry",
    "NEW_CAIRO_LAT",
    "NEW_CAIRO_LON",
    "SolarForecaster",
    "predict_solar"
]
