"""
Hourly Demand Dataset Builder for Person 2 (EV Demand Forecasting).

This script aggregates raw charging transactions into a continuous hourly time series
without modifying the original raw files, aligns with daily weather context,
fills missing hours with zero activity, and saves the verified dataset.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_hourly")


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and tab characters from all object/string columns."""
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def load_and_aggregate_charging(charging_path: str) -> pd.DataFrame:
    """
    Load raw transaction data and aggregate to hourly totals.
    
    Target outputs:
    - energy_kwh: sum of Transaction power/kwh
    - sessions: count of transaction records
    - avg_session_duration_min: average duration of charging sessions
    """
    logger.info(f"Loading raw charging transactions from: {charging_path}")
    df = pd.read_csv(charging_path, encoding="gb18030")
    df = clean_string_columns(df)
    
    # Parse timestamps
    df["start_dt"] = pd.to_datetime(df["Start Time"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["End Time"], errors="coerce")
    
    # Filter invalid records if any
    valid_mask = df["start_dt"].notnull() & (df["Transaction power/kwh"] >= 0)
    df = df[valid_mask].copy()
    
    # Session duration in minutes
    df["duration_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60.0
    df["duration_min"] = df["duration_min"].clip(lower=0, upper=1440)  # max 24 hours
    
    # Assign to hourly bucket
    df["timestamp"] = df["start_dt"].dt.floor("h")
    
    # Group by hourly timestamp
    hourly = df.groupby("timestamp").agg(
        energy_kwh=("Transaction power/kwh", "sum"),
        sessions=("Transaction power/kwh", "count"),
        avg_session_duration_min=("duration_min", "mean")
    ).reset_index()
    
    logger.info(f"Aggregated {len(df)} transactions into {len(hourly)} raw active hourly buckets.")
    return hourly


def load_and_aggregate_weather(weather_path: str) -> pd.DataFrame:
    """
    Load daily weather data across Jiaxing districts and aggregate city-wide averages.
    """
    logger.info(f"Loading weather data from: {weather_path}")
    df_w = pd.read_csv(weather_path, encoding="gb18030")
    df_w = clean_string_columns(df_w)
    
    # Identify column names flexibly
    temp_col = [c for c in df_w.columns if "Temperature" in c][0]
    hum_col = [c for c in df_w.columns if "Humidity" in c][0]
    prec_col = [c for c in df_w.columns if "Precipitation" in c][0]
    
    # Parse date (format YYYYMMDD)
    df_w["date_dt"] = pd.to_datetime(df_w["Date"].astype(str), format="%Y%m%d")
    
    # City-wide average across the districts
    weather_city = df_w.groupby("date_dt").agg(
        temp_c=(temp_col, "mean"),
        humidity=(hum_col, "mean"),
        precip_mm=(prec_col, "mean")
    ).reset_index()
    
    logger.info(f"Loaded {len(weather_city)} days of weather observations.")
    return weather_city


def build_hourly_dataset(
    charging_path: str = r"c:\Users\WASEEM\AppData\Local\Temp\Charging_Data.csv",
    weather_path: str = r"c:\Users\WASEEM\AppData\Local\Temp\Weather_Data.csv",
    output_path: str = "data/processed/hourly_demand.csv",
    start_date: str = "2020-01-01 00:00:00",
    end_date: str = "2021-12-31 23:00:00"
) -> pd.DataFrame:
    """
    Build complete, continuous hourly dataset with aligned weather and zero-filled gaps.
    """
    # 1. Hourly charging aggregations
    hourly_charging = load_and_aggregate_charging(charging_path)
    
    # 2. Build full continuous hourly grid
    full_grid = pd.date_range(start=start_date, end=end_date, freq="h", name="timestamp")
    df_continuous = pd.DataFrame({"timestamp": full_grid})
    
    # 3. Merge charging into full grid
    df_merged = df_continuous.merge(hourly_charging, on="timestamp", how="left")
    df_merged["energy_kwh"] = df_merged["energy_kwh"].fillna(0.0).round(4)
    df_merged["sessions"] = df_merged["sessions"].fillna(0).astype(int)
    df_merged["avg_session_duration_min"] = df_merged["avg_session_duration_min"].fillna(0.0).round(2)
    
    # 4. Merge weather context safely by date
    weather_city = load_and_aggregate_weather(weather_path)
    df_merged["date_dt"] = df_merged["timestamp"].dt.floor("D")
    df_merged = df_merged.merge(weather_city, on="date_dt", how="left")
    df_merged.drop(columns=["date_dt"], inplace=True)
    
    # Forward-fill / back-fill any potential boundary weather if needed
    df_merged["temp_c"] = df_merged["temp_c"].ffill().bfill().round(2)
    df_merged["humidity"] = df_merged["humidity"].ffill().bfill().round(2)
    df_merged["precip_mm"] = df_merged["precip_mm"].ffill().bfill().round(2)
    
    # Validation checks
    assert len(df_merged) == len(full_grid), f"Length mismatch: {len(df_merged)} vs {len(full_grid)}"
    assert df_merged["timestamp"].is_monotonic_increasing, "Timestamps are not strictly chronological!"
    assert df_merged.isnull().sum().sum() == 0, f"Found nulls in dataset: {df_merged.isnull().sum()}"
    
    # Save output
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    logger.info(f"Successfully saved continuous hourly dataset to {output_path} with {len(df_merged)} rows.")
    
    return df_merged


if __name__ == "__main__":
    c_path = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\WASEEM\AppData\Local\Temp\Charging_Data.csv"
    w_path = sys.argv[2] if len(sys.argv) > 2 else r"c:\Users\WASEEM\AppData\Local\Temp\Weather_Data.csv"
    o_path = sys.argv[3] if len(sys.argv) > 3 else "data/processed/hourly_demand.csv"
    build_hourly_dataset(c_path, w_path, o_path)
