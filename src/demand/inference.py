"""
Inference Module for Person 2 (EV Demand Forecasting).

Provides the official integration interface for Person 5 (FastAPI Backend):
- Function: predict_demand(payload: dict) -> dict
- Adheres strictly to schemas/demand_input.json and schemas/demand_output.json.
- Self-contained, production-ready, no notebook or global training state required.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger("demand_inference")

# Global lazy-loaded models and metadata
_MODEL_ENERGY: Optional[xgb.XGBRegressor] = None
_MODEL_SESSIONS: Optional[xgb.XGBRegressor] = None
_METADATA: Optional[Dict[str, Any]] = None


def _load_models(models_dir: str = "models"):
    global _MODEL_ENERGY, _MODEL_SESSIONS, _METADATA
    if _MODEL_ENERGY is not None and _MODEL_SESSIONS is not None:
        return
        
    models_path = Path(models_dir)
    meta_path = models_path / "champion_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Champion metadata not found at {meta_path}. Run train.py first.")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        _METADATA = json.load(f)
        
    energy_model_file = Path(_METADATA["energy_model_file"])
    sessions_model_file = Path(_METADATA["sessions_model_file"])
    
    _MODEL_ENERGY = xgb.XGBRegressor()
    _MODEL_ENERGY.load_model(str(energy_model_file))
    
    _MODEL_SESSIONS = xgb.XGBRegressor()
    _MODEL_SESSIONS.load_model(str(sessions_model_file))
    
    logger.info("Successfully loaded champion EV Demand models for inference.")


def _build_feature_row_from_payload(payload: Dict[str, Any], feature_cols: list) -> pd.DataFrame:
    """
    Construct a single-row feature vector from either direct payload fields
    or recent history array.
    """
    ts_str = payload.get("timestamp")
    if not ts_str:
        raise ValueError("Payload missing required 'timestamp' field.")
        
    target_dt = pd.to_datetime(ts_str)
    
    # 1. Target calendar attributes
    target_hour = target_dt.hour
    target_dow = target_dt.dayofweek
    target_is_weekend = int(target_dow >= 5)
    target_month = target_dt.month
    
    sin_hour = float(np.sin(2 * np.pi * target_hour / 24.0))
    cos_hour = float(np.cos(2 * np.pi * target_hour / 24.0))
    sin_dow = float(np.sin(2 * np.pi * target_dow / 7.0))
    cos_dow = float(np.cos(2 * np.pi * target_dow / 7.0))
    sin_month = float(np.sin(2 * np.pi * target_month / 12.0))
    cos_month = float(np.cos(2 * np.pi * target_month / 12.0))
    
    # 2. Extract or infer history lags
    history = payload.get("history")
    if history and len(history) >= 24:
        df_hist = pd.DataFrame(history)
        df_hist["timestamp"] = pd.to_datetime(df_hist["timestamp"])
        df_hist = df_hist.sort_values("timestamp").reset_index(drop=True)
        
        # Most recent observation is origin t (lag 1h relative to target)
        e_lag1 = float(df_hist["energy_kwh"].iloc[-1])
        s_lag1 = float(df_hist["sessions"].iloc[-1])
        e_lag2 = float(df_hist["energy_kwh"].iloc[-2]) if len(df_hist) >= 2 else e_lag1
        s_lag2 = float(df_hist["sessions"].iloc[-2]) if len(df_hist) >= 2 else s_lag1
        e_lag3 = float(df_hist["energy_kwh"].iloc[-3]) if len(df_hist) >= 3 else e_lag2
        s_lag3 = float(df_hist["sessions"].iloc[-3]) if len(df_hist) >= 3 else s_lag2
        
        e_lag24 = float(df_hist["energy_kwh"].iloc[-24]) if len(df_hist) >= 24 else e_lag1
        s_lag24 = float(df_hist["sessions"].iloc[-24]) if len(df_hist) >= 24 else s_lag1
        e_lag48 = float(df_hist["energy_kwh"].iloc[-48]) if len(df_hist) >= 48 else e_lag24
        s_lag48 = float(df_hist["sessions"].iloc[-48]) if len(df_hist) >= 48 else s_lag24
        e_lag168 = float(df_hist["energy_kwh"].iloc[-168]) if len(df_hist) >= 168 else e_lag24
        s_lag168 = float(df_hist["sessions"].iloc[-168]) if len(df_hist) >= 168 else s_lag24
        
        e_roll_6h = float(df_hist["energy_kwh"].iloc[-6:].mean()) if len(df_hist) >= 6 else e_lag1
        e_roll_std_6h = float(df_hist["energy_kwh"].iloc[-6:].std()) if len(df_hist) >= 6 else 0.0
        e_roll_24h = float(df_hist["energy_kwh"].iloc[-24:].mean()) if len(df_hist) >= 24 else e_lag1
        e_roll_std_24h = float(df_hist["energy_kwh"].iloc[-24:].std()) if len(df_hist) >= 24 else 0.0
        
        s_roll_6h = float(df_hist["sessions"].iloc[-6:].mean()) if len(df_hist) >= 6 else s_lag1
        s_roll_std_6h = float(df_hist["sessions"].iloc[-6:].std()) if len(df_hist) >= 6 else 0.0
        s_roll_24h = float(df_hist["sessions"].iloc[-24:].mean()) if len(df_hist) >= 24 else s_lag1
        s_roll_std_24h = float(df_hist["sessions"].iloc[-24:].std()) if len(df_hist) >= 24 else 0.0
    else:
        # Default or directly provided lag values
        e_lag1 = float(payload.get("energy_lag_1h", 350.0))
        e_lag2 = float(payload.get("energy_lag_2h", e_lag1))
        e_lag3 = float(payload.get("energy_lag_3h", e_lag2))
        e_lag24 = float(payload.get("energy_lag_24h", e_lag1))
        e_lag48 = float(payload.get("energy_lag_48h", e_lag24))
        e_lag168 = float(payload.get("energy_lag_168h", e_lag24))
        
        s_lag1 = float(payload.get("sessions_lag_1h", 20.0))
        s_lag2 = float(payload.get("sessions_lag_2h", s_lag1))
        s_lag3 = float(payload.get("sessions_lag_3h", s_lag2))
        s_lag24 = float(payload.get("sessions_lag_24h", s_lag1))
        s_lag48 = float(payload.get("sessions_lag_48h", s_lag24))
        s_lag168 = float(payload.get("sessions_lag_168h", s_lag24))
        
        e_roll_6h = float(payload.get("energy_rolling_mean_6h", (e_lag1 + e_lag2 + e_lag3) / 3.0))
        e_roll_std_6h = float(payload.get("energy_rolling_std_6h", 30.0))
        e_roll_24h = float(payload.get("energy_rolling_mean_24h", e_lag24))
        e_roll_std_24h = float(payload.get("energy_rolling_std_24h", 45.0))
        
        s_roll_6h = float(payload.get("sessions_rolling_mean_6h", (s_lag1 + s_lag2 + s_lag3) / 3.0))
        s_roll_std_6h = float(payload.get("sessions_rolling_std_6h", 3.0))
        s_roll_24h = float(payload.get("sessions_rolling_mean_24h", s_lag24))
        s_roll_std_24h = float(payload.get("sessions_rolling_std_24h", 4.0))

    # Weather
    temp_c = float(payload.get("temp_c", 18.0))
    humidity = float(payload.get("humidity", 65.0))
    precip_mm = float(payload.get("precip_mm", 0.0))
    
    row_dict = {
        "energy_lag_1h": e_lag1,
        "energy_lag_2h": e_lag2,
        "energy_lag_3h": e_lag3,
        "energy_lag_24h": e_lag24,
        "energy_lag_48h": e_lag48,
        "energy_lag_168h": e_lag168,
        "sessions_lag_1h": s_lag1,
        "sessions_lag_2h": s_lag2,
        "sessions_lag_3h": s_lag3,
        "sessions_lag_24h": s_lag24,
        "sessions_lag_48h": s_lag48,
        "sessions_lag_168h": s_lag168,
        "energy_rolling_mean_6h": e_roll_6h,
        "energy_rolling_std_6h": e_roll_std_6h,
        "energy_rolling_mean_24h": e_roll_24h,
        "energy_rolling_std_24h": e_roll_std_24h,
        "sessions_rolling_mean_6h": s_roll_6h,
        "sessions_rolling_std_6h": s_roll_std_6h,
        "sessions_rolling_mean_24h": s_roll_24h,
        "sessions_rolling_std_24h": s_roll_std_24h,
        "target_hour": target_hour,
        "target_dayofweek": target_dow,
        "target_is_weekend": target_is_weekend,
        "target_month": target_month,
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_dayofweek": sin_dow,
        "cos_dayofweek": cos_dow,
        "sin_month": sin_month,
        "cos_month": cos_month,
        "temp_c": temp_c,
        "humidity": humidity,
        "precip_mm": precip_mm
    }
    
    df_row = pd.DataFrame([row_dict])
    return df_row[feature_cols]


def predict_demand(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Primary API entry point for EV demand forecasting (Person 2 -> Person 5).
    
    Args:
        payload (dict): Conforming to schemas/demand_input.json
        
    Returns:
        dict: Conforming to schemas/demand_output.json
            {"timestamp", "expected_energy_kwh", "expected_sessions", "model_name", "model_version", "horizon"}
    """
    _load_models()
    
    feature_cols = _METADATA["feature_columns"]
    X_input = _build_feature_row_from_payload(payload, feature_cols)
    
    pred_energy = float(_MODEL_ENERGY.predict(X_input)[0])
    pred_sessions = float(_MODEL_SESSIONS.predict(X_input)[0])
    
    # Safe bounds clipping
    expected_energy_kwh = max(0.0, round(pred_energy, 2))
    expected_sessions = max(0.0, round(pred_sessions, 2))
    
    response = {
        "timestamp": str(payload.get("timestamp")),
        "expected_energy_kwh": expected_energy_kwh,
        "expected_sessions": expected_sessions,
        "model_name": f"{_METADATA['champion_model']}_Champion",
        "model_version": _METADATA["mlflow_model_uri"],
        "horizon": "1h"
    }
    return response


if __name__ == "__main__":
    # Test payload for 2022-01-01 14:00:00
    test_payload = {
        "timestamp": "2022-01-01T14:00:00",
        "energy_lag_1h": 420.5,
        "energy_lag_24h": 450.0,
        "sessions_lag_1h": 26.0,
        "sessions_lag_24h": 28.0,
        "temp_c": 12.5,
        "humidity": 55.0,
        "precip_mm": 0.0
    }
    output = predict_demand(test_payload)
    print("Inference Test Output:")
    print(json.dumps(output, indent=2))
