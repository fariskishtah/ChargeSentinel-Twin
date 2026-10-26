"""
Feature Engineering Pipeline for Person 2 (EV Demand Forecasting).

Ensures strict leakage protection:
- All lag and rolling statistics shift strictly before computation.
- Rolling windows only look back into the past.
- Target is next-hour demand (t+1).
- Transformations (scaling, encoders) fit strictly on training splits.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("demand_features")

FEATURE_COLUMNS: List[str] = [
    # Lags (Relative to target t+1: lag 1 is origin t, lag 24 is same hour yesterday, etc.)
    "energy_lag_1h", "energy_lag_2h", "energy_lag_3h", "energy_lag_24h", "energy_lag_48h", "energy_lag_168h",
    "sessions_lag_1h", "sessions_lag_2h", "sessions_lag_3h", "sessions_lag_24h", "sessions_lag_48h", "sessions_lag_168h",
    # Rolling Statistics over past history
    "energy_rolling_mean_6h", "energy_rolling_std_6h",
    "energy_rolling_mean_24h", "energy_rolling_std_24h",
    "sessions_rolling_mean_6h", "sessions_rolling_std_6h",
    "sessions_rolling_mean_24h", "sessions_rolling_std_24h",
    # Target Hour Calendar Features
    "target_hour", "target_dayofweek", "target_is_weekend", "target_month",
    # Cyclical Calendar Encodings
    "sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "sin_month", "cos_month",
    # Known Weather Context
    "temp_c", "humidity", "precip_mm"
]

TARGET_COLUMNS: List[str] = ["target_energy_kwh", "target_sessions"]


def extract_features_df(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    """
    Extract leakage-safe feature set from hourly demand series.
    Each row at index i represents forecast origin t to predict target at t+1.
    """
    df = df.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Ensure sorted by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 1. Target definitions (Next-hour demand at t+1)
    if include_targets:
        df["target_energy_kwh"] = df["energy_kwh"].shift(-1)
        df["target_sessions"] = df["sessions"].shift(-1)
    
    # 2. Lag features (t - lag relative to origin t)
    # At origin t: lag 1h is energy at t; lag 2h is energy at t-1; lag 24h is t-23, etc.
    df["energy_lag_1h"] = df["energy_kwh"]
    df["energy_lag_2h"] = df["energy_kwh"].shift(1)
    df["energy_lag_3h"] = df["energy_kwh"].shift(2)
    df["energy_lag_24h"] = df["energy_kwh"].shift(23)
    df["energy_lag_48h"] = df["energy_kwh"].shift(47)
    df["energy_lag_168h"] = df["energy_kwh"].shift(167)
    
    df["sessions_lag_1h"] = df["sessions"]
    df["sessions_lag_2h"] = df["sessions"].shift(1)
    df["sessions_lag_3h"] = df["sessions"].shift(2)
    df["sessions_lag_24h"] = df["sessions"].shift(23)
    df["sessions_lag_48h"] = df["sessions"].shift(47)
    df["sessions_lag_168h"] = df["sessions"].shift(167)
    
    # 3. Rolling statistics strictly up to origin t
    # Window 6h: past 6 hours [t-5, t]
    df["energy_rolling_mean_6h"] = df["energy_kwh"].rolling(window=6).mean()
    df["energy_rolling_std_6h"] = df["energy_kwh"].rolling(window=6).std().fillna(0.0)
    df["energy_rolling_mean_24h"] = df["energy_kwh"].rolling(window=24).mean()
    df["energy_rolling_std_24h"] = df["energy_kwh"].rolling(window=24).std().fillna(0.0)
    
    df["sessions_rolling_mean_6h"] = df["sessions"].rolling(window=6).mean()
    df["sessions_rolling_std_6h"] = df["sessions"].rolling(window=6).std().fillna(0.0)
    df["sessions_rolling_mean_24h"] = df["sessions"].rolling(window=24).mean()
    df["sessions_rolling_std_24h"] = df["sessions"].rolling(window=24).std().fillna(0.0)
    
    # 4. Target Calendar features (Known at forecast time t+1)
    target_dt = df["timestamp"] + pd.Timedelta(hours=1)
    df["target_hour"] = target_dt.dt.hour
    df["target_dayofweek"] = target_dt.dt.dayofweek
    df["target_is_weekend"] = (df["target_dayofweek"] >= 5).astype(int)
    df["target_month"] = target_dt.dt.month
    
    # 5. Cyclical trigonometric encodings
    df["sin_hour"] = np.sin(2 * np.pi * df["target_hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["target_hour"] / 24.0)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["target_dayofweek"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["target_dayofweek"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["target_month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["target_month"] / 12.0)
    
    return df


class DemandFeaturePipeline:
    """
    Stateful pipeline that fits scaling only on training splits,
    providing clean transforms for training and production inference.
    """
    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> "DemandFeaturePipeline":
        """Fit scaler strictly on training features."""
        self.scaler.fit(X[self.feature_columns])
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features with fitted scaler."""
        if not self.is_fitted:
            raise ValueError("DemandFeaturePipeline must be fitted before transform.")
        return self.scaler.transform(X[self.feature_columns])
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def test_leakage_protection(sample_path: str = "data/processed/hourly_demand.csv") -> bool:
    """
    Formal verification of zero data leakage:
    Perturbing a future transaction/target row (i+1) MUST NOT affect features at row i.
    """
    df = pd.read_csv(sample_path).head(300)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Base feature extraction
    feats_orig = extract_features_df(df, include_targets=True)
    
    # Choose arbitrary row i
    test_idx = 200
    row_i_orig = feats_orig.iloc[test_idx][FEATURE_COLUMNS].copy()
    
    # Perturb row i+1 heavily (simulate massive change in future)
    df_perturbed = df.copy()
    df_perturbed.loc[test_idx + 1, "energy_kwh"] += 99999.0
    df_perturbed.loc[test_idx + 1, "sessions"] += 1000
    
    feats_perturbed = extract_features_df(df_perturbed, include_targets=True)
    row_i_perturbed = feats_perturbed.iloc[test_idx][FEATURE_COLUMNS].copy()
    
    # Assert zero discrepancy
    diff = (row_i_orig - row_i_perturbed).abs().max()
    assert diff == 0.0, f"LEAKAGE DETECTED! Max feature difference at row {test_idx} is {diff}"
    logger.info("PASS: Zero data leakage confirmed. Future row alterations do not affect historical features.")
    return True


if __name__ == "__main__":
    test_leakage_protection()
