"""
Solar Power Forecasting Training & Evaluation Pipeline.

Comprehensive, Leakage-Safe Time Series Machine Learning:
- Chronological Holdout Split (Untouched test period: June 2024)
- Rolling-Origin Walk-Forward Cross-Validation (Jan 22 - May 31, 2024)
- Three Model Families:
    1. Seasonal Persistence & Ridge Regression Baseline
    2. Optuna-Tuned XGBoost Regressor
    3. Compact PyTorch LSTM
- Multi-Horizon Evaluation: 24h and 48h
- Metrics: MAE, RMSE, sMAPE (overall and daylight), R2
- MLflow Experiment Tracking with SQLite Backend
- Global and Local SHAP Explainability
- Saves Model Artifacts and Verification Handoff for Person 5
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import optuna
import shap
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.solar.features import (
    SolarFeaturePipeline,
    calculate_solar_geometry,
    ALL_FEATURE_COLUMNS,
    NEW_CAIRO_LAT,
    NEW_CAIRO_LON,
    PV_SYSTEM_CAPACITY_KW,
)
from src.solar.inference import SolarForecaster

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SolarTraining")

# Default Paths
PVGIS_PATH = os.path.join(
    PROJECT_ROOT, "PVGIS", "pvgis_v6_hourly_2024_new_cairo_100kwp.json"
)
OPENMETEO_PATH = os.path.join(
    PROJECT_ROOT,
    "OpenMeteo",
    "open_meteo_previous_runs_gfs_seamless_2024_h1_new_cairo.json",
)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")
MLFLOW_TRACKING_URI = f"sqlite:///{DB_PATH}"
EXPERIMENT_NAME = "solar_forecasting_new_cairo"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================
# Metrics Implementation
# ============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-4) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics:
    - MAE (kW)
    - RMSE (kW)
    - sMAPE (%) overall
    - R2 overall
    - Daylight MAE, RMSE, sMAPE (for y_true > 0.5 kW)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Physical clipping
    y_pred_clipped = np.clip(y_pred, 0.0, PV_SYSTEM_CAPACITY_KW)

    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))

    # Overall sMAPE
    denom = np.abs(y_true) + np.abs(y_pred_clipped) + eps
    smape = float(np.mean(200.0 * np.abs(y_true - y_pred_clipped) / denom))

    r2 = float(r2_score(y_true, y_pred_clipped)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0

    # Daylight metrics (active solar generation hours)
    day_mask = y_true > 0.5
    if np.sum(day_mask) > 10:
        day_true = y_true[day_mask]
        day_pred = y_pred_clipped[day_mask]
        day_mae = float(mean_absolute_error(day_true, day_pred))
        day_rmse = float(np.sqrt(mean_squared_error(day_true, day_pred)))
        day_denom = np.abs(day_true) + np.abs(day_pred) + eps
        day_smape = float(np.mean(200.0 * np.abs(day_true - day_pred) / day_denom))
        day_r2 = float(r2_score(day_true, day_pred)) if np.var(day_true) > 0 else 0.0
    else:
        day_mae, day_rmse, day_smape, day_r2 = mae, rmse, smape, r2

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "smape": round(smape, 2),
        "r2": round(r2, 4),
        "daylight_mae": round(day_mae, 4),
        "daylight_rmse": round(day_rmse, 4),
        "daylight_smape": round(day_smape, 2),
        "daylight_r2": round(day_r2, 4),
    }


# ============================================================================
# PyTorch Compact LSTM Model Definition
# ============================================================================

class CompactSolarLSTM(nn.Module):
    """Compact 2-Layer LSTM with Linear Projection Head."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, horizon, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out.squeeze(-1)  # (batch_size, horizon)


class LSTMSequenceDataset(Dataset):
    def __init__(self, X_blocks: np.ndarray, y_blocks: np.ndarray):
        # X_blocks shape: (num_origins, horizon, n_features)
        # y_blocks shape: (num_origins, horizon)
        self.X = torch.tensor(X_blocks, dtype=torch.float32)
        self.y = torch.tensor(y_blocks, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMForecasterWrapper:
    """Scikit-Learn compatible wrapper for Compact PyTorch LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        num_layers: int = 2,
        lr: float = 0.003,
        epochs: int = 40,
        batch_size: int = 16,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = CompactSolarLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def fit_blocks(self, X_blocks: np.ndarray, y_blocks: np.ndarray) -> "LSTMForecasterWrapper":
        # Fit scaler on flattened 2D data
        n_origins, horizon, n_feat = X_blocks.shape
        X_flat = X_blocks.reshape(-1, n_feat)
        self.scaler.fit(X_flat)

        X_norm = self.scaler.transform(X_flat).reshape(n_origins, horizon, n_feat)
        dataset = LSTMSequenceDataset(X_norm, y_blocks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for X_b, y_b in dataloader:
                optimizer.zero_grad()
                pred = self.model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()

        return self

    def predict_blocks(self, X_blocks: np.ndarray) -> np.ndarray:
        self.model.eval()
        n_origins, horizon, n_feat = X_blocks.shape
        X_flat = X_blocks.reshape(-1, n_feat)
        X_norm = self.scaler.transform(X_flat).reshape(n_origins, horizon, n_feat)

        with torch.no_grad():
            tensor_x = torch.tensor(X_norm, dtype=torch.float32)
            preds = self.model(tensor_x).cpu().numpy()
        return preds.ravel()


# ============================================================================
# Seasonal Persistence & Baseline Models
# ============================================================================

class SeasonalPersistenceModel:
    """Predicts power using safe historical power from prior day/2 days."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalPersistenceModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "pv_seasonal_persistence" in X.columns:
            return np.maximum(0.0, X["pv_seasonal_persistence"].to_numpy())
        elif "pv_same_hour_prev_day" in X.columns:
            return np.maximum(0.0, X["pv_same_hour_prev_day"].to_numpy())
        return np.zeros(len(X), dtype=float)


# ============================================================================
# Exploratory Data Analysis & Audit Plots
# ============================================================================

def generate_eda_figures(
    df_weather: pd.DataFrame,
    pv_kw: pd.Series,
    save_dir: str = FIGURES_DIR,
) -> None:
    """Generate high-quality EDA figures for the technical report."""
    logger.info("Generating EDA and audit visualization plots...")

    combined = df_weather.copy()
    combined["pv_kw"] = pv_kw
    combined["hour"] = combined.index.hour
    combined["month"] = combined.index.month_name()

    # 1. Diurnal Solar Generation & Seasonality
    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("viridis", n_colors=combined["month"].nunique())
    sns.lineplot(
        data=combined,
        x="hour",
        y="pv_kw",
        hue="month",
        palette=palette,
        errorbar=None,
        linewidth=2.2,
    )
    plt.title("New Cairo PV Power (100 kWp System): Diurnal Profile by Month (2024 H1)", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (UTC)", fontsize=11)
    plt.ylabel("PV Power (kW)", fontsize=11)
    plt.xticks(np.arange(0, 24, 2))
    plt.legend(title="Month", loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_seasonality_and_diurnal.png"))
    plt.close()

    # 2. Correlation Heatmap
    corr_cols = [
        "pv_kw",
        "shortwave_radiation_previous_day1",
        "direct_normal_irradiance_previous_day1",
        "temperature_2m_previous_day1",
        "cloud_cover_previous_day1",
        "wind_speed_10m_previous_day1",
    ]
    sub_corr = combined[corr_cols].dropna().corr()
    sub_corr.columns = [
        "PV Power (kW)",
        "GHI Day-1",
        "DNI Day-1",
        "Temp 2m Day-1",
        "Cloud Cover Day-1",
        "Wind Speed Day-1",
    ]
    sub_corr.index = sub_corr.columns

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        sub_corr,
        annot=True,
        cmap="coolwarm",
        vmin=-0.4,
        vmax=1.0,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation"},
    )
    plt.title("Correlation Matrix: Solar Generation vs NWP Forecast Variables", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_correlation_heatmap.png"))
    plt.close()
    logger.info("EDA figures successfully generated.")


# ============================================================================
# Walk-Forward / Rolling-Origin Cross-Validation & Optuna
# ============================================================================

def run_rolling_origin_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    model_factory,
    n_splits: int = 3,
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Perform rolling-origin walk-forward cross-validation strictly on training data.
    """
    unique_origins = np.sort(meta_train["origin"].unique())
    n_origins = len(unique_origins)
    split_size = n_origins // (n_splits + 1)

    cv_scores = []
    for split in range(1, n_splits + 1):
        train_origins = unique_origins[: split * split_size]
        val_origins = unique_origins[split * split_size : (split + 1) * split_size]

        train_mask = meta_train["origin"].isin(train_origins)
        val_mask = meta_train["origin"].isin(val_origins)

        X_tr, y_tr = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_train[val_mask], y_train[val_mask]

        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        metrics = calculate_metrics(y_val.to_numpy(), preds)
        cv_scores.append(metrics)

    mean_mae = float(np.mean([s["mae"] for s in cv_scores]))
    return mean_mae, cv_scores


def tune_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    n_trials: int = 15,
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna strictly inside training folds.
    """
    logger.info(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        def create_model():
            return xgb.XGBRegressor(**params)

        mean_mae, _ = run_rolling_origin_cv(
            X_train=X_train,
            y_train=y_train,
            meta_train=meta_train,
            model_factory=create_model,
            n_splits=3,
        )
        return mean_mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Optuna complete! Best CV MAE: {study.best_value:.4f} kW")
    logger.info(f"Best parameters: {study.best_params}")
    return study.best_params


# ============================================================================
# Main Training & Evaluation Workflow
# ============================================================================

def train_and_evaluate_solar_models(
    horizon: int = 24,
    origin_step_hours: int = 6,
    optuna_trials: int = 15,
) -> Dict[str, Any]:
    """
    Full pipeline training and multi-horizon comparison across three families:
    1. Seasonal Persistence & Ridge Regression
    2. XGBoost (Optuna-tuned)
    3. Compact PyTorch LSTM
    """
    logger.info(f"============================================================")
    logger.info(f"Starting Solar Forecasting Training Pipeline | Horizon: {horizon}h")
    logger.info(f"============================================================")

    # 1. Load PVGIS & Open-Meteo Data
    with open(OPENMETEO_PATH) as f:
        om_raw = json.load(f)
    with open(PVGIS_PATH) as f:
        pvgis_raw = json.load(f)

    df_weather = pd.DataFrame(om_raw["hourly"])
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    df_weather = df_weather.set_index("time").sort_index()

    # Align PVGIS power to Open-Meteo 2024 H1 timeline
    pvgis_power_w = pvgis_raw["power"][: len(df_weather)]
    pv_power_kw = pd.Series(pvgis_power_w, index=df_weather.index, dtype=float) / 1000.0

    # Clean missing points in PV target
    pv_power_kw = pv_power_kw.interpolate(method="time").fillna(0.0)

    # Generate EDA figures if horizon == 24
    if horizon == 24:
        generate_eda_figures(df_weather, pv_power_kw)

    # 2. Build Leakage-Safe Feature Dataset
    pipeline = SolarFeaturePipeline()
    logger.info("Extracting leakage-safe features across all forecast origins...")
    X_all, y_all, meta_all = pipeline.build_dataset_from_raw(
        df_weather=df_weather,
        pv_power_kw=pv_power_kw,
        horizon=horizon,
        origin_step_hours=origin_step_hours,
        start_date="2024-01-22 00:00:00",
        end_date="2024-06-30 23:00:00",
    )

    # 3. Chronological Train / Test Split
    # Holdout Test: June 2024 (2024-06-01 to 2024-06-30)
    # Train/Val: Jan 22, 2024 to May 31, 2024
    test_start = pd.to_datetime("2024-06-01 00:00:00")
    train_mask = meta_all["origin"] < test_start
    test_mask = meta_all["origin"] >= test_start

    X_train, y_train, meta_train = (
        X_all[train_mask].reset_index(drop=True),
        y_all[train_mask].reset_index(drop=True),
        meta_all[train_mask].reset_index(drop=True),
    )
    X_test, y_test, meta_test = (
        X_all[test_mask].reset_index(drop=True),
        y_all[test_mask].reset_index(drop=True),
        meta_all[test_mask].reset_index(drop=True),
    )

    logger.info(
        f"Chronological Split: Training samples={len(X_train)} (origins: {meta_train['origin'].min()} to {meta_train['origin'].max()})"
    )
    logger.info(
        f"Chronological Split: Test Holdout samples={len(X_test)} (origins: {meta_test['origin'].min()} to {meta_test['origin'].max()})"
    )

    # 4. Setup MLflow Experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results_summary = {}

    # ------------------------------------------------------------------------
    # MODEL 1: Seasonal Persistence Baseline
    # ------------------------------------------------------------------------
    logger.info("Evaluating Seasonal Persistence Baseline...")
    with mlflow.start_run(run_name=f"Seasonal_Persistence_{horizon}h"):
        pers_model = SeasonalPersistenceModel()
        pers_preds = pers_model.predict(X_test)
        pers_metrics = calculate_metrics(y_test.to_numpy(), pers_preds)

        mlflow.log_param("family", "Baseline")
        mlflow.log_param("model_type", "SeasonalPersistence")
        mlflow.log_param("horizon", horizon)
        for k, v in pers_metrics.items():
            mlflow.log_metric(k, v)

        results_summary["Seasonal Persistence"] = {
            "metrics": pers_metrics,
            "preds": pers_preds,
        }
        logger.info(f"Seasonal Persistence -> MAE: {pers_metrics['mae']} kW | RMSE: {pers_metrics['rmse']} kW | sMAPE: {pers_metrics['smape']}%")

    # ------------------------------------------------------------------------
    # MODEL 2: Ridge Regression Baseline
    # ------------------------------------------------------------------------
    logger.info("Training Ridge Regression Baseline...")
    with mlflow.start_run(run_name=f"Ridge_Regression_{horizon}h"):
        ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        ridge_pipe.fit(X_train, y_train)
        ridge_preds = ridge_pipe.predict(X_test)

        # Apply physical night clipping
        is_daylight_test = X_test["is_daylight"].to_numpy()
        elevation_test = X_test["solar_elevation_deg"].to_numpy()
        ridge_preds = np.where((is_daylight_test <= 0.0) | (elevation_test <= 0.0), 0.0, ridge_preds)
        ridge_preds = np.clip(ridge_preds, 0.0, PV_SYSTEM_CAPACITY_KW)

        ridge_metrics = calculate_metrics(y_test.to_numpy(), ridge_preds)

        mlflow.log_param("family", "Baseline_Linear")
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", 10.0)
        mlflow.log_param("horizon", horizon)
        for k, v in ridge_metrics.items():
            mlflow.log_metric(k, v)

        results_summary["Ridge Regression"] = {
            "metrics": ridge_metrics,
            "preds": ridge_preds,
        }
        logger.info(f"Ridge Regression -> MAE: {ridge_metrics['mae']} kW | RMSE: {ridge_metrics['rmse']} kW | sMAPE: {ridge_metrics['smape']}%")

    # ------------------------------------------------------------------------
    # MODEL 3: Tuned XGBoost Regressor
    # ------------------------------------------------------------------------
    logger.info("Tuning and Training XGBoost Regressor...")
    best_params = tune_xgboost_optuna(
        X_train=X_train,
        y_train=y_train,
        meta_train=meta_train,
        n_trials=optuna_trials,
    )

    with mlflow.start_run(run_name=f"XGBoost_{horizon}h") as xgb_run:
        xgb_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        raw_xgb_preds = xgb_model.predict(X_test)

        # Physical post-processing
        xgb_preds = np.clip(raw_xgb_preds, 0.0, PV_SYSTEM_CAPACITY_KW)
        xgb_preds = np.where((is_daylight_test <= 0.0) | (elevation_test <= 0.0), 0.0, xgb_preds)

        xgb_metrics = calculate_metrics(y_test.to_numpy(), xgb_preds)

        mlflow.log_param("family", "GradientBoostedTrees")
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("horizon", horizon)
        for p_name, p_val in best_params.items():
            mlflow.log_param(f"optuna_{p_name}", p_val)
        for k, v in xgb_metrics.items():
            mlflow.log_metric(k, v)

        # Log per-horizon MAE
        meta_test["error"] = np.abs(y_test.to_numpy() - xgb_preds)
        for h_step in range(1, horizon + 1):
            h_mae = float(meta_test[meta_test["horizon"] == h_step]["error"].mean())
            mlflow.log_metric(f"mae_h{h_step}", round(h_mae, 4))

        # SHAP Analysis on Holdout Set (for horizon == 24 or 48)
        logger.info("Computing SHAP explanations with TreeExplainer...")
        explainer = shap.TreeExplainer(xgb_model)
        sample_indices = np.random.choice(len(X_test), size=min(400, len(X_test)), replace=False)
        X_shap_sample = X_test.iloc[sample_indices]
        shap_values = explainer.shap_values(X_shap_sample)

        # 1. Global Summary Plot
        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(shap_values, X_shap_sample, show=False, max_display=12)
        plt.title(f"Global SHAP Feature Importance (XGBoost {horizon}h Forecast)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        shap_summary_path = os.path.join(FIGURES_DIR, f"shap_summary_{horizon}h.png")
        plt.savefig(shap_summary_path)
        plt.close()
        mlflow.log_artifact(shap_summary_path)

        # 2. Local Waterfall Plots (Sunny vs Cloudy day)
        # Find a sunny midday index (high power) and a cloudy midday index (low power, high cloud)
        test_daylight = X_test[(X_test["hour"] == 10) & (X_test["is_daylight"] == 1.0)]
        if len(test_daylight) > 0:
            sunny_idx = test_daylight.index[np.argmax(y_test.loc[test_daylight.index])]
            cloudy_idx = test_daylight.index[np.argmin(y_test.loc[test_daylight.index])]

            # Sunny Waterfall
            plt.figure(figsize=(9, 5), dpi=300)
            sunny_exp = shap.Explanation(
                values=explainer.shap_values(X_test.iloc[[sunny_idx]])[0],
                base_values=explainer.expected_value,
                data=X_test.iloc[sunny_idx].to_numpy(),
                feature_names=X_test.columns.tolist(),
            )
            shap.plots.waterfall(sunny_exp, max_display=10, show=False)
            plt.title(f"Local SHAP Explanation: High-Irradiance Clear Sky Hour ({meta_test.loc[sunny_idx, 'target_time']})", fontsize=11, fontweight="bold")
            plt.tight_layout()
            sunny_path = os.path.join(FIGURES_DIR, f"shap_waterfall_sunny_{horizon}h.png")
            plt.savefig(sunny_path)
            plt.close()
            mlflow.log_artifact(sunny_path)

            # Cloudy Waterfall
            plt.figure(figsize=(9, 5), dpi=300)
            cloudy_exp = shap.Explanation(
                values=explainer.shap_values(X_test.iloc[[cloudy_idx]])[0],
                base_values=explainer.expected_value,
                data=X_test.iloc[cloudy_idx].to_numpy(),
                feature_names=X_test.columns.tolist(),
            )
            shap.plots.waterfall(cloudy_exp, max_display=10, show=False)
            plt.title(f"Local SHAP Explanation: Attenuated / Cloud-Covered Hour ({meta_test.loc[cloudy_idx, 'target_time']})", fontsize=11, fontweight="bold")
            plt.tight_layout()
            cloudy_path = os.path.join(FIGURES_DIR, f"shap_waterfall_cloudy_{horizon}h.png")
            plt.savefig(cloudy_path)
            plt.close()
            mlflow.log_artifact(cloudy_path)

        # Save production bundle
        forecaster = SolarForecaster(
            model=xgb_model,
            feature_columns=X_train.columns.tolist(),
            system_capacity_kw=PV_SYSTEM_CAPACITY_KW,
            model_name=f"XGBoost_{horizon}h",
            metadata={"best_params": best_params, "metrics": xgb_metrics, "horizon": horizon},
        )
        prod_save_path = os.path.join(MODELS_DIR, f"solar_xgboost_{horizon}h.joblib")
        forecaster.save(prod_save_path)
        if horizon == 24:
            forecaster.save(os.path.join(MODELS_DIR, "solar_xgboost_pipeline.joblib"))

        mlflow.log_artifact(prod_save_path)
        mlflow_model_uri = f"runs:/{xgb_run.info.run_id}/model"
        mlflow.xgboost.log_model(xgb_model, artifact_path="model")

        results_summary["XGBoost"] = {
            "metrics": xgb_metrics,
            "preds": xgb_preds,
            "run_id": xgb_run.info.run_id,
            "model_uri": mlflow_model_uri,
        }
        logger.info(f"XGBoost -> MAE: {xgb_metrics['mae']} kW | RMSE: {xgb_metrics['rmse']} kW | sMAPE: {xgb_metrics['smape']}%")

    # ------------------------------------------------------------------------
    # MODEL 4: Compact PyTorch LSTM
    # ------------------------------------------------------------------------
    logger.info("Training Compact PyTorch LSTM...")
    with mlflow.start_run(run_name=f"Compact_LSTM_{horizon}h") as lstm_run:
        # Reshape data into (n_origins, horizon, n_features)
        n_origins_train = len(meta_train["origin"].unique())
        n_origins_test = len(meta_test["origin"].unique())

        X_train_blocks = X_train.to_numpy().reshape(n_origins_train, horizon, X_train.shape[1])
        y_train_blocks = y_train.to_numpy().reshape(n_origins_train, horizon)
        X_test_blocks = X_test.to_numpy().reshape(n_origins_test, horizon, X_test.shape[1])

        lstm_wrapper = LSTMForecasterWrapper(
            input_dim=X_train.shape[1],
            hidden_dim=48,
            num_layers=2,
            lr=0.003,
            epochs=35,
            batch_size=16,
        )
        lstm_wrapper.fit_blocks(X_train_blocks, y_train_blocks)
        raw_lstm_preds = lstm_wrapper.predict_blocks(X_test_blocks)

        # Physical post-processing
        lstm_preds = np.clip(raw_lstm_preds, 0.0, PV_SYSTEM_CAPACITY_KW)
        lstm_preds = np.where((is_daylight_test <= 0.0) | (elevation_test <= 0.0), 0.0, lstm_preds)

        lstm_metrics = calculate_metrics(y_test.to_numpy(), lstm_preds)

        mlflow.log_param("family", "DeepLearning")
        mlflow.log_param("model_type", "CompactLSTM")
        mlflow.log_param("hidden_dim", 48)
        mlflow.log_param("num_layers", 2)
        mlflow.log_param("horizon", horizon)
        for k, v in lstm_metrics.items():
            mlflow.log_metric(k, v)

        # Save LSTM bundle
        lstm_save_path = os.path.join(MODELS_DIR, f"solar_lstm_{horizon}h.joblib")
        joblib.dump(lstm_wrapper, lstm_save_path)
        mlflow.log_artifact(lstm_save_path)

        results_summary["Compact LSTM"] = {
            "metrics": lstm_metrics,
            "preds": lstm_preds,
            "run_id": lstm_run.info.run_id,
        }
        logger.info(f"Compact LSTM -> MAE: {lstm_metrics['mae']} kW | RMSE: {lstm_metrics['rmse']} kW | sMAPE: {lstm_metrics['smape']}%")

    # ------------------------------------------------------------------------
    # Generate Comparison & Horizon Figures
    # ------------------------------------------------------------------------
    _generate_comparison_plots(
        meta_test=meta_test,
        y_test=y_test.to_numpy(),
        results_summary=results_summary,
        horizon=horizon,
        save_dir=FIGURES_DIR,
    )

    return {
        "horizon": horizon,
        "results": results_summary,
        "best_model_uri": results_summary["XGBoost"]["model_uri"],
    }


def _generate_comparison_plots(
    meta_test: pd.DataFrame,
    y_test: np.ndarray,
    results_summary: Dict[str, Any],
    horizon: int,
    save_dir: str = FIGURES_DIR,
) -> None:
    """Generate horizon error degradation and actual vs forecast curves."""
    # 1. Horizon MAE Degradation Plot
    plt.figure(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    colors = {
        "Seasonal Persistence": "#7f7f7f",
        "Ridge Regression": "#1f77b4",
        "Compact LSTM": "#2ca02c",
        "XGBoost": "#d62728",
    }

    horizons_range = np.arange(1, horizon + 1)
    for model_name, res in results_summary.items():
        err = np.abs(y_test - res["preds"])
        meta_copy = meta_test.copy()
        meta_copy["err"] = err
        h_mae = meta_copy.groupby("horizon")["err"].mean().reindex(horizons_range)
        plt.plot(
            horizons_range,
            h_mae,
            label=f"{model_name} (Avg MAE: {res['metrics']['mae']:.2f} kW)",
            color=colors.get(model_name, "black"),
            linewidth=2.2,
            marker="o",
            markersize=4,
        )

    plt.title(f"Forecast Error vs Horizon Step (h=1..{horizon} hours)", fontsize=13, fontweight="bold")
    plt.xlabel("Forecast Horizon Lead Time (Hours Ahead)", fontsize=11)
    plt.ylabel("Mean Absolute Error (kW)", fontsize=11)
    plt.xticks(np.arange(1, horizon + 1, 2 if horizon == 24 else 4))
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"horizon_mae_comparison_{horizon}h.png"))
    plt.close()

    # 2. Time Series: Actual vs Forecast on Test Sample (e.g., 5 consecutive days in June)
    # Pick a 120-hour window from the test set
    sample_window = 120
    t_slice = slice(0, min(sample_window, len(y_test)))
    sample_times = meta_test["target_time"].iloc[t_slice]

    plt.figure(figsize=(13, 5), dpi=300)
    plt.plot(sample_times, y_test[t_slice], label="Actual PVGIS Power", color="black", linewidth=2.5, alpha=0.9)
    plt.plot(
        sample_times,
        results_summary["XGBoost"]["preds"][t_slice],
        label="XGBoost (Tuned)",
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
    )
    plt.plot(
        sample_times,
        results_summary["Compact LSTM"]["preds"][t_slice],
        label="Compact LSTM",
        color="#2ca02c",
        linewidth=1.8,
        linestyle="-.",
    )
    plt.plot(
        sample_times,
        results_summary["Ridge Regression"]["preds"][t_slice],
        label="Ridge Regression",
        color="#1f77b4",
        linewidth=1.5,
        alpha=0.7,
    )

    plt.title(f"Holdout Test Forecast vs Actual (June 2024, {horizon}h Multi-Horizon)", fontsize=13, fontweight="bold")
    plt.xlabel("Timestamp (UTC)", fontsize=11)
    plt.ylabel("PV Power (kW)", fontsize=11)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"forecast_vs_actual_test_{horizon}h.png"))
    plt.close()


def main():
    """Run full training for both 24h and 48h horizons."""
    logger.info("Initializing Solar Forecasting Training Suite...")

    # Run 24h
    res_24 = train_and_evaluate_solar_models(horizon=24, origin_step_hours=6, optuna_trials=15)

    # Run 48h
    res_48 = train_and_evaluate_solar_models(horizon=48, origin_step_hours=12, optuna_trials=12)

    logger.info("============================================================")
    logger.info("ALL TRAINING RUNS COMPLETED SUCCESSFULLY!")
    logger.info(f"24h Best Model URI: {res_24['best_model_uri']}")
    logger.info(f"48h Best Model URI: {res_48['best_model_uri']}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
