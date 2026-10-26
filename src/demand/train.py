"""
Training and Experiment Tracking Pipeline for Person 2 (EV Demand Forecasting).

Implements:
1. Strict chronological time-aware train/val/test splits.
2. Three distinct model families:
   - Family 1: Seasonal / Persistence Baseline
   - Family 2: XGBoost Regressor (with Optuna hyperparameter optimization)
   - Family 3: Compact LSTM (PyTorch Sequence-to-One Network)
3. Evaluates both targets separately:
   - Target 1: Next-hour expected_energy_kwh
   - Target 2: Next-hour expected_sessions
4. Evaluates on MAE, RMSE, sMAPE, and R2.
5. Full MLflow tracking with SQLite backend, logging params, metrics, artifacts, and champion registration.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["MPLCONFIGDIR"] = r"c:\Users\WASEEM\AppData\Local\Temp"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.demand.features import (
    extract_features_df,
    DemandFeaturePipeline,
    FEATURE_COLUMNS,
    TARGET_COLUMNS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demand_train")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE) in percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    # Avoid zero division when both true and pred are 0
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask] / denominator[mask]) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard metrics: MAE, RMSE, sMAPE, R2."""
    y_pred_clipped = np.clip(y_pred, a_min=0.0, a_max=None)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(root_mean_squared_error(y_true, y_pred_clipped))
    smape = float(calculate_smape(y_true, y_pred_clipped))
    r2 = float(r2_score(y_true, y_pred_clipped))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "sMAPE": round(smape, 4),
        "R2": round(r2, 4)
    }


# ==========================================
# 1. BASELINE MODEL: Seasonal Persistence
# ==========================================
class SeasonalPersistenceBaseline:
    """
    Baseline model that combines same-hour-yesterday lag (lag-24)
    with historical median adjustment by (hour, dayofweek).
    """
    def __init__(self):
        self.energy_profile = {}
        self.sessions_profile = {}
        self.default_energy = 0.0
        self.default_sessions = 0.0
        
    def fit(self, df_train: pd.DataFrame):
        self.default_energy = float(df_train["target_energy_kwh"].median())
        self.default_sessions = float(df_train["target_sessions"].median())
        
        # Profile by (target_dayofweek, target_hour)
        ep = df_train.groupby(["target_dayofweek", "target_hour"])["target_energy_kwh"].median()
        sp = df_train.groupby(["target_dayofweek", "target_hour"])["target_sessions"].median()
        self.energy_profile = ep.to_dict()
        self.sessions_profile = sp.to_dict()
        return self
        
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        preds_e = []
        preds_s = []
        for _, row in df.iterrows():
            dow = int(row["target_dayofweek"])
            hr = int(row["target_hour"])
            # Blend 50% lag_24h and 50% seasonal calendar median
            profile_e = self.energy_profile.get((dow, hr), self.default_energy)
            profile_s = self.sessions_profile.get((dow, hr), self.default_sessions)
            
            lag_e = row.get("energy_lag_24h", profile_e)
            lag_s = row.get("sessions_lag_24h", profile_s)
            
            e_pred = 0.5 * lag_e + 0.5 * profile_e
            s_pred = 0.5 * lag_s + 0.5 * profile_s
            preds_e.append(max(0.0, float(e_pred)))
            preds_s.append(max(0.0, float(s_pred)))
            
        return np.array(preds_e), np.array(preds_s)


# ==========================================
# 2. COMPACT LSTM (PyTorch)
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.seq_len = seq_len
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X) - self.seq_len
        
    def __getitem__(self, idx):
        # Sequence from idx to idx+seq_len
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx + self.seq_len]
        )


class CompactLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, output_dim: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


def train_compact_lstm(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 12,
    batch_size: int = 64,
    seq_len: int = 24
) -> Tuple[CompactLSTM, Dict[str, Any]]:
    """Train compact LSTM on past 24h multivariate sequence."""
    logger.info("Training Compact LSTM (PyTorch)...")
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train, seq_len=seq_len)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = CompactLSTM(input_dim=X_train_scaled.shape[1], hidden_dim=32, num_layers=1, output_dim=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    best_loss = float("inf")
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
        val_loss /= len(val_dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"Compact LSTM trained successfully. Best Val Loss: {best_loss:.4f}")
    return model, {"seq_len": seq_len, "best_val_loss": best_loss}


def predict_lstm(model: CompactLSTM, X_scaled: np.ndarray, seq_len: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    dataset = TimeSeriesDataset(X_scaled, np.zeros((len(X_scaled), 2)), seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    preds_list = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_preds = model(batch_x).numpy()
            preds_list.append(batch_preds)
    all_preds = np.vstack(preds_list)
    # The first seq_len predictions are padded with mean of predictions
    pad = np.repeat(all_preds[:1], seq_len, axis=0)
    full_preds = np.vstack([pad, all_preds])
    return np.clip(full_preds[:, 0], a_min=0.0, a_max=None), np.clip(full_preds[:, 1], a_min=0.0, a_max=None)


# ==========================================
# 3. XGBOOST WITH OPTUNA TUNING
# ==========================================
def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target_name: str,
    n_trials: int = 12
) -> Dict[str, Any]:
    """Run Optuna tuning inside training/validation folds only."""
    logger.info(f"Starting Optuna tuning for XGBoost ({target_name}, {n_trials} trials)...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_preds = model.predict(X_val)
        return mean_absolute_error(y_val, np.clip(val_preds, a_min=0.0, a_max=None))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"Optuna Best Params for {target_name}: {study.best_params} (Best Val MAE: {study.best_value:.4f})")
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1})
    return best_params


# ==========================================
# 4. MASTER ORCHESTRATOR & EVALUATION
# ==========================================
def run_demand_training_pipeline(
    data_path: str = "data/processed/hourly_demand.csv",
    mlflow_db: str = "sqlite:///mlflow.db",
    experiment_name: str = "ev_demand_forecasting",
    n_optuna_trials: int = 15
) -> Dict[str, Any]:
    """Execute complete end-to-end training and tracking pipeline."""
    logger.info(f"Loading processed hourly series from: {data_path}")
    raw_df = pd.read_csv(data_path)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    
    # Extract leakage-safe features
    df_features = extract_features_df(raw_df, include_targets=True)
    # Drop rows with NaN from initial lag window (168h) and final row
    df_clean = df_features.dropna().reset_index(drop=True)
    logger.info(f"Clean feature matrix prepared: {df_clean.shape}")
    
    # Chronological Time-Aware Splits:
    # Train: 2020-01-08 to 2021-06-30
    # Val: 2021-07-01 to 2021-09-30 (Used for Optuna & model selection)
    # Test: 2021-10-01 to 2021-12-31 (Strict held-out test set)
    train_mask = df_clean["timestamp"] < "2021-07-01 00:00:00"
    val_mask = (df_clean["timestamp"] >= "2021-07-01 00:00:00") & (df_clean["timestamp"] < "2021-10-01 00:00:00")
    test_mask = df_clean["timestamp"] >= "2021-10-01 00:00:00"
    
    df_train = df_clean[train_mask].copy()
    df_val = df_clean[val_mask].copy()
    df_test = df_clean[test_mask].copy()
    
    logger.info(f"Train set: {len(df_train)} rows ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
    logger.info(f"Val set:   {len(df_val)} rows ({df_val['timestamp'].min()} to {df_val['timestamp'].max()})")
    logger.info(f"Test set:  {len(df_test)} rows ({df_test['timestamp'].min()} to {df_test['timestamp'].max()})")
    
    X_train = df_train[FEATURE_COLUMNS]
    y_train_e = df_train["target_energy_kwh"]
    y_train_s = df_train["target_sessions"]
    
    X_val = df_val[FEATURE_COLUMNS]
    y_val_e = df_val["target_energy_kwh"]
    y_val_s = df_val["target_sessions"]
    
    X_test = df_test[FEATURE_COLUMNS]
    y_test_e = df_test["target_energy_kwh"]
    y_test_s = df_test["target_sessions"]
    
    # Fit Feature Pipeline Scaler strictly on train set
    pipeline = DemandFeaturePipeline(feature_columns=FEATURE_COLUMNS)
    pipeline.fit(df_train)
    joblib.dump(pipeline, "models/feature_pipeline.pkl")
    
    X_train_scaled = pipeline.transform(df_train)
    X_val_scaled = pipeline.transform(df_val)
    X_test_scaled = pipeline.transform(df_test)
    
    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_db)
    mlflow.set_experiment(experiment_name)
    
    results = {}
    
    # ----------------------------------------------------
    # RUN 1: BASELINE (Seasonal Persistence)
    # ----------------------------------------------------
    with mlflow.start_run(run_name="Seasonal_Persistence_Baseline") as baseline_run:
        logger.info("Training Family 1: Seasonal Persistence Baseline...")
        baseline = SeasonalPersistenceBaseline()
        baseline.fit(df_train)
        joblib.dump(baseline, "models/baseline_model.pkl")
        
        # Predict on Test
        base_test_e, base_test_s = baseline.predict(df_test)
        base_metrics_e = evaluate_predictions(y_test_e.values, base_test_e)
        base_metrics_s = evaluate_predictions(y_test_s.values, base_test_s)
        
        mlflow.log_params({"model_family": "Baseline", "strategy": "Lag24_plus_CalendarMedian"})
        for k, v in base_metrics_e.items():
            mlflow.log_metric(f"test_energy_{k}", v)
        for k, v in base_metrics_s.items():
            mlflow.log_metric(f"test_sessions_{k}", v)
            
        results["Baseline"] = {
            "energy": base_metrics_e,
            "sessions": base_metrics_s,
            "run_id": baseline_run.info.run_id
        }
        
    # ----------------------------------------------------
    # RUN 2: COMPACT LSTM (PyTorch)
    # ----------------------------------------------------
    with mlflow.start_run(run_name="Compact_LSTM_PyTorch") as lstm_run:
        logger.info("Training Family 2: Compact LSTM...")
        y_train_combined = np.column_stack([y_train_e.values, y_train_s.values])
        y_val_combined = np.column_stack([y_val_e.values, y_val_s.values])
        
        lstm_model, lstm_info = train_compact_lstm(
            X_train_scaled, y_train_combined,
            X_val_scaled, y_val_combined,
            epochs=12, batch_size=64, seq_len=24
        )
        torch.save(lstm_model.state_dict(), "models/lstm_model.pt")
        
        lstm_test_e, lstm_test_s = predict_lstm(lstm_model, X_test_scaled, seq_len=24)
        lstm_metrics_e = evaluate_predictions(y_test_e.values, lstm_test_e)
        lstm_metrics_s = evaluate_predictions(y_test_s.values, lstm_test_s)
        
        mlflow.log_params({"model_family": "LSTM", "hidden_dim": 32, "num_layers": 1, "seq_len": 24})
        for k, v in lstm_metrics_e.items():
            mlflow.log_metric(f"test_energy_{k}", v)
        for k, v in lstm_metrics_s.items():
            mlflow.log_metric(f"test_sessions_{k}", v)
            
        results["LSTM"] = {
            "energy": lstm_metrics_e,
            "sessions": lstm_metrics_s,
            "run_id": lstm_run.info.run_id
        }

    # ----------------------------------------------------
    # RUN 3: XGBOOST (Optuna Tuned - CHAMPION)
    # ----------------------------------------------------
    with mlflow.start_run(run_name="XGBoost_Optuna_Champion") as xgb_run:
        logger.info("Training Family 3: XGBoost (Optuna Optimized)...")
        # Optuna tuning on training/validation folds
        best_params_e = optimize_xgboost(X_train, y_train_e, X_val, y_val_e, "energy_kwh", n_trials=n_optuna_trials)
        best_params_s = optimize_xgboost(X_train, y_train_s, X_val, y_val_s, "sessions", n_trials=n_optuna_trials)
        
        # Fit final XGBoost models
        model_e = xgb.XGBRegressor(**best_params_e)
        model_e.fit(X_train, y_train_e, eval_set=[(X_val, y_val_e)], verbose=False)
        model_e.save_model("models/xgboost_energy.json")
        
        model_s = xgb.XGBRegressor(**best_params_s)
        model_s.fit(X_train, y_train_s, eval_set=[(X_val, y_val_s)], verbose=False)
        model_s.save_model("models/xgboost_sessions.json")
        
        xgb_test_e = np.clip(model_e.predict(X_test), 0.0, None)
        xgb_test_s = np.clip(model_s.predict(X_test), 0.0, None)
        
        xgb_metrics_e = evaluate_predictions(y_test_e.values, xgb_test_e)
        xgb_metrics_s = evaluate_predictions(y_test_s.values, xgb_test_s)
        
        mlflow.log_params({"model_family": "XGBoost", **{f"xgb_e_{k}": v for k, v in best_params_e.items()}})
        for k, v in xgb_metrics_e.items():
            mlflow.log_metric(f"test_energy_{k}", v)
        for k, v in xgb_metrics_s.items():
            mlflow.log_metric(f"test_sessions_{k}", v)
            
        mlflow.xgboost.log_model(model_e, artifact_path="model_energy")
        mlflow.xgboost.log_model(model_s, artifact_path="model_sessions")
        
        champion_uri = f"runs:/{xgb_run.info.run_id}/model_energy"
        
        results["XGBoost"] = {
            "energy": xgb_metrics_e,
            "sessions": xgb_metrics_s,
            "run_id": xgb_run.info.run_id,
            "mlflow_model_uri": champion_uri
        }
        
        # ----------------------------------------------------
        # GENERATE EVALUATION CHARTS
        # ----------------------------------------------------
        # 1. Prediction vs Actual Forecast Curves (Holdout Test 168-hour sample)
        fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
        sample_hours = 168  # 1 week zoom
        t_sample = df_test["timestamp"].iloc[:sample_hours]
        
        axes[0].plot(t_sample, y_test_e.iloc[:sample_hours], label="Actual Energy (kWh)", color="black", linewidth=2.0)
        axes[0].plot(t_sample, xgb_test_e[:sample_hours], label="XGBoost Forecast", color="#1f77b4", linestyle="--", linewidth=2.0)
        axes[0].plot(t_sample, base_test_e[:sample_hours], label="Baseline Forecast", color="#7f7f7f", linestyle=":", alpha=0.8)
        axes[0].set_title("EV Energy Demand (kWh) - Next-Hour Forecast vs Ground Truth (Holdout Test Week)")
        axes[0].set_ylabel("Energy (kWh)")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        axes[1].plot(t_sample, y_test_s.iloc[:sample_hours], label="Actual Sessions Count", color="black", linewidth=2.0)
        axes[1].plot(t_sample, xgb_test_s[:sample_hours], label="XGBoost Forecast", color="#2ca02c", linestyle="--", linewidth=2.0)
        axes[1].plot(t_sample, base_test_s[:sample_hours], label="Baseline Forecast", color="#7f7f7f", linestyle=":", alpha=0.8)
        axes[1].set_title("EV Charging Sessions Count - Next-Hour Forecast vs Ground Truth (Holdout Test Week)")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_ylabel("Sessions Count")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("reports/figures/holdout_forecast_comparison.png", dpi=150)
        plt.close()
        
        # 2. Residuals Distribution
        res_e = y_test_e.values - xgb_test_e
        res_s = y_test_s.values - xgb_test_s
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(res_e, bins=40, color="#1f77b4", edgecolor="black", alpha=0.7)
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title(f"Energy Residuals (MAE: {xgb_metrics_e['MAE']} kWh)")
        axes[0].set_xlabel("Prediction Error (Actual - Predicted)")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        axes[1].hist(res_s, bins=40, color="#2ca02c", edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set_title(f"Sessions Residuals (MAE: {xgb_metrics_s['MAE']} sessions)")
        axes[1].set_xlabel("Prediction Error (Actual - Predicted)")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig("reports/figures/residuals_distribution.png", dpi=150)
        plt.close()
        
        # 3. Feature Importance Plot
        importance_e = model_e.feature_importances_
        idx_sorted = np.argsort(importance_e)[-15:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(idx_sorted)), importance_e[idx_sorted], color="#1f77b4", align="center")
        plt.yticks(range(len(idx_sorted)), [FEATURE_COLUMNS[i] for i in idx_sorted])
        plt.xlabel("XGBoost Relative Feature Importance")
        plt.title("Top 15 Feature Importances for Next-Hour EV Energy Demand")
        plt.tight_layout()
        plt.savefig("reports/figures/feature_importance.png", dpi=150)
        plt.close()
        
        # Log artifacts to MLflow
        mlflow.log_artifact("reports/figures/holdout_forecast_comparison.png")
        mlflow.log_artifact("reports/figures/residuals_distribution.png")
        mlflow.log_artifact("reports/figures/feature_importance.png")
        mlflow.log_artifact("models/feature_pipeline.pkl")

    # Save champion metadata
    champion_metadata = {
        "champion_model": "XGBoost",
        "energy_model_file": "models/xgboost_energy.json",
        "sessions_model_file": "models/xgboost_sessions.json",
        "feature_pipeline_file": "models/feature_pipeline.pkl",
        "mlflow_model_uri": champion_uri,
        "feature_columns": FEATURE_COLUMNS,
        "comparison_results": results
    }
    with open("models/champion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(champion_metadata, f, indent=2)
        
    logger.info("Training pipeline finished successfully! Model and evaluation artifacts saved.")
    return champion_metadata


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_demand_training_pipeline(n_optuna_trials=trials)
