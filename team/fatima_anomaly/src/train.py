
"""Train and evaluate the EV charging anomaly detection model."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from team.fatima_anomaly.src.features import TARGET, prepare_features

RANDOM_STATE = 42
THRESHOLD = 0.41
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = ARTIFACT_DIR / "catboost_anomaly_model.cbm"
CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and validate the target."""
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. "
            "Set CHARGING_DATA_PATH to the CSV location."
        )

    last_error = None
    df = None

    for encoding in ["utf-8", "cp1252", "latin1", "utf-8-sig"]:
        try:
            df = pd.read_csv(
                csv_path,
                encoding=encoding,
                low_memory=False,
            )
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if df is None:
        raise ValueError(f"Could not decode the CSV: {last_error}")

    df = df.loc[:, ~df.columns.duplicated()].copy()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[TARGET]).copy()

    if not set(df[TARGET].unique()).issubset(
        {0, 1, 0.0, 1.0, True, False}
    ):
        mapping = {
            "normal": 0,
            "abnormal": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
        }
        mapped = df[TARGET].astype(str).str.strip().str.lower().map(
            mapping
        )

        if mapped.isna().any():
            raise ValueError(
                f"Unsupported target values: {df[TARGET].unique()}"
            )

        df[TARGET] = mapped.astype(int)
    else:
        df[TARGET] = df[TARGET].astype(int)

    if df[TARGET].nunique() != 2:
        raise ValueError("The target must contain both classes: 0 and 1.")

    return df


def main() -> None:
    data_path = os.getenv("CHARGING_DATA_PATH")

    if not data_path:
        raise ValueError(
            "Set CHARGING_DATA_PATH to the location of "
            "Charging_Data.csv before training."
        )

    df = load_data(data_path)

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(df[TARGET].value_counts().sort_index())
    print("Abnormal rate:", round(df[TARGET].mean(), 4))

    # Keep identifiers for auditing, but never use them as model features.
    id_names = {
        "charger_id",
        "charger id",
        "charging_station_id",
        "station_id",
        "session_id",
        "transaction_id",
    }

    id_col = next(
        (
            col for col in df.columns
            if col.lower().strip() in id_names
            and "charger" in col.lower()
        ),
        None,
    )

    if id_col is None:
        id_col = next(
            (
                col for col in df.columns
                if col.lower().strip() in id_names
            ),
            None,
        )

    X = prepare_features(df)
    y = df[TARGET].copy()

    # Remove empty and constant columns consistently before splitting.
    X = X.dropna(axis=1, how="all")
    constant_cols = [
        col for col in X.columns
        if X[col].nunique(dropna=False) <= 1
    ]
    X = X.drop(columns=constant_cols)

    if X.empty:
        raise ValueError("No usable features remain after preprocessing.")

    if y.value_counts().min() < 3:
        raise ValueError(
            "At least three records per class are required "
            "for stratified train/validation/test splits."
        )

    feature_names = X.columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    relative_test_size = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print(
        f"Split sizes — train: {len(X_train)}, "
        f"validation: {len(X_valid)}, test: {len(X_test)}"
    )

    train_pool = Pool(
        X_train,
        y_train,
        cat_features=categorical_features,
    )
    valid_pool = Pool(
        X_valid,
        y_valid,
        cat_features=categorical_features,
    )

    model = CatBoostClassifier(
        iterations=1600,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=100,
        early_stopping_rounds=120,
        allow_writing_files=False,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )

    valid_probs = model.predict_proba(X_valid)[:, 1]
    valid_preds = (valid_probs >= THRESHOLD).astype(int)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_preds, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "CatBoostClassifier",
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "best_iteration": int(model.best_iteration_),
        "validation_precision": float(
            precision_score(y_valid, valid_preds, zero_division=0)
        ),
        "validation_recall": float(
            recall_score(y_valid, valid_preds, zero_division=0)
        ),
        "validation_f1": float(
            f1_score(y_valid, valid_preds, zero_division=0)
        ),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_precision": float(
            precision_score(y_test, test_preds, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, test_preds, zero_division=0)
        ),
        "test_f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
        "test_pr_auc": float(average_precision_score(y_test, test_probs)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "test_rows": int(len(X_test)),
        "feature_count": len(feature_names),
        "excluded_constant_features": constant_cols,
        "features": feature_names,
        "categorical_features": categorical_features,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))

    config = {
        "model_type": "CatBoostClassifier",
        "target_column": TARGET,
        "decision_threshold": THRESHOLD,
        "positive_class": 1,
        "random_state": RANDOM_STATE,
        "model_version": datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ),
        "model_artifact": MODEL_PATH.name,
        "features": feature_names,
        "categorical_features": categorical_features,
        "excluded_constant_features": constant_cols,
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(IMPORTANCE_PATH, index=False)

    print("\nTest results")
    for name in [
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
    ]:
        print(f"{name}: {metrics[name]:.4f}")

    print("\nSaved model:", MODEL_PATH)
    print("Saved configuration:", CONFIG_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved feature importance:", IMPORTANCE_PATH)


if __name__ == "__main__":
    main()
