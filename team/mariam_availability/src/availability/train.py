"""Train, compare, and register charger session-duration models.

Run from this team's directory with:
    python -m src.availability.train --trials 25 --timeout 2700

The chronological partitions, feature pipeline, metrics, and selection rule are
shared with the notebooks. Optuna sees only rolling-origin folds inside train;
validation selects the champion and the test partition is scored once at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


def project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "data" / "processed" / "Charging_Data_clean.csv").is_file():
            return candidate
    raise FileNotFoundError("Missing data/processed/Charging_Data_clean.csv")


ROOT = project_root()
TEAM = ROOT / "team" / "mariam_availability"
sys.path.insert(0, str(TEAM))
from .features import FeatureEncoder, split_by_time  # noqa: E402


DATA_PATH = ROOT / "data" / "processed" / "Charging_Data_clean.csv"
ARTIFACTS = TEAM / "artifacts"
REPORTS = TEAM / "reports"
EXPERIMENT = "chargesentinel-availability"
MODEL_NAME = "availability_duration"
SEED = 42
CAP_QUANTILE = 0.995
MIN_TUNED_GAIN = 0.01
TARGET_MODE = "log1p"


def data_fingerprint(path: Path, n_rows: int, train_end: pd.Timestamp,
                     val_end: pd.Timestamp) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "data_file": path.name,
        "data_sha256": digest.hexdigest(),
        "n_rows": n_rows,
        "train_end": str(train_end),
        "val_end": str(val_end),
    }


def transformed_target(values: pd.Series | np.ndarray, cap: float) -> np.ndarray:
    return np.log1p(np.minimum(np.asarray(values, dtype=float), cap))


def inverse_target(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(values, dtype=float)))


def score(frame: pd.DataFrame, predictions: np.ndarray, split: str) -> dict[str, float]:
    actual = frame.target_min.to_numpy(dtype=float)
    errors = np.asarray(predictions, dtype=float) - actual
    normal_errors = errors[frame.is_abnormal.to_numpy() == 0]
    return {
        f"{split}_mae_normal": float(np.mean(np.abs(normal_errors))),
        f"{split}_rmse_normal": float(np.sqrt(np.mean(normal_errors ** 2))),
        f"{split}_median_ae_normal": float(np.median(np.abs(normal_errors))),
        f"{split}_mae_all": float(np.mean(np.abs(errors))),
    }


def baseline_model(train_normal: pd.DataFrame):
    """Fit a median lookup with charger/hour, charger, location, then global fallback."""
    hour_bucket = lambda hours: pd.cut(hours, [-1, 5, 10, 14, 18, 23], labels=False)
    fit = train_normal.copy()
    fit["hour_bucket"] = hour_bucket(fit.hour)
    by_charger_hour = fit.groupby(["charger_id", "hour_bucket"]).target_min.median().to_dict()
    by_charger = fit.groupby("charger_id").target_min.median().to_dict()
    by_location = fit.groupby("location").target_min.median().to_dict()
    global_median = float(fit.target_min.median())

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([
            by_charger_hour.get(
                (charger, bucket),
                by_charger.get(charger, by_location.get(location, global_median)),
            )
            for charger, bucket, location in zip(
                frame.charger_id, hour_bucket(frame.hour), frame.location
            )
        ], dtype=float)

    table = {
        "by_charger_hour": [
            {"charger_id": str(charger), "hour_bucket": int(bucket), "median_duration_min": float(value)}
            for (charger, bucket), value in by_charger_hour.items()
        ],
        "by_charger": {str(k): float(v) for k, v in by_charger.items()},
        "by_location": {str(k): float(v) for k, v in by_location.items()},
        "global_median": global_median,
    }
    return predict, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout < 1:
        parser.error("--trials and --timeout must be positive")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    features_path = ARTIFACTS / "features_all.pkl"
    feature_manifest = ARTIFACTS / "features_all.manifest.json"
    input_digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    cache_is_current = False
    if features_path.is_file() and feature_manifest.is_file():
        cached_manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
        cache_is_current = cached_manifest.get("data_sha256") == input_digest
    if cache_is_current:
        features = pd.read_pickle(features_path)
    else:
        from .features import build_features
        from src.occupancy import load_sessions
        features = build_features(load_sessions(DATA_PATH))
        features.to_pickle(features_path)
        feature_manifest.write_text(json.dumps({"data_sha256": input_digest,
                                                "n_rows": len(features)}, indent=2),
                                    encoding="utf-8")

    train, validation, test = split_by_time(features)
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological train/validation/test partitions must all contain rows")
    train_normal = train.loc[train.is_abnormal == 0].copy()
    validation_normal = validation.loc[validation.is_abnormal == 0].copy()
    if train_normal.empty or validation_normal.empty:
        raise ValueError("Train and validation need normal sessions for duration fitting")

    encoder = FeatureEncoder().fit(train_normal)
    x_train = encoder.transform(train_normal)
    x_validation_normal = encoder.transform(validation_normal)
    x_validation = encoder.transform(validation)
    x_test = encoder.transform(test)
    cap = float(train_normal.target_min.quantile(CAP_QUANTILE))
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Could not derive a positive duration cap from training data")
    fingerprint = data_fingerprint(DATA_PATH, len(features), train.start.min().normalize(),
                                   validation.start.min().normalize())

    tracking_env = os.getenv("CHARGESENTINEL_MLFLOW_TRACKING_URI")
    tracking_uri = tracking_env or ("sqlite:///" + (ARTIFACTS / "mlflow.db").resolve().as_posix())
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(EXPERIMENT)
    tags = {
        "data_sha256": fingerprint["data_sha256"],
        "feature_version": "start_known_features_v1",
        "target_mode": TARGET_MODE,
        "test_set_previously_scored": "true",
        "prior_test_metrics_file": "notebooks/reports/duration_test_metrics.csv",
    }

    def log_common(family: str, role: str = "validation") -> None:
        mlflow.set_tags({**tags, "model_family": family, "split_role": role})

    def log_model(flavor, model, example, trusted_types=None):
        # XGBoost 3.0.5 with the platform's scikit-learn version can fit and
        # predict, but its sklearn wrapper cannot serialize via save_model.
        # Log the native Booster while the portable serving bundle retains the
        # fitted wrapper and train-fitted encoder.
        if flavor is mlflow.xgboost and isinstance(model, xgb.XGBModel):
            model = model.get_booster()
        options = inspect.signature(flavor.log_model).parameters
        path_arg = "name" if "name" in options else "artifact_path"
        kwargs = {path_arg: "model", "input_example": example}
        if trusted_types and "skops_trusted_types" in options:
            kwargs["skops_trusted_types"] = trusted_types
        return flavor.log_model(model, **kwargs)

    baseline_predict, baseline_table = baseline_model(train_normal)
    baseline_predictions = baseline_predict(validation)
    baseline_metrics = score(validation, baseline_predictions, "validation")
    with mlflow.start_run(run_name="baseline"):
        log_common("baseline")
        mlflow.log_params({"method": "median_charger_hour_with_fallbacks", "hour_bucket_edges": "-1,5,10,14,18,23"})
        mlflow.log_metrics(baseline_metrics)
        mlflow.log_dict(baseline_table, "baseline_lookup_tables.json")

    rf_params = {
        "n_estimators": 200, "min_samples_leaf": 5, "max_features": 0.5,
        "max_samples": 0.5, "n_jobs": -1, "random_state": args.seed,
    }
    rf = RandomForestRegressor(**rf_params)
    rf.fit(x_train, transformed_target(train_normal.target_min, cap))
    rf_metrics = score(validation, inverse_target(rf.predict(x_validation)), "validation")
    with mlflow.start_run(run_name="random_forest"):
        log_common("random_forest")
        mlflow.log_params({**rf_params, "target_mode": TARGET_MODE, "duration_cap_min": cap})
        mlflow.log_metrics(rf_metrics)
        log_model(mlflow.sklearn, rf, x_validation_normal.head(5),
                  trusted_types=["sklearn.tree._tree.Tree"])

    default_params = {
        "n_estimators": 3000, "learning_rate": 0.05, "max_depth": 8,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "hist", "n_jobs": -1, "random_state": args.seed,
        "early_stopping_rounds": 100, "eval_metric": "mae",
    }
    default_model = xgb.XGBRegressor(**default_params)
    default_model.fit(
        x_train, transformed_target(train_normal.target_min, cap),
        eval_set=[(x_validation_normal, transformed_target(validation_normal.target_min, cap))],
        verbose=False,
    )
    default_metrics = score(validation, inverse_target(default_model.predict(x_validation)), "validation")
    with mlflow.start_run(run_name="xgboost_default"):
        log_common("xgboost_default")
        mlflow.log_params({**default_params, "target_mode": TARGET_MODE, "duration_cap_min": cap})
        mlflow.log_metrics(default_metrics)
        mlflow.set_tag("best_iteration", str(default_model.best_iteration))
        log_model(mlflow.xgboost, default_model, x_validation_normal.head(5))

    # Rolling-origin folds are chronological subsets of the training partition.
    ordered = train_normal.sort_values("start").index.to_numpy()
    folds = list(TimeSeriesSplit(n_splits=3).split(ordered))
    cv_folds = [(ordered[tr], ordered[va]) for tr, va in folds]
    for tr, va in cv_folds:
        if train_normal.loc[tr].start.max() > train_normal.loc[va].start.min():
            raise AssertionError("Optuna fold violates chronological ordering")

    def objective(trial: optuna.Trial) -> float:
        sampled = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        fold_maes, best_iterations = [], []
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            log_common("xgboost_optuna_trial", "time_ordered_cv")
            mlflow.log_params({**sampled, "trial_number": trial.number,
                               "n_estimators": 2000, "early_stopping_rounds": 50,
                               "target_mode": TARGET_MODE, "duration_cap_min": cap})
            for fold_idx, (tr, va) in enumerate(cv_folds, 1):
                fold_train, fold_valid = train_normal.loc[tr], train_normal.loc[va]
                fold_encoder = FeatureEncoder().fit(fold_train)
                x_tr, x_va = fold_encoder.transform(fold_train), fold_encoder.transform(fold_valid)
                model = xgb.XGBRegressor(
                    n_estimators=2000, early_stopping_rounds=50, tree_method="hist",
                    n_jobs=-1, random_state=args.seed, eval_metric="mae", **sampled,
                )
                model.fit(x_tr, transformed_target(fold_train.target_min, cap),
                          eval_set=[(x_va, transformed_target(fold_valid.target_min, cap))],
                          verbose=False)
                pred = inverse_target(model.predict(x_va))
                fold_mae = float(np.mean(np.abs(pred - fold_valid.target_min.to_numpy())))
                best_iteration = int(model.best_iteration)
                fold_maes.append(fold_mae)
                best_iterations.append(best_iteration)
                trial.set_user_attr(f"fold_{fold_idx}_mae", fold_mae)
                mlflow.log_metrics({f"fold_{fold_idx}_mae_minutes": fold_mae,
                                    f"fold_{fold_idx}_best_iteration": best_iteration})
            mean_mae = float(np.mean(fold_maes))
            trial.set_user_attr("median_best_iteration", int(np.median(best_iterations)))
            mlflow.log_metric("mean_cv_mae_minutes", mean_mae)
            mlflow.log_metric("median_best_iteration", float(np.median(best_iterations)))
        return mean_mae

    with mlflow.start_run(run_name="optuna_xgboost"):
        log_common("optuna_study", "training_period_cv")
        mlflow.log_params({"sampler": "TPESampler", "seed": args.seed,
                           "max_trials": args.trials, "timeout_seconds": args.timeout,
                           "folds": 3, "objective": "mean_time_ordered_fold_mae_min"})
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout, gc_after_trial=True)
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("Optuna completed no trials")
        study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
            REPORTS / "availability_xgboost_optuna_trials.csv", index=False)
        joblib.dump(study, ARTIFACTS / "availability_xgboost_optuna_study.joblib")
        mlflow.log_artifact(str(REPORTS / "availability_xgboost_optuna_trials.csv"), "optuna")
        mlflow.log_dict({"best_params": study.best_params, "best_cv_mae_minutes": study.best_value,
                         "completed_trials": len(completed), "timeout_seconds": args.timeout},
                        "optuna_best_result.json")

    tuned_params = {
        **study.best_params, "n_estimators": 3000, "tree_method": "hist", "n_jobs": -1,
        "random_state": args.seed, "early_stopping_rounds": 100, "eval_metric": "mae",
    }
    tuned_model = xgb.XGBRegressor(**tuned_params)
    tuned_model.fit(x_train, transformed_target(train_normal.target_min, cap),
                    eval_set=[(x_validation_normal, transformed_target(validation_normal.target_min, cap))],
                    verbose=False)
    tuned_metrics = score(validation, inverse_target(tuned_model.predict(x_validation)), "validation")
    with mlflow.start_run(run_name="xgboost_tuned"):
        log_common("xgboost_tuned")
        mlflow.log_params({**tuned_params, "target_mode": TARGET_MODE, "duration_cap_min": cap})
        mlflow.log_metrics(tuned_metrics)
        mlflow.set_tag("best_iteration", str(tuned_model.best_iteration))
        log_model(mlflow.xgboost, tuned_model, x_validation_normal.head(5))

    candidates = [
        ("Baseline", baseline_metrics), ("Random Forest", rf_metrics),
        ("XGBoost default", default_metrics), ("XGBoost tuned", tuned_metrics),
    ]
    comparison = pd.DataFrame([{"model": name, **metrics} for name, metrics in candidates])
    comparison_path = REPORTS / "availability_duration_validation_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    default_mae = default_metrics["validation_mae_normal"]
    tuned_mae = tuned_metrics["validation_mae_normal"]
    gain = (default_mae - tuned_mae) / default_mae
    if gain >= MIN_TUNED_GAIN:
        champion_name, champion_model, champion_params = "xgboost_tuned", tuned_model, tuned_params
    else:
        champion_name, champion_model, champion_params = "xgboost_default", default_model, default_params

    fingerprint.update({"train_end": str(train.start.max()), "val_end": str(validation.start.max())})
    fingerprint["data_sha256_12"] = fingerprint["data_sha256"][:12]
    selection = {
        "selection_metric": "normal-session validation MAE (minutes)",
        "minimum_tuned_gain_fraction": MIN_TUNED_GAIN,
        "default_xgboost_validation_mae_normal": default_mae,
        "tuned_xgboost_validation_mae_normal": tuned_mae,
        "tuned_relative_gain_fraction": gain,
        "champion": champion_name,
        "test_set_previously_scored_before_tuning": True,
        "prior_test_metrics_file": "notebooks/reports/duration_test_metrics.csv",
        "data_version": fingerprint,
    }
    (REPORTS / "availability_duration_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8")

    bundle_path = ARTIFACTS / "availability_duration_bundle_optuna_mlflow.joblib"
    bundle = {
        "model": champion_model, "model_name": champion_name,
        "encoder_state": encoder.get_state(), "feature_names": encoder.feature_names,
        "target_mode": TARGET_MODE, "duration_cap_min": cap,
        "split_dates": {"train_end": str(train.start.max()), "validation_end": str(validation.start.max())},
        "data_version": fingerprint, "baseline": baseline_table,
        "validation_metrics": comparison.to_dict("records"),
        "bundle_version": f"availability-duration-{champion_name}-v1@{fingerprint['data_sha256_12']}",
    }
    joblib.dump(bundle, bundle_path)

    # As in the existing experiment, the test partition has already been scored
    # during earlier notebook work. The disclosure is retained on each run.
    test_predictions = inverse_target(champion_model.predict(x_test))
    test_metrics = score(test, test_predictions, "test")
    baseline_test_metrics = score(test, baseline_predict(test), "test")
    test_path = REPORTS / "availability_duration_champion_test_metrics.csv"
    pd.DataFrame([{"model": champion_name, **test_metrics}]).to_csv(test_path, index=False)
    test_comparison_path = REPORTS / "availability_duration_test_comparison.csv"
    pd.DataFrame([
        {"model": "Median baseline", **baseline_test_metrics},
        {"model": champion_name, **test_metrics},
    ]).to_csv(test_comparison_path, index=False)
    with mlflow.start_run(run_name="champion") as run:
        log_common(champion_name, "validation_selected_test_scored_once_prior_test_score_disclosed")
        mlflow.set_tag("prior_test_evaluation_before_tuning", "true")
        mlflow.log_params({**champion_params, "selected_model": champion_name,
                           "target_mode": TARGET_MODE, "duration_cap_min": cap,
                           "minimum_tuned_gain_fraction": MIN_TUNED_GAIN})
        selected_metrics = default_metrics if champion_name == "xgboost_default" else tuned_metrics
        mlflow.log_metrics({**selected_metrics, **test_metrics, "tuned_relative_gain_fraction": gain})
        model_info = log_model(mlflow.xgboost, champion_model, x_validation_normal.head(5))
        mlflow.log_artifact(str(bundle_path), "bundle")
        mlflow.log_artifact(str(comparison_path), "evaluation")
        mlflow.log_artifact(str(test_path), "evaluation")
        mlflow.log_artifact(str(test_comparison_path), "evaluation")
        mlflow.log_dict({"test_set_previously_scored_before_tuning": True,
                         "prior_test_metrics_file": "notebooks/reports/duration_test_metrics.csv",
                         "this_run_test_scoring_count": 1, "champion": champion_name,
                         "metrics": test_metrics}, "test_evaluation_disclosure.json")
        client = MlflowClient()
        registered = mlflow.register_model(model_info.model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "champion", str(registered.version))

    handoff = {
        "registered_model_uri": f"models:/{MODEL_NAME}@champion",
        "champion_run_id": run.info.run_id,
        "tracking_uri": "sqlite:///team/mariam_availability/artifacts/mlflow.db (local; override with CHARGESENTINEL_MLFLOW_TRACKING_URI)",
        "experiment_name": EXPERIMENT,
        "bundle_path": "team/mariam_availability/artifacts/availability_duration_bundle_optuna_mlflow.joblib (local generated artifact)",
        "validation_comparison_file": "team/mariam_availability/reports/availability_duration_validation_comparison.csv",
        "champion_test_metrics_file": "team/mariam_availability/reports/availability_duration_champion_test_metrics.csv",
        "test_comparison_file": "team/mariam_availability/reports/availability_duration_test_comparison.csv",
        "selection_report_file": "team/mariam_availability/reports/availability_duration_selection.json",
        "data_version": fingerprint,
        "prior_test_scoring_disclosed": True,
        "prior_test_metrics_file": "notebooks/reports/duration_test_metrics.csv",
        "test_metrics_are_for_selected_champion_only": True,
    }
    (REPORTS / "availability_duration_mlflow_handoff.json").write_text(
        json.dumps(handoff, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Selected champion: {champion_name}; test metrics: {test_metrics}")
    print(f"Registered model: models:/{MODEL_NAME}@champion")
    print(f"MLflow tracking: {tracking_uri}")


if __name__ == "__main__":
    main()
