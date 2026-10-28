"""Reproducible chronological Jiaxing handoff; never uses session outcomes as inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from team.fatima_anomaly.integration.features import CATEGORICAL, FEATURES, raw_frame

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "bundle" / "anomaly-v1"
REPORT = ROOT / "reports" / "integration_hardening"
SOURCE_COMMIT = "e9e953f3d206067d3eeb9dd9821edb3dbe4ac2f5"


def score_set(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main(raw_path: str) -> None:
    raw = Path(raw_path)
    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    df = pd.read_csv(raw, encoding="cp1252", low_memory=False)
    start = pd.to_datetime(df["Start Time"].astype(str).str.strip(), errors="coerce")
    if start.isna().any() or not set(df["is_abnormal"].unique()).issubset({0, 1}):
        raise ValueError("Source timestamp or target invalid.")
    if not ((df["is_abnormal"] == 1) == (~df["end_cause"].isin(["Charging ends normally", "User stops charging"]))).all():
        raise ValueError("Outcome-label provenance changed.")
    order = np.argsort(start.to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    start = start.iloc[order].reset_index(drop=True)
    x = raw_frame(df)
    y = df["is_abnormal"].to_numpy(dtype=int)
    boundary1 = start.iloc[int(len(start) * 0.70)]
    boundary2 = start.iloc[int(len(start) * 0.85)]
    train = start < boundary1
    valid = (start >= boundary1) & (start < boundary2)
    test = start >= boundary2
    splits = {}
    for name, mask in (("train", train), ("validation", valid), ("test", test)):
        yy = y[mask.to_numpy()]
        splits[name] = {"first": start[mask].min().isoformat(), "last": start[mask].max().isoformat(),
                        "rows": int(len(yy)), "normal": int((yy == 0).sum()), "abnormal": int((yy == 1).sum()),
                        "prevalence": float(yy.mean())}
    model = CatBoostClassifier(iterations=450, depth=6, learning_rate=0.06, loss_function="Logloss",
                               eval_metric="AUC", auto_class_weights="Balanced", random_seed=42,
                               verbose=False, early_stopping_rounds=60, allow_writing_files=False, thread_count=4)
    model.fit(Pool(x[train], y[train.to_numpy()], cat_features=CATEGORICAL),
              eval_set=Pool(x[valid], y[valid.to_numpy()], cat_features=CATEGORICAL), use_best_model=True)
    val_scores = model.predict_proba(x[valid])[:, 1]
    val_y = y[valid.to_numpy()]
    thresholds = np.round(np.arange(0.05, 0.951, 0.025), 3)
    curve = []
    for t in thresholds:
        m = score_set(val_y, val_scores, float(t))
        curve.append({"threshold": float(t), "precision": m["precision"], "recall": m["recall"],
                      "f1": m["f1"], "fp": m["fp"], "fn": m["fn"]})
    eligible = [r for r in curve if r["precision"] >= splits["validation"]["prevalence"]]
    if not eligible:
        eligible = curve
    selected = max(eligible, key=lambda r: (5*r["precision"]*r["recall"]/(4*r["precision"]+r["recall"]+1e-12), r["f1"]))
    threshold = selected["threshold"]
    # Threshold is frozen before the held-out test scores are computed.
    test_scores = model.predict_proba(x[test])[:, 1]
    metrics = score_set(y[test.to_numpy()], test_scores, threshold)
    baseline_rates = pd.DataFrame({"location": x.loc[train, "location"], "y": y[train.to_numpy()]}).groupby("location")["y"].mean()
    baseline = x.loc[test, "location"].map(baseline_rates).fillna(splits["train"]["prevalence"]).to_numpy()
    baseline_metrics = score_set(y[test.to_numpy()], baseline, splits["train"]["prevalence"])
    BUNDLE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    model_path = BUNDLE / "catboost.cbm"
    model.save_model(str(model_path))
    config = {"model_version": "anomaly-v1", "feature_version": "start-safe-v1", "source_commit": SOURCE_COMMIT,
              "dataset_version": "sha256:" + source_hash, "inference_moment": "session start",
              "target": "abnormal end-cause outcome", "features": FEATURES, "categorical": CATEGORICAL,
              "threshold": threshold, "class_weighting": "Balanced", "best_iteration": int(model.best_iteration_),
              "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "splits": splits}
    (BUNDLE / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (REPORT / "metrics.json").write_text(json.dumps({"test": metrics, "baseline_location_rule_test": baseline_metrics,
                                                     "validation_selected": selected, "splits": splits}, indent=2) + "\n")
    pd.DataFrame(curve).to_csv(REPORT / "threshold_curve.csv", index=False)
    pd.DataFrame({"feature": FEATURES, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False).to_csv(REPORT / "feature_importance.csv", index=False)
    shap_sample = x.loc[test].head(512)
    shap_values = model.get_feature_importance(Pool(shap_sample, cat_features=CATEGORICAL), type="ShapValues")
    pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(np.asarray(shap_values)[:, :-1]).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).to_csv(REPORT / "shap_global_importance.csv", index=False)
    (REPORT / "confusion_matrix.json").write_text(json.dumps({k:metrics[k] for k in ("tn","fp","fn","tp")}, indent=2)+"\n")
    example_idx = int(np.flatnonzero(test.to_numpy())[0])
    row = x.iloc[example_idx]
    example = {"charger_id": str(row["charger_id"]), "features": {"location": str(row["location"]),
               "district": str(row["district"]), "start_time": start.iloc[example_idx].isoformat()}}
    (BUNDLE / "real_test_fixture.json").write_text(json.dumps(example, indent=2)+"\n")
    from app.api.schemas.models import AnomalyRequest, AnomalyResponse
    (BUNDLE / "input.schema.json").write_text(json.dumps(AnomalyRequest.model_json_schema(), indent=2)+"\n")
    (BUNDLE / "output.schema.json").write_text(json.dumps(AnomalyResponse.model_json_schema(), indent=2)+"\n")
    from team.fatima_anomaly.integration.inference import load_bundle, predict_anomaly
    load_bundle.cache_clear()
    explanation = predict_anomaly(example)
    (REPORT / "real_xai_example.json").write_text(json.dumps(explanation, indent=2)+"\n")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///tmp/chargesentinel-anomaly-mlruns"))
    mlflow.set_experiment("chargesentinel-anomaly")
    with mlflow.start_run(run_name="anomaly-v1-chronological-start-safe") as run:
        mlflow.set_tags({"module":"anomaly", "owner":"Fatima", "source_commit":SOURCE_COMMIT,
                         "integration_hardening":"Faris"})
        mlflow.log_params({"model_family":"CatBoost", "threshold":threshold, "feature_version":"start-safe-v1",
                           "dataset_version":config["dataset_version"], "train_end":splits["train"]["last"],
                           "validation_start":splits["validation"]["first"], "validation_end":splits["validation"]["last"],
                           "test_start":splits["test"]["first"], "class_weighting":"Balanced",
                           "iterations":450,"depth":6,"learning_rate":0.06,"best_iteration":config["best_iteration"]})
        mlflow.log_metrics({**metrics, "selected_threshold":threshold})
        mlflow.log_artifacts(str(BUNDLE), artifact_path="bundle")
        mlflow.log_artifacts(str(REPORT), artifact_path="reports")
        run_id=run.info.run_id
    (REPORT / "mlflow_run.json").write_text(json.dumps({"experiment":"chargesentinel-anomaly", "run_id":run_id,
                                                        "tracking_store":"local file store; not required for inference"}, indent=2)+"\n")
    print(json.dumps({"threshold": threshold, "metrics": metrics, "baseline": baseline_metrics,
                      "splits": splits, "mlflow_run": run_id}, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
