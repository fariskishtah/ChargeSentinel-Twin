"""Portable start-of-session risk inference and CatBoost SHAP explanation."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from team.fatima_anomaly.integration.features import CATEGORICAL, FEATURES, feature_record

BUNDLE = Path(__file__).resolve().parent.parent / "bundle" / "anomaly-v1"


@lru_cache(maxsize=2)
def load_bundle(bundle_dir: str = str(BUNDLE)):
    root = Path(bundle_dir)
    config = json.loads((root / "config.json").read_text())
    model_path = root / "catboost.cbm"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != config["model_sha256"]:
        raise ValueError("Anomaly model checksum mismatch.")
    if config["features"] != FEATURES or config["categorical"] != CATEGORICAL:
        raise ValueError("Anomaly feature contract mismatch.")
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    if list(model.feature_names_) != FEATURES:
        raise ValueError("Anomaly model feature mismatch.")
    return model, config


def predict_anomaly(payload: dict[str, object], bundle_dir: str = str(BUNDLE)) -> dict[str, object]:
    record = feature_record(str(payload["charger_id"]), payload["features"])
    model, config = load_bundle(bundle_dir)
    frame = pd.DataFrame([record], columns=FEATURES)
    score = float(model.predict_proba(frame)[0, 1])
    threshold = float(config["threshold"])
    anomalous = score >= threshold
    shap = np.asarray(model.get_feature_importance(Pool(frame, cat_features=CATEGORICAL), type="ShapValues"))[0, :-1]
    order = np.argsort(-shap if anomalous else shap)
    selected = [int(i) for i in order if (shap[i] > 0 if anomalous else shap[i] < 0)][:3]
    if not selected:
        selected = [int(order[0])]
    direction = "higher" if anomalous else "lower"
    reasons = [f"{FEATURES[i]} ({record[FEATURES[i]]}) contributed to {direction} estimated abnormal-end risk" for i in selected]
    return {"charger_id": record["charger_id"], "anomaly": bool(anomalous),
            "score": score, "threshold": threshold, "reason": reasons,
            "model_version": config["model_version"]}
