"""Inference from explicit real hourly history; no synthetic fallback inputs."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, HISTORY_HOURS, TARGETS, extract_features, validate_hourly

MODULE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = MODULE_DIR / "bundle" / "v1"


def load_bundle(bundle_dir=None):
    configured = bundle_dir or os.environ.get("CHARGESENTINEL_DEMAND_BUNDLE")
    path = Path(configured).expanduser() if configured else DEFAULT_BUNDLE
    if configured and not path.is_absolute():
        raise ValueError("Configured bundle path must be absolute")
    with (path / "metadata.json").open() as stream:
        metadata = json.load(stream)
    if metadata["features"] != FEATURES:
        raise ValueError("Bundle feature version mismatch")
    return path, metadata


def predict_demand(payload):
    if not isinstance(payload, dict) or set(payload) != {"forecast_origin", "history"}:
        raise ValueError("Required inputs: forecast_origin and real hourly history only")
    origin = pd.Timestamp(payload["forecast_origin"])
    if pd.isna(origin) or origin.tzinfo is not None or origin != origin.floor("h"):
        raise ValueError("forecast_origin must be a naive whole-hour operational timestamp")
    if not isinstance(payload["history"], list) or len(payload["history"]) < HISTORY_HOURS:
        raise ValueError(f"At least {HISTORY_HOURS} real consecutive hourly history records required")
    if any(set(row) != {"timestamp", *TARGETS} for row in payload["history"]):
        raise ValueError("History records require timestamp, energy_kwh, sessions")
    history = validate_hourly(pd.DataFrame(payload["history"]))
    if history.timestamp.iloc[-1] != origin - pd.Timedelta(hours=1):
        raise ValueError("History must end exactly one hour before forecast_origin; future history forbidden")
    history = history.tail(HISTORY_HOURS)
    # Placeholder target is never a feature: every demand feature is shifted >=1.
    target = pd.DataFrame([{"timestamp": origin, "energy_kwh": 0., "sessions": 0}])
    features = extract_features(pd.concat([history, target], ignore_index=True)).dropna(subset=FEATURES)
    path, meta = load_bundle()
    results = []
    shared_lstm = None
    for j, c in enumerate(TARGETS):
        family = meta["champions"][c]
        if family == "Baseline":
            profiles = json.loads((path / "baseline.json").read_text())
            row = features.iloc[-1]
            pred = .5 * row[f"{c}_lag_24"] + .5 * profiles[c][f"{int(row.dayofweek)}:{int(row.hour)}"]
        elif family == "XGBoost":
            from xgboost import XGBRegressor
            model = XGBRegressor()
            model.load_model(path / f"xgboost_{c}.json")
            pred = model.predict(features[FEATURES].tail(1))[0]
        elif family == "LSTM":
            import torch
            from .models import CompactLSTM
            if shared_lstm is None:
                scaling = json.loads((path / "scaling.json").read_text())
                x = (features[FEATURES].tail(24).to_numpy() - scaling["x_mean"]) / scaling["x_scale"]
                model = CompactLSTM(len(FEATURES))
                model.load_state_dict(torch.load(path / "lstm.pt", map_location="cpu", weights_only=True))
                model.eval()
                with torch.no_grad():
                    p = model(torch.tensor(x[None], dtype=torch.float32)).numpy()[0]
                shared_lstm = p * np.asarray(scaling["y_scale"]) + np.asarray(scaling["y_mean"])
            pred = shared_lstm[j]
        else:
            raise ValueError(f"Unsupported champion family: {family}")
        if not np.isfinite(pred):
            raise ValueError("Model produced a nonfinite prediction")
        results.append(float(max(0., pred)))
    return {"timestamp": origin.isoformat(), "expected_energy_kwh": results[0],
            "expected_sessions": results[1], "model_version": meta["model_version"]}
