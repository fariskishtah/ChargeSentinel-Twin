"""Score the saved duration champion and its training-only median baseline."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FeatureEncoder, split_by_time
from ..occupancy import load_sessions
from .inference import DEFAULT_BUNDLE_PATH, TEAM_DIR


REPORTS = TEAM_DIR / "reports"
FEATURES_PATH = TEAM_DIR / "artifacts" / "features_all.pkl"
DATA_PATH = TEAM_DIR.parents[1] / "data" / "processed" / "Charging_Data_clean.csv"


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, name: str) -> dict[str, float]:
    actual = frame.target_min.to_numpy(dtype=float)
    errors = np.asarray(prediction, dtype=float) - actual
    normal = errors[frame.is_abnormal.to_numpy() == 0]
    return {
        "model": name,
        "test_mae_normal": float(np.mean(np.abs(normal))),
        "test_rmse_normal": float(np.sqrt(np.mean(normal ** 2))),
        "test_median_ae_normal": float(np.median(np.abs(normal))),
        "test_mae_all": float(np.mean(np.abs(errors))),
        "test_rows": int(len(frame)),
        "test_normal_rows": int((frame.is_abnormal == 0).sum()),
    }


def _median_predict(frame: pd.DataFrame, baseline: dict) -> np.ndarray:
    by_charger_hour = {}
    raw_charger_hour = baseline["by_charger_hour"]
    if isinstance(raw_charger_hour, list):
        for row in raw_charger_hour:
            by_charger_hour[(str(row["charger_id"]), int(row["hour_bucket"]))] = float(row["median_duration_min"])
    else:
        by_charger_hour = {
            (str(key[0]), int(key[1])): float(value)
            for key, value in raw_charger_hour.items()
        }
    by_charger = {str(k): float(v) for k, v in baseline["by_charger"].items()}
    by_location = {str(k): float(v) for k, v in baseline["by_location"].items()}
    global_median = float(baseline["global_median"])
    bucket = pd.cut(frame.hour, [-1, 5, 10, 14, 18, 23], labels=False)
    result = []
    for charger, hour_bucket, location in zip(frame.charger_id, bucket, frame.location):
        prediction = by_charger_hour.get(
            (str(charger), int(hour_bucket)),
            by_charger.get(str(charger), by_location.get(str(location), global_median)),
        )
        result.append(prediction)
    return np.asarray(result, dtype=float)


def evaluate_saved_champion(bundle_path: str | Path = DEFAULT_BUNDLE_PATH) -> pd.DataFrame:
    bundle = joblib.load(bundle_path)
    if FEATURES_PATH.is_file():
        features = pd.read_pickle(FEATURES_PATH)
    else:
        from .features import build_features
        features = build_features(load_sessions())
    _, _, test = split_by_time(features)
    if test.empty:
        raise ValueError("The configured held-out test partition is empty")

    encoder = FeatureEncoder.from_state(bundle["encoder_state"])
    prediction = np.asarray(bundle["model"].predict(encoder.transform(test)), dtype=float)
    if bundle.get("target_mode", "log1p") == "log1p":
        prediction = np.expm1(prediction)
    cap = float(bundle.get("duration_cap_min", bundle.get("CAP")))
    prediction = np.clip(prediction, 0.0, cap)

    result = pd.DataFrame([
        _metrics(test, _median_predict(test, bundle["baseline"]), "Median baseline"),
        _metrics(test, prediction, bundle["model_name"]),
    ])
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "availability_duration_test_comparison.csv"
    result.to_csv(csv_path, index=False)

    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    champion = result.iloc[1]
    baseline = result.iloc[0]
    report = [
        "# Duration model held-out evaluation",
        "",
        f"Data: `{DATA_PATH.name}` ({len(features):,} cleaned rows; SHA-256 `{digest}`).",
        f"Chronological train and validation boundaries come from `src/features.py`; the held-out partition starts at 2021-10-01 and contains {len(test):,} rows.",
        "",
        "| Model | Normal test MAE (min) | Normal test RMSE (min) | All-session test MAE (min) |",
        "|---|---:|---:|---:|",
        f"| {baseline['model']} | {baseline['test_mae_normal']:.3f} | {baseline['test_rmse_normal']:.3f} | {baseline['test_mae_all']:.3f} |",
        f"| {champion['model']} | {champion['test_mae_normal']:.3f} | {champion['test_rmse_normal']:.3f} | {champion['test_mae_all']:.3f} |",
        "",
        f"The saved champion {'beats' if champion['test_mae_normal'] < baseline['test_mae_normal'] else 'does not beat'} the median baseline on normal-session held-out MAE.",
        "",
        "The test partition was scored before the Optuna tuning work. These results are retained for handoff transparency and are not a pristine untouched final estimate. Normal-session metrics exclude sessions later labelled abnormal; all-session MAE is included to show the effect of early fault-terminated sessions.",
        "",
        "Detailed metrics are in `availability_duration_test_comparison.csv`.",
    ]
    (REPORTS / "availability_duration_evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(evaluate_saved_champion().to_string(index=False))
