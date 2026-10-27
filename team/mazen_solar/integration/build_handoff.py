"""Select a causal solar champion on May validation; evaluate June once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from team.mazen_solar.integration.source_features import SolarFeaturePipeline, PV_SYSTEM_CAPACITY_KW, NWP_WEATHER_FEATURES

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "bundle" / "solar-v1"
REPORT = ROOT / "reports" / "integration_hardening"
SOURCE_COMMIT = "bfc088839fce1229c9f3045ab44e6ffedc0d1c37"


def read_source(pvgis_path: str, weather_path: str):
    pvgis_bytes = Path(pvgis_path).read_bytes()
    weather_bytes = Path(weather_path).read_bytes()
    pv = json.loads(pvgis_bytes)
    weather = json.loads(weather_bytes)
    if weather["utc_offset_seconds"] != 0:
        raise ValueError("Expected UTC Open-Meteo hour index.")
    df_weather = pd.DataFrame(weather["hourly"])
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    df_weather = df_weather.set_index("time").sort_index()
    power = pd.Series(pv["power"][:len(df_weather)], index=df_weather.index, dtype=float) / 1000.0
    if len(power) != 4368 or int(power.isna().sum()) != 2:
        raise ValueError("PVGIS H1 source shape changed.")
    return df_weather, power, hashlib.sha256(pvgis_bytes).hexdigest(), hashlib.sha256(weather_bytes).hexdigest()


def main(pvgis_path: str, weather_path: str) -> None:
    weather, pv, pv_hash, weather_hash = read_source(pvgis_path, weather_path)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    history_path = BUNDLE / "pvgis_h1_history.csv"
    pd.DataFrame({"timestamp": pv.index.strftime("%Y-%m-%dT%H:%M:%S"), "power_kw": pv.to_numpy()}).to_csv(history_path, index=False, float_format="%.17g")
    history_hash = hashlib.sha256(history_path.read_bytes()).hexdigest()
    weather_path_out = BUNDLE / "openmeteo_previous_runs_h1.csv"
    weather[NWP_WEATHER_FEATURES].rename_axis("timestamp").to_csv(weather_path_out, float_format="%.17g")
    weather_out_hash = hashlib.sha256(weather_path_out.read_bytes()).hexdigest()
    summary = {"source_commit": SOURCE_COMMIT, "source_pvgis_sha256": pv_hash,
               "source_openmeteo_sha256": weather_hash, "history_sha256": history_hash,
               "weather_sha256": weather_out_hash,
               "source_rows": len(pv), "missing_target_hours": [t.isoformat() for t in pv[pv.isna()].index],
               "timestamp_alignment": "PVGIS HH:09 readings aligned by ordinal hour to UTC Open-Meteo H1 labels; only labels strictly before origin used as history",
               "selection": "lowest May 2024 chronological validation MAE; June test evaluated after selection", "horizons": {}}
    pipeline = SolarFeaturePipeline()
    mlflow.set_tracking_uri("file:///tmp/chargesentinel-solar-mlruns")
    mlflow.set_experiment("chargesentinel-solar")
    for horizon, step in ((24, 6), (48, 12)):
        x, y, meta = pipeline.build_dataset_from_raw(weather, pv, horizon=horizon,
            origin_step_hours=step, start_date="2024-01-22 00:00:00", end_date="2024-06-30 23:00:00")
        origin = pd.to_datetime(meta["origin"])
        target = pd.to_datetime(meta["target_time"])
        train = (origin < pd.Timestamp("2024-05-01")) & (target < pd.Timestamp("2024-05-01"))
        valid = (origin >= pd.Timestamp("2024-05-01")) & (origin < pd.Timestamp("2024-06-01")) & (target < pd.Timestamp("2024-06-01"))
        test = origin >= pd.Timestamp("2024-06-01")
        if min(train.sum(), valid.sum(), test.sum()) == 0:
            raise ValueError("Empty chronological solar split.")
        families = {
            "Seasonal Persistence": None,
            "Ridge Regression": Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
            "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.06,
                                     subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=4),
        }
        results = {}
        for name, model in families.items():
            if model is None:
                predict = lambda mask: x.loc[mask, "pv_seasonal_persistence"].to_numpy()
            else:
                model.fit(x.loc[train], y.loc[train])
                predict = lambda mask, fitted=model: fitted.predict(x.loc[mask])
            val_pred = np.clip(predict(valid), 0, PV_SYSTEM_CAPACITY_KW)
            val_pred = np.where(x.loc[valid, "is_daylight"].to_numpy() > 0, val_pred, 0.0)
            results[name] = {"validation_mae_kw": float(mean_absolute_error(y.loc[valid], val_pred))}
        champion = min(results, key=lambda name: results[name]["validation_mae_kw"])
        if champion == "Seasonal Persistence":
            test_pred = x.loc[test, "pv_seasonal_persistence"].to_numpy()
        elif champion == "Ridge Regression":
            test_pred = families[champion].predict(x.loc[test])
        else:
            raise ValueError(f"Unbundled solar champion {champion}; do not force a different family.")
        test_pred = np.clip(test_pred, 0, PV_SYSTEM_CAPACITY_KW)
        test_pred = np.where(x.loc[test, "is_daylight"].to_numpy() > 0, test_pred, 0.0)
        test_mae = float(mean_absolute_error(y.loc[test], test_pred))
        artifact = {"model_version":f"solar-{champion.lower().replace(' ', '-')}-{horizon}h-v1", "family":champion,
                    "horizon_hours":horizon, "capacity_kw":PV_SYSTEM_CAPACITY_KW,
                    "lag_rule":"h<24: target-24h; 24<=h<48: target-48h; h=48: target-72h; all < origin",
                    "history_file":history_path.name, "history_sha256":history_hash,
                    "source_commit":SOURCE_COMMIT, "features":["pv_seasonal_persistence"] if champion == "Seasonal Persistence" else list(x.columns),
                    "weather_required":champion == "Ridge Regression", "validation_mae_kw":results[champion]["validation_mae_kw"],
                    "test_mae_kw":test_mae}
        if champion == "Ridge Regression":
            fitted = families[champion]
            artifact["weather_file"] = weather_path_out.name
            artifact["weather_sha256"] = weather_out_hash
            artifact["ridge"] = {"mean":fitted.named_steps["scale"].mean_.tolist(),
                                  "scale":fitted.named_steps["scale"].scale_.tolist(),
                                  "coef":fitted.named_steps["ridge"].coef_.tolist(),
                                  "intercept":float(fitted.named_steps["ridge"].intercept_)}
        artifact_path = BUNDLE / f"solar_{horizon}h.json"
        artifact_path.write_text(json.dumps(artifact, indent=2)+"\n")
        summary["horizons"][str(horizon)] = {"artifact":artifact_path.name,
            "sha256":hashlib.sha256(artifact_path.read_bytes()).hexdigest(), "champion":champion,
            "families":results, "test_mae_kw":test_mae,
            "rows":{"train":int(train.sum()), "validation":int(valid.sum()), "test":int(test.sum())},
            "origins":{"train":int(origin[train].nunique()), "validation":int(origin[valid].nunique()),
                       "test":int(origin[test].nunique())},
            "date_ranges":{"train":[str(origin[train].min()),str(origin[train].max())],
                           "validation":[str(origin[valid].min()),str(origin[valid].max())],
                           "test":[str(origin[test].min()),str(origin[test].max())]}}
        with mlflow.start_run(run_name=artifact["model_version"]) as run:
            mlflow.set_tags({"module":"solar", "owner":"Mazen", "source_commit":SOURCE_COMMIT,
                             "integration_hardening":"Faris"})
            mlflow.log_params({"model_family":champion,"horizon_hours":horizon,
                               "selection":"May chronological validation MAE", "missing_target_handling":"exclude"})
            mlflow.log_metrics({"validation_mae_kw":results[champion]["validation_mae_kw"],
                                "test_mae_kw":test_mae, "train_rows":int(train.sum()),
                                "validation_rows":int(valid.sum()), "test_rows":int(test.sum())})
            mlflow.log_artifact(str(artifact_path))
            summary["horizons"][str(horizon)]["mlflow_run_id"] = run.info.run_id
    (BUNDLE / "checksums.json").write_text(json.dumps({"solar_24h.json":summary["horizons"]["24"]["sha256"],
        "solar_48h.json":summary["horizons"]["48"]["sha256"],
        history_path.name:history_hash, weather_path_out.name:weather_out_hash}, indent=2)+"\n")
    from app.api.schemas.models import SolarForecastRequest, SolarForecastResponse
    (BUNDLE / "input.schema.json").write_text(json.dumps(SolarForecastRequest.model_json_schema(), indent=2)+"\n")
    (BUNDLE / "output.schema.json").write_text(json.dumps(SolarForecastResponse.model_json_schema(), indent=2)+"\n")
    (REPORT / "selection.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
