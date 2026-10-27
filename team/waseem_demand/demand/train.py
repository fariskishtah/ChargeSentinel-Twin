"""Bounded CPU training, validation-only selection, and one final test evaluation."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .build_hourly import build_hourly
from .features import FEATURES, TARGETS, FEATURE_VERSION, extract_features, split_masks, sequence_arrays

MODULE = Path(__file__).resolve().parents[1]
EXPERIMENT = "chargesentinel-demand"
SEED = 42


def metrics(actual, predicted):
    p = np.maximum(0., np.asarray(predicted))
    a = np.asarray(actual)
    denominator = np.abs(a) + np.abs(p)
    ratios = np.divide(200 * np.abs(a-p), denominator, out=np.zeros_like(a, dtype=float), where=denominator != 0)
    return {"MAE": float(np.abs(a-p).mean()), "RMSE": float(np.sqrt(np.square(a-p).mean())),
            "sMAPE": float(ratios.mean())}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_scalers(x, y, train_mask):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit(x[train_mask]), StandardScaler().fit(y[train_mask])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--log-only", action="store_true", help="Resume tracking from frozen bundle; do not retrain")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.log_only:
        log_runs(json.loads((MODULE / "bundle/v1/metadata.json").read_text()),
                 MODULE / "bundle/v1", MODULE / "reports", MODULE / "runtime")
        return
    if args.raw is None:
        parser.error("--raw is required for training")
    if args.trials < 1 or args.epochs < 1:
        parser.error("trials and epochs must be positive")
    import torch
    import optuna
    import mlflow
    import mlflow.xgboost
    from xgboost import XGBRegressor
    from .models import CompactLSTM, fit_baseline, baseline_predict
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bundle = MODULE / "bundle" / "v1"
    reports = MODULE / "reports"
    runtime = MODULE / "runtime"
    for path in (bundle, reports, runtime):
        path.mkdir(parents=True, exist_ok=True)
    (reports / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    hourly, audit = build_hourly(args.raw)
    save_json(reports / "data_audit.json", audit)
    save_json(reports / "eda_summary.json", {
        "distributions": hourly[TARGETS].describe(percentiles=[.01, .5, .95, .99]).to_dict(),
        "correlation": float(hourly.energy_kwh.corr(hourly.sessions)),
        "hourly_seasonality": hourly.groupby(hourly.timestamp.dt.hour)[TARGETS].mean().to_dict(),
        "lag24_energy_autocorrelation": float(hourly.energy_kwh.autocorr(24)),
        "lag168_energy_autocorrelation": float(hourly.energy_kwh.autocorr(168))})
    # Derived full data stays local. The only committed data is a small inference fixture.
    hourly.to_csv(runtime / "hourly.csv", index=False)
    d = extract_features(hourly).dropna(subset=FEATURES).reset_index(drop=True)
    x, y = d[FEATURES].to_numpy(), d[TARGETS].to_numpy()
    masks = split_masks(d.timestamp)
    sx, sy = fit_scalers(x, y, masks["train"])
    xs, ys = sx.transform(x), sy.transform(y)
    seq_x, seq_y = sequence_arrays(xs, ys)
    # All families use exactly the same target rows, including first validation/test hour.
    eval_frame = d.iloc[23:].reset_index(drop=True)
    eval_masks = split_masks(eval_frame.timestamp)
    truth = eval_frame[TARGETS].to_numpy()
    train_seq, val_seq = eval_masks["train"], eval_masks["validation"]
    profiles = fit_baseline(d[masks["train"]])
    save_json(bundle / "baseline.json", profiles)
    predictions = {"Baseline": baseline_predict(eval_frame, profiles)}
    tuning, xgb_models, parameters = {}, [], {}
    for j, target in enumerate(TARGETS):
        def objective(trial):
            params = {"n_estimators": trial.suggest_int("n_estimators", 100, 240, step=70),
                      "max_depth": trial.suggest_int("max_depth", 3, 6),
                      "learning_rate": trial.suggest_float("learning_rate", .025, .12, log=True),
                      "subsample": .9, "colsample_bytree": .9, "reg_lambda": 2.,
                      "random_state": SEED, "n_jobs": 2, "tree_method": "hist"}
            model = XGBRegressor(**params).fit(x[masks["train"]], y[masks["train"], j])
            p = model.predict(x[masks["validation"]])
            return metrics(y[masks["validation"], j], p)["MAE"]
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=args.trials)
        params = {**study.best_params, "subsample": .9, "colsample_bytree": .9,
                  "reg_lambda": 2., "random_state": SEED, "n_jobs": 2, "tree_method": "hist"}
        model = XGBRegressor(**params).fit(x[masks["train"]], y[masks["train"], j])
        model.save_model(bundle / f"xgboost_{target}.json")
        xgb_models.append(model)
        parameters[target] = params
        tuning[target] = {"trials": args.trials, "validation_MAE": study.best_value,
                          "best_params": params, "trials_detail": [
                              {"number": t.number, "params": t.params, "value": t.value} for t in study.trials]}
    # Predictions are computed once with frozen models; test is never used to fit/select.
    predictions["XGBoost"] = np.column_stack([m.predict(x[23:]) for m in xgb_models])
    net = CompactLSTM(len(FEATURES))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=1e-4)
    train_x = torch.tensor(seq_x[train_seq], dtype=torch.float32)
    train_y = torch.tensor(seq_y[train_seq], dtype=torch.float32)
    val_x = torch.tensor(seq_x[val_seq], dtype=torch.float32)
    val_y = torch.tensor(seq_y[val_seq], dtype=torch.float32)
    best_loss, best_state, best_epoch = float("inf"), None, None
    epoch_history = []
    for epoch in range(args.epochs):
        net.train()
        for offset in range(0, len(train_x), 128):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(net(train_x[offset:offset+128]), train_y[offset:offset+128])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(torch.nn.functional.mse_loss(net(val_x), val_y))
        epoch_history.append({"epoch": epoch+1, "validation_scaled_MSE": val_loss})
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, copy.deepcopy(net.state_dict()), epoch+1
        print(f"LSTM epoch {epoch+1}/{args.epochs}: validation MSE {val_loss:.5f}", flush=True)
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), bundle / "lstm.pt")
    net.eval()
    with torch.no_grad():
        output = np.concatenate([net(torch.tensor(seq_x[i:i+256], dtype=torch.float32)).numpy()
                                 for i in range(0, len(seq_x), 256)])
    predictions["LSTM"] = sy.inverse_transform(output)
    vm = eval_masks["validation"]
    results = {family: {"validation": {c: metrics(truth[vm, j], pred[vm, j])
                                      for j, c in enumerate(TARGETS)}}
               for family, pred in predictions.items()}
    champions = {c: min(results, key=lambda family: results[family]["validation"][c]["MAE"])
                 for c in TARGETS}
    tm = eval_masks["test"]
    for family, pred in predictions.items():
        results[family]["test"] = {c: metrics(truth[tm, j], pred[tm, j]) for j, c in enumerate(TARGETS)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=MODULE, text=True).strip()
    meta = {"model_version": "demand-completion-v1", "feature_version": FEATURE_VERSION,
            "dataset_version": "sha256:" + audit["raw_sha256"], "training_code_commit": commit,
            "features": FEATURES, "champions": champions,
            "selection": "Lowest validation MAE per target; test is reporting only",
            "aggregation": "Completed-session energy recognized at End Time hour; arrivals at Start Time hour",
            "forecast_origin": "First boundary of target hour; history ends at origin minus one hour",
            "expected_sessions_type": "Nonnegative float statistical expected count",
            "splits": {s: {"rows": int(m.sum()), "first": str(eval_frame.timestamp[m].min()),
                            "last": str(eval_frame.timestamp[m].max())} for s, m in eval_masks.items()},
            "results": results, "optuna": tuning, "lstm": {"epochs": args.epochs, "best_epoch": best_epoch,
            "best_validation_scaled_MSE": best_loss, "history": epoch_history}}
    save_json(bundle / "metadata.json", meta)
    save_json(bundle / "scaling.json", {"x_mean": sx.mean_.tolist(), "x_scale": sx.scale_.tolist(),
               "y_mean": sy.mean_.tolist(), "y_scale": sy.scale_.tolist(),
               "fit_rows": int(masks["train"].sum()), "fit_last_target": str(d.timestamp[masks["train"]].max())})
    save_json(reports / "optuna.json", tuning)
    save_json(reports / "metrics.json", results)
    save_json(bundle / "features.json", FEATURES)
    # A real, contiguous 192-hour history before the first held-out prediction.
    origin = pd.Timestamp("2021-10-01")
    fixture_history = hourly[hourly.timestamp < origin].tail(192).copy()
    fixture_history["timestamp"] = fixture_history.timestamp.map(lambda v: v.isoformat())
    fixture = {"forecast_origin": origin.isoformat(), "history": fixture_history.to_dict("records")}
    save_json(MODULE / "fixtures" / "request.json", fixture)
    from .inference import predict_demand
    save_json(MODULE / "fixtures" / "expected_response.json", predict_demand(fixture))
    make_plots(hourly, eval_frame, predictions, truth, eval_masks["test"], reports)
    write_report(meta, audit, reports)
    receipt = log_runs(meta, bundle, reports, runtime)
    from .build_hourly import sha256
    assert sha256(args.raw) == audit["raw_sha256"], "Raw data changed"
    print(json.dumps({"champions": champions, "results": results, "mlflow": receipt}, indent=2), flush=True)


def log_runs(meta, bundle, reports, runtime):
    """Track the frozen, already-evaluated bundle without needing raw data or retraining."""
    (reports / "tracking_environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    import torch
    import mlflow
    import mlflow.xgboost
    from xgboost import XGBRegressor
    from .models import CompactLSTM
    from .build_hourly import sha256
    commit, results = meta["training_code_commit"], meta["results"]
    profiles = json.loads((bundle / "baseline.json").read_text())
    fixture = json.loads((MODULE / "fixtures/request.json").read_text())
    target = {"timestamp": fixture["forecast_origin"], "energy_kwh": 0., "sessions": 0}
    d = extract_features(pd.DataFrame([*fixture["history"], target])).dropna(subset=FEATURES)
    scaling = json.loads((bundle / "scaling.json").read_text())
    seq_x = ((d[FEATURES].tail(24).to_numpy() - scaling["x_mean"]) / scaling["x_scale"])[None]
    net = CompactLSTM(len(FEATURES))
    net.load_state_dict(torch.load(bundle / "lstm.pt", map_location="cpu", weights_only=True))
    net.eval()
    xgb_models = []
    for target_name in TARGETS:
        model = XGBRegressor()
        model.load_model(bundle / f"xgboost_{target_name}.json")
        xgb_models.append(model)
    parameters = {c: meta["optuna"][c]["best_params"] for c in TARGETS}
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///" + str(runtime / "mlflow.db"))
    mlflow.set_tracking_uri(tracking_uri)
    if not os.environ.get("MLFLOW_TRACKING_URI") and mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(runtime / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    run_ids = {}
    for family in results:
        with mlflow.start_run(run_name=f"repaired-{family}") as run:
            mlflow.set_tags({"owner": "Waseem", "module": "demand", "git_commit": commit})
            params = {"model_name": family, "dataset_version": meta["dataset_version"],
                      "feature_version": FEATURE_VERSION, "seed": SEED}
            if family == "XGBoost":
                params.update({f"{c}_{k}": v for c in TARGETS for k, v in parameters[c].items()})
                params["optuna_trials_per_target"] = meta["optuna"]["energy_kwh"]["trials"]
            elif family == "LSTM":
                params.update({"hidden_dim": 32, "sequence_length": 24, "epochs": meta["lstm"]["epochs"],
                               "best_epoch": meta["lstm"]["best_epoch"], "learning_rate": .002, "batch_size": 128})
            else:
                params["strategy"] = "50% lag24 + 50% training hour/dayofweek median"
            mlflow.log_params(params)
            for split in ("validation", "test"):
                for c in TARGETS:
                    for metric, value in results[family][split][c].items():
                        mlflow.log_metric(f"{split}_{c}_{metric}", value)
            mlflow.log_artifacts(str(bundle), artifact_path="bundle")
            mlflow.log_artifacts(str(reports), artifact_path="evaluation")
            mlflow.log_artifacts(str(MODULE / "schemas"), artifact_path="schemas")
            if family == "XGBoost":
                for j, c in enumerate(TARGETS):
                    mlflow.xgboost.log_model(xgb_models[j], name=f"model_{c}",
                                             input_example=d[FEATURES].head(2))
            elif family == "LSTM":
                import mlflow.pytorch
                traced = torch.jit.trace(net, torch.tensor(seq_x[:2], dtype=torch.float32))
                mlflow.pytorch.log_model(traced, name="scaled_sequence_model", input_example=seq_x[:2].astype(np.float32))
            else:
                import mlflow.pyfunc
                def seasonal_for_mlflow(model_input):
                    return np.column_stack([
                        .5 * model_input[f"{c}_lag_24"].to_numpy() + .5 * np.array([
                            profiles[c][f"{int(d)}:{int(h)}"] for d, h in zip(model_input.dayofweek, model_input.hour)])
                        for c in ["energy_kwh", "sessions"]])
                mlflow.pyfunc.log_model(name="seasonal_model",
                    python_model=seasonal_for_mlflow,
                    input_example=d[FEATURES].head(2))
            run_ids[family] = {"run_id": run.info.run_id,
                               "bundle_artifact_uri": f"runs:/{run.info.run_id}/bundle"}
    receipt = {"experiment": EXPERIMENT, "runs": run_ids,
               "champion_artifact_uri": run_ids["XGBoost"]["bundle_artifact_uri"],
               "portable_artifact": "bundle/v1", "tracking_store": "runtime/mlflow.db (local, excluded from git)"}
    save_json(reports / "mlflow_receipt.json", receipt)
    with (reports / "evaluation.md").open("a") as stream:
        stream.write("\n## MLflow\n\nExperiment: `chargesentinel-demand`. Real runs and bundle URIs:\n\n")
        for family, info in run_ids.items():
            stream.write(f"- {family}: `{info['bundle_artifact_uri']}`\n")
        stream.write("\nThe SQLite store remains local under `runtime/`. These URIs require that store; "
                     "the committed `bundle/v1` is the portable handoff. No remote registry is claimed.\n")
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    for info in run_ids.values():
        client.log_artifact(info["run_id"], str(reports / "evaluation.md"), "evaluation")
        client.log_artifact(info["run_id"], str(reports / "mlflow_receipt.json"), "evaluation")
    checksums = {p.name: sha256(p) for p in bundle.iterdir() if p.is_file() and p.name != "checksums.json"}
    save_json(bundle / "checksums.json", checksums)
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


def make_plots(hourly, frame, predictions, truth, test, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].hist(hourly.energy_kwh, bins=50)
    axes[0, 0].set(title="Completed energy distribution", xlabel="kWh / completion hour")
    hourly.groupby(hourly.timestamp.dt.hour)[TARGETS].mean().plot(ax=axes[0, 1], secondary_y="sessions")
    axes[0, 1].set_title("Mean hourly seasonality")
    hourly.groupby(hourly.timestamp.dt.month).energy_kwh.mean().plot(ax=axes[1, 0])
    axes[1, 0].set_title("Monthly mean completed energy")
    axes[1, 1].scatter(hourly.sessions, hourly.energy_kwh, s=1, alpha=.1)
    axes[1, 1].set(title="Arrival / completion-energy correlation", xlabel="Arrivals", ylabel="Completed energy kWh")
    fig.tight_layout(); fig.savefig(reports / "eda.png", dpi=120); plt.close(fig)
    indices = np.flatnonzero(test)[:168]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for j, c in enumerate(TARGETS):
        axes[j].plot(frame.timestamp.iloc[indices], truth[indices, j], label="Actual", color="black")
        for family, p in predictions.items():
            axes[j].plot(frame.timestamp.iloc[indices], np.maximum(0, p[indices, j]), label=family, alpha=.7)
        axes[j].set_ylabel(c); axes[j].legend(ncol=4)
    fig.tight_layout(); fig.savefig(reports / "holdout.png", dpi=120); plt.close(fig)


def write_report(meta, audit, reports):
    lines = ["# Repaired EV demand evaluation", "", "## Data audit and target semantics", "",
             f"{audit['raw_rows']:,} raw rows; {audit['hourly_rows']:,} continuous hourly rows. "
             f"Coverage: {audit['first_hour']} to {audit['last_hour']}. No valid event hours are discarded.",
             "", "Energy is recognized in each session's completion hour, when final session energy becomes "
             "observable. Session count is arrivals in the start hour. All energy and arrivals are conserved. "
             "This replaces the old start-hour energy simplification and is not physical intra-session grid load.",
             "", "No raw rows are altered. Full schema, missingness, duplicate and identifier checks, source "
             "SHA256, coverage and zero counts are in `data_audit.json`. UserID is not a transaction ID. "
             "No recorded events are treated as zero under the assumption of complete capture; telemetry outages are unknown.",
             "", "Source timezone not independently verified. Local/naive operational time is retained, never relabeled UTC.",
             "", "## Features and leakage controls", "",
             "Calendar hour/day-of-week/weekend/month and cyclic hour/day encodings; energy/session lags "
             "1, 2, 3, 24, 48, 168; shifted mean/std over 6 and 24 hours. Every row at target hour u uses "
             "demand through u-1 only. No weather, TOU price, fault flags or post-event outcome fields enter features. "
             "Completion energy is used only after its completion hour closes. No current target enters its features.",
             "", "Weather is excluded because its observed daily aggregates were unavailable at origin. "
             "TOU is optional and excluded without a pricing provenance/ablation claim.",
             "", "## Chronological evaluation", "",
             "Train targets precede 2021-07-01; validation is July–September 2021; test begins 2021-10-01. "
             "Target timestamp sets are disjoint. Scalers and seasonal profiles use train only. "
             "Optuna tunes on validation only. LSTM uses validation for checkpoint selection. "
             "Per-target champion is chosen by validation MAE before final test reporting. "
             "All families use identical evaluation hours with real historical context across split boundaries. "
             "This is rolling one-step-ahead evaluation, not an open-loop multihour forecast.", "",
             "| Split | Rows | First target | Last target |", "|---|---:|---|---|"]
    for split, values in meta["splits"].items():
        lines.append(f"| {split} | {values['rows']} | {values['first']} | {values['last']} |")
    lines += ["", "## Model comparison", "", "Metrics: MAE / RMSE / sMAPE (%). Both-zero sMAPE terms count as zero.", "",
              "| Family | Validation energy | Test energy | Validation sessions | Test sessions |",
              "|---|---|---|---|---|"]
    for family, values in meta["results"].items():
        cells = [" / ".join(f"{values[split][c][metric]:.4f}" for metric in ["MAE", "RMSE", "sMAPE"])
                 for c in TARGETS for split in ["validation", "test"]]
        lines.append("| " + " | ".join([family, *cells]) + " |")
    lines += ["", f"Champions selected on validation: {meta['champions']}.", "",
              "Seasonal baseline retains the original 50% lag24 / 50% training calendar median blend. "
              "XGBoost uses bounded, seeded Optuna trials (see `optuna.json`). Compact LSTM retains "
              "the original 32-unit architecture, fixes sequence alignment, and scales both inputs and targets. "
              "Every sequence includes the latest feature row; no padded predictions are evaluated.", "",
              "## EDA", "", "`eda.png` shows demand distributions, hourly and monthly seasonality, and "
              "arrival/energy correlation. Zero hours and source missingness are audited; large observations "
              "are retained rather than silently clipped as outliers. `holdout.png` compares all families "
              "on the first test week. Numerical distributions/correlations are recorded in `eda_summary.json`.", "",
              "## Limitations", "", "- Source timezone provenance remains unresolved.",
              "- Completed-session energy is not instantaneous station load or energy consumed within that hour.",
              "- End-of-hour transaction availability and complete source capture are assumed; reporting delays are not modeled.",
              "- Weather and pricing are excluded; the model has not been validated on Egyptian stations.",
              "- Requires at least 192 real continuous hourly records ending immediately before origin.",
              "- Expected sessions remain a nonnegative float statistical estimate, not an exact future arrival count.",
              "- Small CPU model/tuning budgets and one chronological holdout limit generalization evidence."]
    (reports / "evaluation.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
