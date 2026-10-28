"""Peak-hour counterfactual replay for recommendation policies.

Run from team/mariam_availability:
    python -m src.recommendation.replay

Uses held-out real session starts as arrivals. Occupancy at each arrival is
observable state; actual session ends/outcomes are used only to score realized
wait and the later-fault proxy. Demand is a train-period historical charger by
hour profile, solar is the configured mock fixture, and no queue behavior is
simulated.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd

from ..availability.inference import AvailabilityPredictor
from ..occupancy import load_sessions
from .ranker import load_config, recommend as rank_candidates


TEAM_DIR = Path(__file__).resolve().parents[2]
ROOT = TEAM_DIR.parents[1]
REPORT_DIR = TEAM_DIR / "reports"
FEATURES_PATH = TEAM_DIR / "artifacts" / "features_all.pkl"
TRAIN_END = pd.Timestamp("2021-07-01")
TEST_START = pd.Timestamp("2021-10-01")
PEAK_WINDOWS = ((11, 15), (19, 23))
SEED = 2026


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _demand_profile(sessions: pd.DataFrame, train_mask: np.ndarray) -> dict:
    train = sessions.loc[train_mask].copy()
    train["hour"] = train["start"].dt.hour
    counts = train.groupby(["location", "hour", "charger_id"]).size().rename("sessions").reset_index()
    exposure_days = max((train.start.max().normalize() - train.start.min().normalize()).days + 1, 1)
    counts["per_day"] = counts.sessions / exposure_days
    maxima = counts.groupby(["location", "hour"]).per_day.transform("max").replace(0, np.nan)
    counts["pressure"] = (counts.per_day / maxima).fillna(0.5).clip(0, 1)
    return {
        (str(row.location), int(row.hour), str(row.charger_id)): float(row.pressure)
        for row in counts.itertuples(index=False)
    }


def _cv(counter: Counter, charger_ids: list[str]) -> float:
    values = np.asarray([counter.get(charger, 0) for charger in charger_ids], dtype=float)
    mean = float(values.mean()) if len(values) else 0.0
    return float(values.std(ddof=0) / mean) if mean > 0 else 0.0


def _gini(counter: Counter, charger_ids: list[str]) -> float:
    values = np.sort(np.asarray([counter.get(charger, 0) for charger in charger_ids], dtype=float))
    total = float(values.sum())
    if total == 0 or len(values) == 0:
        return 0.0
    n = len(values)
    return float((2 * np.dot(np.arange(1, n + 1), values) / (n * total)) - (n + 1) / n)


def run_replay() -> dict:
    sessions = load_sessions().copy()
    sessions["start"] = pd.to_datetime(sessions.start)
    sessions["end"] = pd.to_datetime(sessions.end)
    sessions = sessions.sort_values(["charger_id", "start"], kind="mergesort").reset_index(drop=True)
    if not FEATURES_PATH.is_file():
        from ..availability.features import build_features
        build_features(sessions).to_pickle(FEATURES_PATH)
    features = pd.read_pickle(FEATURES_PATH).sort_values(["charger_id", "start"], kind="mergesort").reset_index(drop=True)
    if len(features) != len(sessions) or not features[["charger_id", "start"]].equals(sessions[["charger_id", "start"]]):
        raise ValueError("features_all.pkl does not match the cleaned session data; regenerate features before replay")

    predictor = AvailabilityPredictor()
    train_mask = sessions.start.to_numpy() < TRAIN_END.to_datetime64()
    test_mask = sessions.start.to_numpy() >= TEST_START.to_datetime64()
    hour = sessions.start.dt.hour.to_numpy()
    peak_mask = ((hour >= 11) & (hour < 15)) | ((hour >= 19) & (hour < 23))
    arrival_mask = test_mask & peak_mask
    arrival_indices = np.flatnonzero(arrival_mask)
    if len(arrival_indices) == 0:
        raise ValueError("No held-out peak-hour arrivals were found")

    # Predict session totals in batches once; replay decisions never inspect End Time.
    prediction = np.empty(len(features), dtype=float)
    batch_size = 25_000
    for start_idx in range(0, len(features), batch_size):
        stop_idx = min(start_idx + batch_size, len(features))
        x = predictor.encoder.transform(features.iloc[start_idx:stop_idx])
        raw = np.asarray(predictor.model.predict(x), dtype=float)
        if predictor.target_mode == "log1p":
            raw = np.expm1(raw)
        prediction[start_idx:stop_idx] = np.clip(raw, 0, predictor.cap)

    anomaly_rate = sessions.loc[train_mask].groupby("charger_id").is_abnormal.mean().to_dict()
    global_risk = float(sessions.loc[train_mask].is_abnormal.mean())
    demand = _demand_profile(sessions, train_mask)
    solar_mock = load_config()["mock_signals"]["solar_benefit"]

    timelines = {}
    charger_to_location = {}
    all_chargers = []
    for charger_id, group in sessions.groupby("charger_id", sort=False):
        idx = group.index.to_numpy()
        cid = str(charger_id)
        location = str(group.location.iloc[0])
        if group.location.nunique() != 1:
            raise ValueError(f"Charger {cid} maps to multiple locations")
        charger_to_location[cid] = location
        all_chargers.append(cid)
        timelines[cid] = {
            "indices": idx,
            "starts": group.start.to_numpy(dtype="datetime64[ns]"),
            "ends": group.end.to_numpy(dtype="datetime64[ns]"),
            "abnormal": group.is_abnormal.to_numpy(dtype=int),
        }
    location_chargers = {}
    for charger_id, location in charger_to_location.items():
        location_chargers.setdefault(location, []).append(charger_id)
    for location in location_chargers:
        location_chargers[location].sort()

    rng = random.Random(SEED)
    policy_names = ("ranker", "first_free", "random_free")
    wait_samples = {name: [] for name in policy_names}
    free_choice = Counter()
    assignments = {name: Counter() for name in policy_names}
    later_fault = {name: [] for name in policy_names}
    free_candidate_arrivals = 0
    location_arrival_counts = Counter()
    solar_benefit = {}
    for location, ids in location_chargers.items():
        solar_benefit[location] = solar_mock

    rank_config = load_config()
    for arrival_idx in arrival_indices:
        arrival = sessions.at[arrival_idx, "start"]
        location = str(sessions.at[arrival_idx, "location"])
        ids = location_chargers[location]
        location_arrival_counts[location] += 1
        states = []
        for charger_id in ids:
            timeline = timelines[charger_id]
            # Query immediately before arrivals at this timestamp enter service.
            pos = int(np.searchsorted(timeline["starts"], arrival.to_datetime64(), side="left") - 1)
            active = pos >= 0 and timeline["ends"][pos] > arrival.to_datetime64()
            session_index = int(timeline["indices"][pos]) if active else None
            if active:
                predicted_free = sessions.at[session_index, "start"] + pd.to_timedelta(prediction[session_index], unit="m")
                predicted_free = max(arrival, predicted_free)
                realised_end = sessions.at[session_index, "end"]
                outcome_index = session_index
            else:
                predicted_free = arrival
                realised_end = arrival
                next_pos = pos + 1
                outcome_index = int(timeline["indices"][next_pos]) if next_pos < len(timeline["indices"]) else None
            states.append({
                "charger_id": charger_id,
                "busy": bool(active),
                "pos": pos,
                "session_index": session_index,
                "predicted_free": predicted_free,
                "realised_end": realised_end,
                "outcome_index": outcome_index,
                "risk": float(anomaly_rate.get(charger_id, global_risk)),
                "demand": float(demand.get((location, int(arrival.hour), charger_id), 0.5)),
            })
        free_now = [state for state in states if not state["busy"]]
        if free_now:
            free_candidate_arrivals += 1

        score_candidates = [{
            "charger_id": state["charger_id"],
            "location": location,
            "busy": state["busy"],
            "expected_free_time": state["predicted_free"].isoformat(),
            "anomaly_risk": state["risk"],
            "demand_pressure": state["demand"],
            "solar_benefit": solar_benefit[location],
        } for state in states]
        ranked = rank_candidates({
            "location": location,
            "user_eta": arrival.isoformat(),
            "candidates": score_candidates,
        }, config=rank_config)
        choices = {state["charger_id"]: state for state in states}
        selected = {"ranker": choices[ranked["recommended_charger"]]}
        if free_now:
            selected["first_free"] = min(free_now, key=lambda item: item["charger_id"])
            selected["random_free"] = rng.choice(free_now)
        else:
            soonest = min(states, key=lambda item: (item["predicted_free"], item["charger_id"]))
            selected["first_free"] = soonest
            selected["random_free"] = soonest

        for policy in policy_names:
            state = selected[policy]
            charger_id = state["charger_id"]
            assignments[policy][charger_id] += 1
            free_choice[policy] += int(not state["busy"])
            wait = max(0.0, (state["realised_end"] - arrival).total_seconds() / 60.0)
            wait_samples[policy].append(wait)
            outcome_idx = state["outcome_index"]
            if outcome_idx is not None:
                later_fault[policy].append(int(sessions.at[outcome_idx, "is_abnormal"]))

    candidate_ids = sorted(all_chargers)
    n_arrivals = len(arrival_indices)
    metrics = {}
    for policy in policy_names:
        waits = np.asarray(wait_samples[policy], dtype=float)
        fault_values = np.asarray(later_fault[policy], dtype=float)
        metrics[policy] = {
            "arrivals": n_arrivals,
            "mean_realised_wait_minutes": float(waits.mean()),
            "median_realised_wait_minutes": float(np.median(waits)),
            "p95_realised_wait_minutes": float(np.quantile(waits, 0.95)),
            "selected_free_share": float(free_choice[policy] / n_arrivals),
            "assignments": int(sum(assignments[policy].values())),
            "assignment_cv": _cv(assignments[policy], candidate_ids),
            "assignment_gini": _gini(assignments[policy], candidate_ids),
            "observed_followup_outcomes": int(len(fault_values)),
            "selected_charger_later_fault_share": float(fault_values.mean()) if len(fault_values) else None,
        }

    data_path = ROOT / "data" / "processed" / "Charging_Data_clean.csv"
    result = {
        "data_sha256": _sha256(data_path),
        "sessions": int(len(sessions)),
        "test_period_start": str(sessions.loc[test_mask, "start"].min()),
        "test_period_end": str(sessions.loc[test_mask, "start"].max()),
        "arrival_window": "held-out real session starts during 11:00-14:59 and 19:00-22:59 local clock time",
        "arrival_count": n_arrivals,
        "share_arrivals_with_any_free_charger": free_candidate_arrivals / n_arrivals,
        "model_version": predictor.model_version,
        "ranker_version": rank_config["version"],
        "ranker_weights": rank_config["weights"],
        "prefer_free_if_available": rank_config["prefer_free_if_available"],
        "random_seed": SEED,
        "policies": metrics,
        "limitations": [
            "Counterfactual replay: observed real starts are used as arrivals, but choices can differ from actual drivers.",
            "There is no queue data; drivers who found every charger busy and left are unobserved.",
            "Busy state at arrival is treated as observable. Predicted free time uses duration inference; actual end time is scoring-only.",
            "Historical demand-by-hour is a proxy, solar benefit is a neutral configured mock, and risk is a training-period charger rate.",
            "For a free charger, later-fault scoring uses its next observed session as a proxy outcome, not the counterfactual session created by this arrival.",
            "Only held-out test-period arrivals in the specified peak windows are included.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "availability_recommendation_replay.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    _write_report(result)
    return result


def _write_report(result: dict) -> None:
    lines = [
        "# Availability recommendation offline replay",
        "",
        "## Method",
        "",
        f"The replay evaluated {result['arrival_count']:,} real session starts from the held-out period "
        f"{result['test_period_start']} to {result['test_period_end']}, restricted to 11:00–14:59 and 19:00–22:59 local time. "
        f"The clean dataset SHA-256 is `{result['data_sha256']}`. At least one charger was free at "
        f"{result['share_arrivals_with_any_free_charger']:.1%} of those arrivals.",
        "",
        f"Ranker version `{result['ranker_version']}` uses weights {result['ranker_weights']} for normalized wait, train-period charger anomaly rate, train-period charger/hour demand pressure, and neutral solar placeholder. If any candidate is actually free, it scores only free candidates. First-free chooses the lexicographically first currently free charger. Random-free chooses a currently free charger using seed 2026. When every charger is busy, the two baselines select the soonest model-predicted release, with charger ID as tie-break. Realized wait and subsequent fault labels are scoring-only.",
        "",
        "## Results",
        "",
        "| Policy | Mean wait (min) | Median wait (min) | P95 wait (min) | Selected free | Assignment CV | Assignment Gini | Later-fault share* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, values in result["policies"].items():
        fault = values["selected_charger_later_fault_share"]
        fault_display = "n/a" if fault is None else f"{fault:.1%}"
        lines.append(
            f"| {policy} | {values['mean_realised_wait_minutes']:.2f} | "
            f"{values['median_realised_wait_minutes']:.2f} | {values['p95_realised_wait_minutes']:.2f} | "
            f"{values['selected_free_share']:.1%} | {values['assignment_cv']:.3f} | "
            f"{values['assignment_gini']:.3f} | {fault_display} |"
        )
    lines.extend([
        "",
        "\\*For a charger busy at arrival, this is the active session's later abnormal label. For a free charger, it is the next observed session's label. The latter is a proxy outcome, not a real counterfactual session.",
        "",
        "Utilization balance is reported as assignment concentration (CV and Gini) across chargers during this arrival replay; it is not a measured change in physical utilization.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in result["limitations"]],
        "",
        "This is a policy comparison under observed occupancy and a duration model. It does not estimate demand from drivers who abandoned a busy site and does not establish causal effects of recommendations.",
    ])
    (REPORT_DIR / "availability_recommendation_evaluation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    report = run_replay()
    print(json.dumps(report["policies"], indent=2))
