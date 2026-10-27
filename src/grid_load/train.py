"""Reproduce a causal rule/Isolation Forest comparison and injected sensitivity audit.

Run with the training environment; only the rule configuration is served.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from src.grid_load.detector import CONFIG, HISTORY, interval_features, score

REPORT = Path(__file__).resolve().parents[2] / "team/fatima_anomaly/reports/grid_load"
TRAIN_END = datetime(2021, 1, 1)
VALID_END = datetime(2021, 7, 1)
TEST_END = datetime(2022, 1, 1)
WINDOW_HOURS = 3


def _vector(item: dict) -> list[float]:
    stamp = datetime.fromisoformat(item["timestamp"])
    return [item["energy_kwh"], item["sessions"],
            item["energy_deviation"], item["sessions_deviation"],
            item["expected_energy_kwh"], item["expected_sessions"],
            float(stamp.hour), float(stamp.weekday())]


def _quantile(values: list[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), fraction))


def main() -> None:
    with HISTORY.open(newline="") as handle:
        rows = [{"timestamp": datetime.fromisoformat(row["timestamp"]),
                 "energy_kwh": float(row["energy_kwh"]),
                 "sessions": float(row["sessions"])} for row in csv.DictReader(handle)]
    indexed = [(i, interval_features(rows, i)) for i in range(8 * 168, len(rows))]
    train = [(i, x) for i, x in indexed if x["timestamp"] < TRAIN_END.isoformat()]
    valid = [(i, x) for i, x in indexed if TRAIN_END.isoformat() <= x["timestamp"] < VALID_END.isoformat()]
    test = [(i, x) for i, x in indexed if VALID_END.isoformat() <= x["timestamp"] < TEST_END.isoformat()]
    if not train or not valid or not test:
        raise ValueError("Incomplete chronological splits")

    forest = IsolationForest(n_estimators=100, max_samples=256,
                             contamination="auto", random_state=42, n_jobs=1)
    forest.fit(np.asarray([_vector(x) for _, x in train]))
    rule_threshold = _quantile([score(x) for _, x in valid], 0.99)
    forest_valid = -forest.score_samples(np.asarray([_vector(x) for _, x in valid]))
    forest_threshold = _quantile(forest_valid.tolist(), 0.99)

    def injections(split: list[tuple[int, dict]], case_type: str) -> list[dict]:
        # Fixed, spaced three-hour windows; each injected hour is evaluated
        # against the unchanged earlier real history, never written into it.
        starts = [split[j][0] for j in range(48, len(split) - WINDOW_HOURS, 360)]
        cases = []
        for start in starts:
            for offset in range(WINDOW_HOURS):
                index = start + offset
                original = rows[index]
                injected_energy = original["energy_kwh"] * 1.8 + 200.0
                injected_sessions = original["sessions"] * 1.5 + 10.0
                item = interval_features(rows, index, energy=injected_energy,
                                         sessions=injected_sessions)
                cases.append({"window_start": rows[start]["timestamp"].isoformat(),
                              "timestamp": item["timestamp"],
                              "original_energy_kwh": original["energy_kwh"],
                              "injected_energy_kwh": injected_energy,
                              "original_sessions": original["sessions"],
                              "injected_sessions": injected_sessions,
                              "rule_score": score(item),
                              "forest_score": float(-forest.score_samples(
                                  np.asarray([_vector(item)]))[0]),
                              "case_type": case_type})
        return cases

    validation_cases = injections(valid, "SIMULATED / INJECTED VALIDATION CASE")
    test_cases = injections(test, "SIMULATED / INJECTED HELD-OUT TEST CASE")
    REPORT.mkdir(parents=True, exist_ok=True)
    for name, cases in [("validation_injections.csv", validation_cases),
                        ("test_injections.csv", test_cases)]:
        with (REPORT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(cases)

    def summary(split: list[tuple[int, dict]], cases: list[dict]) -> dict:
        raw_forest = -forest.score_samples(np.asarray([_vector(x) for _, x in split]))
        return {
            "natural_unlabeled_intervals": len(split),
            "simulated_injected_hours": len(cases),
            "rule_natural_alert_rate": sum(score(x) >= rule_threshold for _, x in split) / len(split),
            "forest_natural_alert_rate": float(np.mean(raw_forest >= forest_threshold)),
            "rule_injected_detection_rate": sum(x["rule_score"] >= rule_threshold for x in cases) / len(cases),
            "forest_injected_detection_rate": sum(x["forest_score"] >= forest_threshold for x in cases) / len(cases),
        }

    config = {
        "detector_version": "grid-load-seasonal-rule-v1",
        "history_sha256": hashlib.sha256(HISTORY.read_bytes()).hexdigest(),
        "source": "Jiaxing completed-session energy proxy, not measured grid power",
        "reference": "previous eight same hour-of-week observations only",
        "threshold": rule_threshold,
        "threshold_selection": "99th percentile of natural, unlabeled January-June 2021 validation scores",
        "validation_end_exclusive": VALID_END.isoformat(),
        "test_start": VALID_END.isoformat(),
    }
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    report = {
        "data": "Jiaxing hourly derived completed-session energy and start-hour arrivals",
        "history_sha256": config["history_sha256"],
        "splits": {
            "reference_train": [train[0][1]["timestamp"], train[-1][1]["timestamp"], len(train)],
            "validation": [valid[0][1]["timestamp"], valid[-1][1]["timestamp"], len(valid)],
            "held_out_test": [test[0][1]["timestamp"], test[-1][1]["timestamp"], len(test)]},
        "rule_threshold": rule_threshold,
        "forest_threshold": forest_threshold,
        "forest_parameters": {"n_estimators": 100, "max_samples": 256,
                              "contamination": "auto", "random_state": 42},
        "injection": {"window_hours": WINDOW_HOURS, "spacing_hours": 360,
                      "first_offset_hours": 48,
                      "energy_rule": "observed * 1.8 + 200 kWh",
                      "sessions_rule": "observed * 1.5 + 10",
                      "history_modified": False},
        "validation": summary(valid, validation_cases),
        "held_out_test": summary(test, test_cases),
        "selected": "robust seasonal rule",
        "selection_reason": "causal, interpretable, portable; fixed 1% validation alert budget",
        "ground_truth_note": "No maintenance-confirmed hourly fault labels. Natural alert rate is not false-positive rate; injected detection rate is sensitivity to synthetic perturbations, not real fault recall.",
    }
    (REPORT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
