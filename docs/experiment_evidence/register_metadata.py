"""Optionally register portable receipts in a live MLflow service.

Creates explicitly labeled evidence runs. It never trains, imports historical runs,
or registers a production model. Requires an explicit MLFLOW_TRACKING_URI.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = Path(__file__).resolve().parent
EXPERIMENT = "chargesentinel-release-evidence"


def main() -> None:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri or not (uri.startswith("http://") or uri.startswith("https://")):
        raise ValueError("Set an explicit reachable HTTP(S) MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    existing = client.search_runs([experiment_id], max_results=1000)
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    for name in ("demand_runs.json", "solar_runs.json", "anomaly_runs.json", "grid_load_runs.json", "availability_runs.json"):
        path = RECEIPTS / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        match = next((run for run in existing if run.data.tags.get("receipt_sha256") == digest
                      and run.data.tags.get("run_kind") == "artifact/evidence registration"), None)
        if match:
            print(f"{name}: existing evidence registration {match.info.run_id}")
            continue
        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"{name.removesuffix('.json')}-artifact-evidence-registration") as run:
            mlflow.set_tags({
                "run_kind": "artifact/evidence registration",
                "original_training_run": "false",
                "receipt_sha256": digest,
                "release_source_commit": source_head,
                "historical_run_ids": "see portable receipt; never remapped",
            })
            mlflow.log_params({"receipt_file": name, "evidence_only": True})
            mlflow.log_artifact(str(path), artifact_path="portable_receipt")
            print(f"{name}: new evidence registration {run.info.run_id}")


if __name__ == "__main__":
    main()
