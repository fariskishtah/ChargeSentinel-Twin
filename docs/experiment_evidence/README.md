# Portable experiment receipts

These JSON files transcribe committed model reports and, where still readable, the original local MLflow runs. They are **evidence receipts**, not new training runs, reconstructed MLflow databases, or claims that historical `runs:/` URIs work in a clean clone.

| File | Original tracking evidence | Portable limit |
|---|---|---|
| `demand_runs.json` | Three committed run IDs and artifact URIs in Waseem's receipt; metrics and parameters in committed reports. | Original local SQLite store is absent here. Its run tags and live artifact URI could not be independently re-read. |
| `solar_runs.json` | Two selected run IDs, parameters, metrics and tags verified read-only in the original local file store and cross-checked with committed selection report. | Original local file store is not in Git. |
| `anomaly_runs.json` | Selected run ID, parameters, metrics and tags verified read-only in original local file store and cross-checked with committed bundle/report. | Original local file store is not in Git. |
| `grid_load_runs.json` | Deterministic comparison report and versioned rule JSON. | No original MLflow run exists; the receipt says so. |
| `availability_runs.json` | Mariam's original run receipt and a separately identified local portability rebuild. | Original SQLite store and original untracked model bundle are absent; the one-trial rebuild is not the 25-trial owner experiment. Prior test exposure is disclosed. |

Each JSON receipt now states its represented run type: **ORIGINAL TRAINING RUN** for the historical demand, solar and anomaly receipts; **REPRODUCTION RUN** for the separately identified availability rebuild; and **EVIDENCE-ONLY REGISTRATION** for the grid-load comparison and optional live metadata registrations. Availability also retains the owner's original-run receipt. These are classifications of the evidence represented; a JSON receipt is not itself an imported MLflow run. All paths in receipts are repository-relative. `not portable / unavailable` is literal where the source evidence does not establish a value. Original owner attribution is preserved. A fresh MLflow service may have no historical runs; use these receipts and the committed bundles for the demo.

Optional live metadata registration: with a healthy demo MLflow server, run `MLFLOW_TRACKING_URI=http://localhost:5050 python docs/experiment_evidence/register_metadata.py`. It creates or reuses **evidence-only** runs for the five receipts, tags `original_training_run=false`, and logs no model training metrics or binaries. The registration is local to that server; the JSON receipts remain the portable record.
