# Portable experiment receipts

These JSON files transcribe committed model reports and, where still readable, the original local MLflow runs. They are **evidence receipts**, not new training runs, reconstructed MLflow databases, or claims that historical `runs:/` URIs work in a clean clone.

| File | Original tracking evidence | Portable limit |
|---|---|---|
| `demand_runs.json` | Three committed run IDs and artifact URIs in Waseem's receipt; metrics and parameters in committed reports. | Original local SQLite store is absent here. Its run tags and live artifact URI could not be independently re-read. |
| `solar_runs.json` | Two selected run IDs, parameters, metrics and tags verified read-only in the original local file store and cross-checked with committed selection report. | Original local file store is not in Git. |
| `anomaly_runs.json` | Selected run ID, parameters, metrics and tags verified read-only in original local file store and cross-checked with committed bundle/report. | Original local file store is not in Git. |
| `grid_load_runs.json` | Deterministic comparison report and versioned rule JSON. | No original MLflow run exists; the receipt says so. |

All paths in receipts are repository-relative. `not portable / unavailable` is literal where the source evidence does not establish a value. Original owner attribution is preserved. A fresh MLflow service may have no historical runs; use these receipts and the committed bundles for the demo. Any later metadata registration must be labeled **artifact/evidence registration**, never an original training run.

Optional live metadata registration: with a healthy demo MLflow server, run `MLFLOW_TRACKING_URI=http://localhost:5050 python docs/experiment_evidence/register_metadata.py`. It creates or reuses **evidence-only** runs for the four receipts, tags `original_training_run=false`, and logs no model training metrics or binaries. The registration is local to that server; the JSON receipts remain the portable record.
