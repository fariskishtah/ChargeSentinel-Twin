# Demonstration fallback plan

The primary demo is the exact historical sequence in `docs/DEMO_RUNBOOK.md`. Keep claims tied to recorded evidence; never present a static report as live inference. No fallback video or screenshot has been created.

| Failure | Recovery | Honest fallback if recovery fails |
|---|---|---|
| Docker build/start is slow | Start the stack before the presentation; wait for `docker compose ps` to show healthy API/UI/MLflow. Retry `DOCKER_MLFLOW_URL=http://localhost:5050 python deployment/smoke_test.py` once healthy. | Show committed bundle versions, `docs/VALIDATION.md`, and the real model/evaluation reports. Say the live stack is unavailable. |
| Port occupied | Check the owning process for 8000/8501/5050. The documented stack already maps MLflow to host 5050. Free a conflicting port only if it is safe to stop that process, then rerun the runbook. | Use committed reports and exact request examples; do not claim a live service response. |
| Network unavailable | Already built local Docker images can start without downloading packages. The API does not call remote services for historical inference. | If images are absent, show committed portable artifacts/reports and explain that live startup needs local images. |
| MLflow UI unavailable | Check `http://localhost:5050/health`; use the portable receipts in `docs/experiment_evidence/`. | Show receipts and original report paths. Solar/anomaly historical run IDs require nonportable local stores; demand's original SQLite store is absent; grid-load had no MLflow run. Do not imply the current UI contains original training runs. |

Existing real fallback evidence includes demand `team/waseem_demand/reports/metrics.json`, solar `team/mazen_solar/reports/integration_hardening/selection.json`, session-risk `team/fatima_anomaly/reports/integration_hardening/metrics.json` and `real_xai_example.json`, grid-load `team/fatima_anomaly/reports/grid_load/comparison.json` and injected-case CSVs, and `docs/VALIDATION.md`. The injected cases are sensitivity tests, not observed faults.
