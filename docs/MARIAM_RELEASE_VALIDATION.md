# Availability integration release gate

Validated 2026-10-28 on `feature/faris-integration` before commit/push, using a no-cache Docker rebuild. This supersedes the availability/recommendation 503 status in the earlier `FINAL_DEMO_VALIDATION.md` snapshot; that file remains the record for the prior candidate.

| Gate | Result |
|---|---|
| Clean Docker image rebuild | API, UI, MLflow built with `docker compose build --no-cache --quiet`. |
| Health | API, UI and MLflow containers healthy; health endpoints HTTP 200. |
| Root suite in Python 3.12 | 157 passed, zero failed, one warning with `RUN_DOCKER_TESTS=1`. This includes API, Docker, UI, frozen models, Digital Twin and simulator tests. |
| Mariam owner suite | 47 passed, six skipped in portable checkout. One skip requires the generated 86 MB cleaned CSV; in the separate review worktree with that CSV and artifacts present, 53 passed. |
| Waseem owner suite | 34 passed, zero failed in a separate training environment with Torch and scikit-learn. |
| Stack smoke | `deployment/smoke_test.py` passed all health, frozen-model, grid-load and simulator checks. |
| Historical availability | Docker API HTTP 200 on `real_jiaxing_availability_request.json`, charger 2, rebuilt duration bundle. |
| Historical recommendation | Docker API HTTP 200 on `real_jiaxing_recommendation_request.json`, charger 8; `mock_signals_used=true`. Fault/offline exclusion and no-eligible 422 checked by integration tests. |
| Streamlit | Availability and recommendation cards rendered without exceptions; recommendation displayed charger 8 and the neutral-signal warning. |
| MLflow | Five portable receipts registered to the demo server as **evidence-only** runs with `original_training_run=false`; no original training run was recreated. |

The frozen demand, solar 24/48, session anomaly and grid-load endpoint tests remained green. No frozen artifact was retrained or changed. Historical demo readiness is separate from production validation: the platform has no live station telemetry, verified equipment fault labels, feeder load, or Egyptian deployment evidence. Mariam's retrospective test metric was exposed before tuning. Recommendation replay has no demonstrated wait improvement because a free charger existed at each evaluated arrival; its neutral candidate-level demand/solar terms are explicitly identified.
