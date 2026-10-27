# Final hardening validation — 2026-10-28

Starting branch/HEAD: `feature/faris-integration` at `673e8663894eda60ab21662d93276ef810a6af55`, clean before work. PR #1 source ZIP was rehashed (`db3b665abfe63901110c4029d54f7ab797981bad8149aa11b5314a8522ac2fc5`) and old config/code reconfirmed random stratification, fixed 0.41, and completed-session inputs. The PR was commented on and closed without merge; remote `feature/fatima-anomaly` remains at `99f157a54c689d385a7aba4f823b612468822eb8`.

The `python` commands below were executed with an isolated Python 3.12 environment; interpreter paths are omitted to keep the instructions portable.

## Commands and results

| Gate | Exact command | Result |
|---|---|---|
| Current model checksum | `shasum -a 256 team/fatima_anomaly/bundle/anomaly-v1/catboost.cbm` | `3492f063fd6b46ba1c511e67b8d3fa094bd40a2c2edeaf64d429358c716e006d`, matching config. |
| Grid-load comparison reproduction | `python -m src.grid_load.train` | Produced deterministic JSON/CSV evidence from committed hourly history; no source history modified. |
| Waseem restored source tests | `python -m pytest team/waseem_demand/tests -q` | 34 passed. |
| Local practical suite | `python -m pytest tests -q` | 133 passed, 22 Docker-gated skipped. |
| Fresh container install | `docker compose build --no-cache` | API, UI, MLflow images built successfully from Python 3.12 slim. `docker compose build api` refreshed source after final detector edits. |
| Container start | `MLFLOW_HOST_PORT=5050 docker compose up -d --force-recreate` | All three services healthy. |
| Full Docker-gated suite | `RUN_DOCKER_TESTS=1 MLFLOW_HOST_PORT=5050 DOCKER_MLFLOW_URL=http://localhost:5050 python -m pytest tests -q` | **155 passed, 0 failed, 0 skipped**. |
| Container smoke | `DOCKER_MLFLOW_URL=http://localhost:5050 python deployment/smoke_test.py` | API/UI/MLflow health, demand, solar 24/48, session anomaly, grid-load, Digital Twin, and UI-to-API passed. |
| API import outside project CWD | `docker compose exec -T --workdir /tmp api python ...` | Python 3.12.16; grid-load actual alert and anomaly-v1 bundle loaded. |
| Merge rehearsal | See `docs/RELEASE_REHEARSAL.md` | No conflicts; 133 practical tests passed; no main merge. |

Runtime serving uses committed portable assets only; no raw Jiaxing file, Google Drive path, notebook execution, or local MLflow tracking store is required. Demand and solar champion binaries/checksums are unchanged from starting HEAD. Session-risk test precision/recall/F1/ROC-AUC/PR-AUC remain 0.2147/0.8879/0.3458/0.6423/0.2377. Availability and recommendation remain safe 503. The grid-load screen has only historical proxy and explicitly injected sensitivity evidence; there is no real grid-fault ground truth or Egypt production validation.
