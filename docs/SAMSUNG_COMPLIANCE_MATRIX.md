# Samsung AI Graduation Project compliance matrix

Acceptance basis: local `SIC8 - AI Project Final .pdf`, especially Smart Energy requirements on pages 5, 13–14 and final evidence on page 17. `DONE` means repository evidence exists for the historical demonstration; it does not mean production validation.

| Requirement | Status | Evidence / repository path | Remaining limitation |
|---|---|---|---|
| Renewable 24–48h generation forecast | DONE | `team/mazen_solar/bundle/solar-v1/`; `docs/SOLAR_ANOMALY_INTEGRATION.md` | PVGIS modelled reference, not measured station PV. |
| Generation or grid-load interval anomaly | PARTIAL | `src/grid_load/`; `team/fatima_anomaly/reports/grid_load/`; `docs/GRID_LOAD_ANOMALY.md` | Jiaxing demand proxy; no real grid-fault labels or Egypt validation. Injected sensitivity is explicit. |
| EV session anomaly as separate function | DONE | `team/fatima_anomaly/bundle/anomaly-v1/`; `POST /api/v1/anomaly/predict` | Abnormal-ending risk, not confirmed fault; high false-positive burden. |
| Time-aware validation and future-information control | DONE | `docs/SPLITS.md`; `docs/PREPROCESSING.md`; `tests/integration/test_solar_anomaly.py`; `tests/integration/test_grid_load.py` | Source timezone unverified. |
| Data audit / EDA | DONE | `team/waseem_demand/reports/data_audit.json`, `eda_summary.json`, `eda.png`; solar and anomaly reports | Raw sources excluded from Git. |
| Reproducible preprocessing | DONE | `team/waseem_demand/demand/build_hourly.py`, `features.py`; `team/mazen_solar/integration/`; `team/fatima_anomaly/integration/`; `src/grid_load/` | Requires original raw files to retrain full models. |
| Model-family comparisons | DONE | Demand `reports/metrics.json`; solar `reports/integration_hardening/selection.json`; grid-load `reports/grid_load/comparison.json` | Grid-load has unlabeled natural hours. |
| Tracked experiments | PARTIAL | `docs/MLFLOW_GUIDE.md`; solar/anomaly local run receipts; Waseem `reports/mlflow_receipt.json` | Local MLflow stores are not committed; receipt reproducibility should be checked in demo environment. |
| Systematic tuning | PARTIAL | Waseem `reports/optuna.json`; solar selection and anomaly threshold curve | No bounded Optuna study for every final model family. |
| Regression/classification metrics | DONE | Demand and solar evaluation reports; session-risk `metrics.json`; grid-load `comparison.json` | Injected rates are not real fault metrics. |
| Global/local explanations and threshold | DONE | Anomaly SHAP CSV and real XAI fixture; grid-load reasons and JSON threshold | Session-risk explanations concern model risk, not fault causes. |
| FastAPI + Pydantic | DONE | `app/api/main.py`, `schemas/models.py`, routers | Historical serving contracts. |
| Streamlit API/UI | DONE | `app/ui/`; separate session-risk and grid-load cards | Historical inputs must be selected. |
| Docker / containerized REST | DONE | `docker-compose.yml`; `deployment/Dockerfile.*`; smoke test | Verify current build in final gate. |
| Versioned portable artifacts | DONE | Demand/solar/session-risk bundles; grid-load JSON plus source checksum | Raw data excluded. |
| Architecture diagram | DONE | `docs/ARCHITECTURE.md`, `docs/architecture.mmd` | SVG optional if renderer unavailable. |
| Data dictionary, provenance, splits | DONE | `docs/DATA_DICTIONARY.md`, `DATA_PROVENANCE.md`, `SPLITS.md` | Provider units/licensing should be checked against raw acquisition records. |
| Functional tests and demo runbook | PARTIAL | `tests/`; `docs/DEMO_RUNBOOK.md` | Final clean-install/Docker gate must pass. |
| Team contribution summary | DONE | `docs/TEAM_CONTRIBUTIONS.md`; `docs/TEAM_STATUS.md` | Mariam not integrated. |
