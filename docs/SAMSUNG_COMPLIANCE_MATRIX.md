# Samsung AI Graduation Project compliance matrix

Acceptance basis: local `SIC8 - AI Project Final .pdf`, especially Smart Energy requirements on pages 5, 13–14 and final evidence on page 17. `DONE` means repository evidence exists for the historical demonstration; it does not mean production validation.

| Requirement | Status | Evidence / repository path | Remaining limitation |
|---|---|---|---|
| Renewable 24–48h generation forecast | DONE | `team/mazen_solar/bundle/solar-v1/`; `docs/SOLAR_ANOMALY_INTEGRATION.md` | PVGIS modelled reference, not measured station PV. |
| Generation or grid-load interval anomaly | PARTIAL — DEMONSTRATED WITH PROXY | `src/grid_load/`; `team/fatima_anomaly/reports/grid_load/`; `docs/GRID_LOAD_ANOMALY.md` | Historical Jiaxing charging-load proxy, not feeder telemetry; no real grid-fault labels or Egypt validation. Injected sensitivity is explicit. |
| EV session anomaly as separate function | DONE | `team/fatima_anomaly/bundle/anomaly-v1/`; `POST /api/v1/anomaly/predict` | Abnormal-ending risk, not confirmed fault; high false-positive burden. |
| Time-aware validation and future-information control | DONE | `docs/SPLITS.md`; `docs/PREPROCESSING.md`; `tests/integration/test_solar_anomaly.py`; `tests/integration/test_grid_load.py` | Source timezone unverified. |
| Data audit / EDA | PARTIAL | `team/waseem_demand/reports/data_audit.json`, `eda_summary.json`, `eda.png`; solar selection report records timestamp alignment and two missing PV targets | Complete raw-source EDA for every energy source is not committed. |
| Reproducible preprocessing | DONE | `team/waseem_demand/demand/build_hourly.py`, `features.py`; `team/mazen_solar/integration/`; `team/fatima_anomaly/integration/`; `src/grid_load/` | Requires original raw files to retrain full models. |
| Model-family comparisons | DONE | Demand `reports/metrics.json`; solar `reports/integration_hardening/selection.json`; grid-load `reports/grid_load/comparison.json` | Grid-load has unlabeled natural hours. |
| Tracked experiments | PARTIAL | `docs/experiment_evidence/`; Waseem, solar and session-risk original run IDs/receipts | Original stores are local; demand SQLite store is absent. Grid-load comparison has no MLflow run. |
| Systematic tuning | PARTIAL | Waseem `reports/optuna.json`; anomaly threshold curve; `docs/TUNING_COVERAGE.md` | Demand has 12 Optuna trials; solar and session-risk model hyperparameters are fixed, not HPO-tuned. |
| Regression/classification metrics | DONE | Demand and solar evaluation reports; session-risk `metrics.json`; grid-load `comparison.json` | Injected rates are not real fault metrics. |
| Global/local explanations and threshold | DONE | Anomaly SHAP CSV and real XAI fixture; grid-load reasons and JSON threshold | Session-risk explanations concern model risk, not fault causes. |
| FastAPI + Pydantic | DONE | `app/api/main.py`, `schemas/models.py`, routers | Historical serving contracts. |
| Streamlit API/UI | DONE | `app/ui/`; separate session-risk and grid-load cards | Historical inputs must be selected. |
| Time-series view linking interval alerts to observed behavior | PARTIAL | `app/ui/views.py` shows score, threshold, recorded energy and reason; `docs/GRID_LOAD_ANOMALY.md` explains temporal baseline | Grid-load card lacks an in-UI historical time-series chart. |
| Docker / containerized REST | DONE | `docker-compose.yml`; `deployment/Dockerfile.*`; `docs/FINAL_DEMO_VALIDATION.md` | Fresh-clone no-cache build and 155 tests passed on 2026-10-28; host Docker availability remains a demo prerequisite. |
| Versioned portable artifacts | DONE | Demand/solar/session-risk bundles; grid-load JSON plus source checksum | Raw data excluded. |
| Architecture diagram | DONE | `docs/ARCHITECTURE.md`, `docs/architecture.mmd` | SVG optional if renderer unavailable. |
| Data dictionary, provenance, splits | DONE | `docs/DATA_DICTIONARY.md`, `DATA_PROVENANCE.md`, `SPLITS.md` | Provider units/licensing should be checked against raw acquisition records. |
| Functional tests and demo runbook | DONE | `tests/`; `docs/DEMO_RUNBOOK.md`; `docs/VALIDATION.md` | Historical demo only; no field validation. |
| Team contribution summary | DONE | `docs/TEAM_CONTRIBUTIONS.md`; `docs/TEAM_STATUS.md` | Mariam not integrated. |
