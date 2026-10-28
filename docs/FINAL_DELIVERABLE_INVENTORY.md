# Final deliverable inventory

Mapped to the 2026 Samsung AI Graduation Project Handbook (Smart Energy brief, pp. 13–14; final package, p. 17). **READY** means present for a historical demonstration, not real-grid validation.

| Deliverable | Status | Repository path / evidence | Limit |
|---|---|---|---|
| Demand preprocessing and hourly aggregation | READY | `team/waseem_demand/demand/build_hourly.py`, `features.py` | Raw Jiaxing CSV excluded. |
| Solar feature preparation | READY | `team/mazen_solar/integration/source_features.py`, `build_handoff.py` | Raw PVGIS/Open-Meteo excluded. |
| Session-risk preprocessing | READY | `team/fatima_anomaly/integration/features.py` | Jiaxing source timezone unverified. |
| Grid-load interval features | READY | `src/grid_load/detector.py` | Historical demand proxy. |
| Training/evaluation source | READY | `team/waseem_demand/demand/train.py`, `team/mazen_solar/integration/build_handoff.py`, `team/fatima_anomaly/integration/train.py`, `team/mariam_availability/src/availability/train.py`, `src/grid_load/train.py` | Availability portable rebuild is separately documented; raw sources required for full reproduction. |
| Inference source | READY | `team/waseem_demand/demand/inference.py`, `team/mazen_solar/integration/inference.py`, `team/fatima_anomaly/integration/inference.py`, `team/mariam_availability/src/`, `src/grid_load/detector.py` | Historical contracts. |
| FastAPI/Pydantic source | READY | `app/api/main.py`, `app/api/routers/`, `app/api/schemas/models.py` | Availability/recommendation require explicit historical context. |
| Streamlit/UI source | READY | `app/ui/` | Grid-load card lacks a historical time-series chart. |
| Root README, dependencies, run instructions | READY | `README.md`, `requirements/`, `team/waseem_demand/requirements-training.txt`, `deployment/README.md` | Training needs source data. |
| Architecture diagram | READY | `docs/ARCHITECTURE.md`, `docs/architecture.mmd` | Mermaid source; no separate SVG. |
| Data provenance and dictionary | READY | `docs/DATA_PROVENANCE.md`, `docs/DATA_DICTIONARY.md` | Raw provider licenses require source-side verification. |
| Split and preprocessing explanations | READY | `docs/SPLITS.md`, `docs/PREPROCESSING.md` | Source timezone unverified. |
| Energy data audit/EDA | PARTIAL | `team/waseem_demand/reports/data_audit.json`, `eda_summary.json`, `eda.png`; Mariam `notebooks/02_eda.ipynb` and `reports/tables/`; solar selection report | Full raw-source EDA for every energy source is not committed. |
| Tracked experiment receipts | PARTIAL | `docs/experiment_evidence/*_runs.json`; original local solar/anomaly MLflow runs audited | Original stores not portable; demand SQLite missing; grid-load had no MLflow run. |
| Tuning record | PARTIAL | `team/waseem_demand/reports/optuna.json`, anomaly `threshold_curve.csv`, `docs/TUNING_COVERAGE.md` | Frozen solar/session-risk model hyperparameters lacked HPO search. |
| Model comparisons and held-out metrics | READY | Demand `reports/metrics.json`, solar `reports/integration_hardening/selection.json`, anomaly `reports/integration_hardening/metrics.json`, grid-load `reports/grid_load/comparison.json` | Grid-load injections are not real fault labels. |
| SHAP/XAI and alert reasons | READY | `team/fatima_anomaly/reports/integration_hardening/shap_global_importance.csv`, `real_xai_example.json`; `src/grid_load/detector.py` | Explanations are model contributions/deviations, not failure diagnosis. |
| Functional tests | READY | `tests/`, `team/waseem_demand/tests/`, `.github/workflows/ci.yml` | Final fresh-clone gate recorded separately. |
| Versioned model/runtime artifacts | READY | `team/waseem_demand/bundle/v1/`, `team/mazen_solar/bundle/solar-v1/`, `team/fatima_anomaly/bundle/anomaly-v1/`, `team/mariam_availability/bundle/availability-v1/`, `src/grid_load/grid-load-v1.json` | Only selected serving artifacts ship; Mariam bundle is a documented rebuild. |
| Containerized API/UI/MLflow | READY | `docker-compose.yml`, `deployment/Dockerfile.*`, `deployment/smoke_test.py`, `docs/MARIAM_RELEASE_VALIDATION.md` | Current host availability must be checked at demo time. |
| Demo runbook and rehearsal | READY | `docs/DEMO_RUNBOOK.md`, `docs/FINAL_DEMO_VALIDATION.md`, `docs/DEMO_FALLBACK_PLAN.md` | Historical inputs only. |
| Team contribution summary | READY | `docs/TEAM_CONTRIBUTIONS.md`, `docs/TEAM_STATUS.md` | Mariam source work is integrated after rebuild. |
| Presentation deck | MISSING | No deck committed in repository | Prepare separately for final submission. |
| Business recommendations | PARTIAL | `docs/DIGITAL_TWIN.md`, simulator scenario output, `docs/GRID_LOAD_ANOMALY.md` | No verified economic ROI or field deployment recommendation. |
