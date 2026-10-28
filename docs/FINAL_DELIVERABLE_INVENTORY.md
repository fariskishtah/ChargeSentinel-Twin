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
| Streamlit/UI source | READY | `app/ui/` | Grid-load time series is supplied as a linked reviewer/demo figure, while its UI card shows interval details. |
| Root README, dependencies, run instructions | READY | `README.md`, `requirements/`, `team/waseem_demand/requirements-training.txt`, `deployment/README.md` | Training needs source data. |
| Architecture diagram | READY | `docs/ARCHITECTURE.md`, `docs/architecture.mmd` | Mermaid source; no separate SVG. |
| Data provenance and dictionary | READY | `docs/DATA_PROVENANCE.md`, `docs/DATA_DICTIONARY.md` | Raw provider licenses require source-side verification. |
| Split and preprocessing explanations | READY | `docs/SPLITS.md`, `docs/PREPROCESSING.md` | Source timezone unverified. |
| Energy data audit/EDA | READY | `docs/FINAL_EDA_EVIDENCE.md`; Waseem audit/EDA; Mariam EDA tables; committed solar derived-data audit | Raw sources excluded; source receipts and derived files support the documented analysis. |
| Tracked experiment receipts | READY | `docs/experiment_evidence/*_runs.json` and `README.md` classify original, reproduction and evidence-only records | Original local stores are not portable; grid-load has a comparison receipt rather than an MLflow training run. |
| Tuning record | READY | Waseem Optuna report, Mariam 25-trial report, anomaly threshold curve, `docs/TUNING_COVERAGE.md` | Solution-level evidence; frozen solar/session-risk hyperparameters lacked independent HPO. |
| Model comparisons and held-out metrics | READY | `docs/FINAL_MODEL_RESULTS.md` and cited owner reports | Grid-load injections are not real fault labels; Mariam test was exposed before tuning. |
| SHAP/XAI and alert reasons | READY | `team/fatima_anomaly/reports/integration_hardening/shap_global_importance.csv`, `real_xai_example.json`; `src/grid_load/detector.py` | Explanations are model contributions/deviations, not failure diagnosis. |
| Functional tests | READY | `tests/`, `team/waseem_demand/tests/`, `.github/workflows/ci.yml` | Final fresh-clone gate recorded separately. |
| Versioned model/runtime artifacts | READY | `team/waseem_demand/bundle/v1/`, `team/mazen_solar/bundle/solar-v1/`, `team/fatima_anomaly/bundle/anomaly-v1/`, `team/mariam_availability/bundle/availability-v1/`, `src/grid_load/grid-load-v1.json` | Only selected serving artifacts ship; Mariam bundle is a documented rebuild. |
| Containerized API/UI/MLflow | READY | `docker-compose.yml`, `deployment/Dockerfile.*`, `deployment/smoke_test.py`, `docs/MARIAM_RELEASE_VALIDATION.md` | Current host availability must be checked at demo time. |
| Grid-load historical visualization | READY | `docs/assets/grid_load_anomaly_example.png`; `docs/tools/build_grid_load_chart.py`; `docs/GRID_LOAD_ANOMALY.md` | Charging-load proxy, not feeder measurement. |
| Demo runbook and rehearsal | READY | `docs/DEMO_RUNBOOK.md`, `docs/FINAL_DEMO_VALIDATION.md`, `docs/DEMO_FALLBACK_PLAN.md` | Historical inputs only. |
| Team contribution summary | READY | `docs/TEAM_CONTRIBUTIONS.md`, `docs/TEAM_STATUS.md` | Mariam source work is integrated after rebuild. |
| Presentation source and asset inventory | READY | `docs/PRESENTATION_CONTENT.md` (15 slides); `docs/PRESENTATION_ASSETS.md`; `docs/assets/` | PowerPoint assembly remains a formatting step; no invented screenshot or logo is committed. |
| Qualitative operational recommendations | READY | `docs/DIGITAL_TWIN.md`, simulator scenario output, `docs/GRID_LOAD_ANOMALY.md` | No verified economic ROI or field deployment outcome is claimed. |
