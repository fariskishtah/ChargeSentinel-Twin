# ChargeSentinel Twin — presentation source

This 15-slide source is ready for PowerPoint production. Use the blue identity of the proposal, large readable figures, and short on-slide text. All metrics are from committed reports. The project is a **historical research prototype**, not a deployed Egyptian grid system.

## Slide 1 — Title

- **ChargeSentinel Twin**
- Renewable-Aware EV Charging & Grid-Load Anomaly Intelligence Platform
- Samsung Innovation Campus AI Track · Faris Mohamed Kishtah · 2026
- **Visual:** Clean title treatment and official logo from the supplied project proposal asset; preserve its proportions and colors.
- **Source:** `README.md`; `docs/ARCHITECTURE.md`.
- **Presenter note:** Introduce the platform as an integrated historical demonstration.

## Slide 2 — Business problem

- EV charging demand and renewable output vary by hour.
- Operators need early risk signals and understandable scenarios; drivers need estimated charger release times.
- Decisions need explicit uncertainty and human review.
- **Visual:** Four simple icons: PV, EV, interval alert, operator.
- **Source:** `docs/DIGITAL_TWIN.md`; `docs/GRID_LOAD_ANOMALY.md`.
- **Presenter note:** State the practical workflow without claiming measured savings or production deployment.

## Slide 3 — Samsung Smart Energy scope

- 24–48 hour renewable generation forecasting.
- Charging demand forecasting and anomaly screening.
- EDA, time-aware evaluation, comparisons, tuning evidence, MLflow, XAI, API/UI and Docker.
- **Visual:** Compact requirement-to-component matrix.
- **Source:** `docs/SAMSUNG_COMPLIANCE_MATRIX.md`.
- **Presenter note:** Differentiate implemented functions from external field validation.

## Slide 4 — System architecture

- Data and documented derived history feed owner pipelines.
- Versioned portable artifacts serve through integration adapters, FastAPI and Streamlit.
- Digital Twin and charging simulator remain deterministic, human-reviewed tools.
- **Visual:** Render `docs/architecture.mmd` / `docs/ARCHITECTURE.md` as the architecture diagram.
- **Source:** `docs/ARCHITECTURE.md`.
- **Presenter note:** MLflow records experiments; live serving does not depend on its availability.

## Slide 5 — Data and provenance

- Jiaxing: 441,077 source sessions, 92 chargers, 2019–2021; 17,547 derived hourly demand rows.
- PVGIS: 4,368 modelled PV reference hours in 2024 H1; Open-Meteo previous-run weather aligned by origin.
- No Egyptian production station or feeder telemetry is claimed.
- **Visual:** Two-row source/timeline graphic plus a small EDA callout.
- **Source:** `docs/DATA_PROVENANCE.md`; `docs/FINAL_EDA_EVIDENCE.md`.
- **Presenter note:** Jiaxing source timezone is unverified; PVGIS is a modelled reference, not measured station PV.

## Slide 6 — Solar forecasting

- Three families compared: Seasonal Persistence, Ridge Regression, XGBoost.
- 24 h champion: Seasonal Persistence; test MAE **0.361 kW**.
- 48 h champion: Ridge Regression; test MAE **1.257 kW**.
- **Visual:** `docs/assets/solar_reference_week.png` plus a compact 24/48 champion callout. Label the plotted line as reference PV, not a forecast trace.
- **Source:** `team/mazen_solar/reports/integration_hardening/selection.json`; `docs/FINAL_MODEL_RESULTS.md`.
- **Presenter note:** May validation selected the families; June test assessed them.

## Slide 7 — EV demand forecasting

- Baseline, XGBoost and LSTM compared on chronological splits.
- XGBoost test MAE: **103.67 kWh** completion-hour energy; **6.34** arrivals/hour.
- Twelve Optuna trials: six energy, six arrivals.
- **Visual:** `team/waseem_demand/reports/holdout.png` or `eda.png`.
- **Source:** `team/waseem_demand/reports/metrics.json`; `docs/TUNING_COVERAGE.md`.
- **Presenter note:** Completed-session energy is a charging-demand proxy, not instantaneous physical grid load.

## Slide 8 — Session anomaly intelligence

- Session-start CatBoost predicts risk of a later recorded abnormal ending.
- Test precision **0.215**, recall **0.888**, F1 **0.346**, PR-AUC **0.238** at threshold **0.40**.
- SHAP provides local reasons; high false-positive burden remains.
- **Visual:** A ranked feature chart from `shap_global_importance.csv` or the real explanation JSON.
- **Source:** `team/fatima_anomaly/reports/integration_hardening/metrics.json`; `real_xai_example.json`.
- **Presenter note:** This is not a confirmed equipment-fault diagnosis.

## Slide 9 — Grid-load interval anomaly

- Causal robust seasonal rule compares each recorded hour with eight earlier same-week hours.
- Natural held-out alert rate **0.70%**; injected sensitivity **24/39** hours.
- Separate from session-start risk; no maintenance-confirmed fault labels.
- **Visual:** `docs/assets/grid_load_anomaly_example.png` (recorded history only).
- **Source:** `docs/GRID_LOAD_ANOMALY.md`; `team/fatima_anomaly/reports/grid_load/comparison.json`.
- **Presenter note:** Say “historical charging-load proxy,” never “measured grid load.”

## Slide 10 — Availability and recommendation

- Default XGBoost duration model: owner test MAE **34.948 min**, RMSE **83.049 min** on normal sessions.
- Real held-out charger 2 fixture estimates free time from start-known context; actual End Time is excluded.
- Deterministic ranker excludes fault/offline candidates and explains weights and alternatives.
- **Visual:** `team/mariam_availability/reports/availability_validation_comparison.png` plus a simple API response callout.
- **Source:** `docs/MARIAM_AVAILABILITY_REVIEW.md`; `team/mariam_availability/tests/fixtures/real_jiaxing_*`.
- **Presenter note:** Recommendation is historical replay; neutral demand/solar score terms are flagged and are not forecasts. Original test was exposed before tuning.

## Slide 11 — Digital Twin and charging simulator

- Scenario modifiers vary solar, demand and charger capacity without controlling real assets.
- FIFO simulator shows charger states, charging energy and queue transitions.
- Operator approval remains required for any real-world action.
- **Visual:** Scenario input/output flow or live Streamlit demonstration; do not fabricate a screenshot.
- **Source:** `docs/DIGITAL_TWIN.md`; `docs/CHARGING_STATION_SIMULATOR.md`.
- **Presenter note:** Explicitly label all outputs “simulation.”

## Slide 12 — API, UI, Docker and MLflow

- FastAPI/Pydantic validates historical requests; Streamlit calls API endpoints.
- Docker Compose packages API, UI and MLflow; portable bundles remove notebook and host-path dependencies.
- Five portable receipts distinguish original runs, a reproduction run and evidence-only registrations.
- **Visual:** Architecture service lane from slide 4, or live `/docs` and MLflow views. No screenshot is committed.
- **Source:** `deployment/README.md`; `docs/experiment_evidence/README.md`; `docs/MARIAM_RELEASE_VALIDATION.md`.
- **Presenter note:** A fresh MLflow demo store is not the original training store.

## Slide 13 — Results at a glance

- Solar: 24 h **0.361 kW** and 48 h **1.257 kW** test MAE on modelled PV reference.
- Demand: XGBoost **103.67 kWh** and **6.34 arrivals** test MAE.
- Availability: default XGBoost **34.948 min** owner retrospective normal-session test MAE.
- **Visual:** Three separate metric cards with units; do not rank unlike targets against one another.
- **Source:** `docs/FINAL_MODEL_RESULTS.md`.
- **Presenter note:** Explain that each task uses a different target and data source.

## Slide 14 — Demonstration and limitations

- Demo sequence: health → solar 24/48 → demand → session risk → grid-load interval → availability → recommendation → Digital Twin → simulator → MLflow evidence.
- Historical dates only; no live Egyptian telemetry or verified feeder-fault labels.
- Candidate-level demand/solar terms in recommendation are neutral placeholders when absent.
- **Visual:** Numbered workflow strip and a compact “research prototype” banner.
- **Source:** `docs/DEMO_RUNBOOK.md`; `docs/FINAL_DEMO_VALIDATION.md`.
- **Presenter note:** Use the documented fixture requests; reset simulator after the demonstration.

## Slide 15 — Team and conclusion

- Waseem: demand; Mazen: solar; Fatima: anomaly/XAI; Mariam: availability/recommendation; Faris: integration, hardening, API/UI, simulation and release.
- Frozen components pass the integrated test, Docker and CI gates.
- Next scientific step: field telemetry, confirmed labels, timezone verification and operational evaluation.
- **Visual:** Team responsibility grid; no fabricated ROI chart.
- **Source:** `docs/TEAM_CONTRIBUTIONS.md`; `docs/FINAL_DEMO_VALIDATION.md`.
- **Presenter note:** Close with the verified historical capability and the external validation needed before deployment.
