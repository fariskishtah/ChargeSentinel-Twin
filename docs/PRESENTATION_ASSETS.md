# Presentation asset inventory

Use these committed artifacts with `docs/PRESENTATION_CONTENT.md`. Preserve source captions and units. A file listed as evidence is not automatically a slide-ready graphic.

| Slides | Visual or evidence | Repository path | Use and caveat |
|---|---|---|---|
| 4, 12 | Architecture source | `docs/architecture.mmd`; rendered Markdown in `docs/ARCHITECTURE.md` | Export Mermaid as a diagram; maintain the historical/proxy labels. |
| 5, 7 | Jiaxing demand EDA | `team/waseem_demand/reports/eda.png`; `data_audit.json`; `eda_summary.json` | Recorded Jiaxing transaction and derived hourly-history evidence. |
| 7 | Demand holdout plot | `team/waseem_demand/reports/holdout.png` | Forecast comparison; label completion-hour energy and arrivals. |
| 6 | Solar reference week | `docs/assets/solar_reference_week.png` | Generated from committed PVGIS-derived hourly file; **modelled PV reference**, not measured output or model forecast. Rebuild with `python docs/tools/build_solar_reference_chart.py`. |
| 8 | Session-risk XAI values | `team/fatima_anomaly/reports/integration_hardening/shap_global_importance.csv`; `real_xai_example.json` | Build a small chart or response callout from actual values. No PNG is committed. |
| 9 | Grid-load anomaly example | `docs/assets/grid_load_anomaly_example.png` | Recorded history, causal seasonal baseline and natural alert points only. Rebuild with `PYTHONPATH=. python docs/tools/build_grid_load_chart.py`. |
| 10 | Availability validation chart and XAI | `team/mariam_availability/reports/availability_validation_comparison.png`; `reports/figures/xai/shap_mean_abs_top15.png` | Owner-reported experiment evidence; the deployed portable rebuild has separate metrics. |
| 10 | Real availability/recommendation response | `team/mariam_availability/tests/fixtures/real_jiaxing_availability_response.json`; `real_jiaxing_recommendation_response.json` | Format as short callouts. Keep historical and neutral-signal caveats. |
| 12 | MLflow record | `docs/experiment_evidence/*.json`; `team/mariam_availability/reports/availability_mlflow_runs.png` | The PNG is an owner-run table, not a live MLflow UI screenshot. Use a live demo or make a concise table from receipts. |
| 13 | Final comparison tables | `docs/FINAL_MODEL_RESULTS.md` | Copy values with task-specific units and retrospective-test caveat. |
| 14 | Demo steps | `docs/DEMO_RUNBOOK.md`; `docs/FINAL_DEMO_VALIDATION.md` | Use live API/UI demo; no screenshot is required. |

**Screenshots:** No authentic Streamlit or MLflow UI screenshot is committed. The presentation source calls for live views or documented figures instead of fabricated captures. The official Samsung Innovation Campus logo supplied for the original proposal is not present in this repository; source it from that user-provided file when assembling the deck, without resizing its proportions or recoloring it.
