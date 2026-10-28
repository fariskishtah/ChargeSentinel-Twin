# Solar and anomaly integration handoff

Source branches were fetched on 2026-10-27. All hardening and integration lives on `feature/faris-integration`; the owner branches remain unchanged.

| Module | Read-only source | Portable handoff | Public endpoint |
|---|---|---|---|
| Anomaly | `origin/feature/fatima-anomaly` at `e9e953f3d206067d3eeb9dd9821edb3dbe4ac2f5` | `team/fatima_anomaly/bundle/anomaly-v1/` | `POST /api/v1/anomaly/predict` |
| Solar | `origin/feature/mazen-solar` at `bfc088839fce1229c9f3045ab44e6ffedc0d1c37` | `team/mazen_solar/bundle/solar-v1/` | `POST /api/v1/solar/forecast` |

## Anomaly: session-start risk

The current operator UI describes a live check, so inference is at **session start**. The prediction is risk of a later abnormal end-cause, not a detected or confirmed fault. Inputs are charger ID, location, district, and a naive source-operational start timestamp. `End Time`, completed duration, final energy/rate, payment, `end_cause`, and all outcome flags are excluded. The public Pydantic schema rejects extra features. Jiaxing source timezone is unverified.

Fatima's source code had no committed model or metrics, used `train_test_split`, fixed threshold 0.41, post-event duration/rate features, and an explanation fallback that swallowed SHAP failures. Its tests used fake models; the `.ipynb` file is a Colab-dependent script. The integration handoff at `team/fatima_anomaly/integration/` retrains CatBoost from the real local Jiaxing source with a strict chronological train/validation/test split and a validation-only threshold sweep. No notebook, Drive path, or local MLflow store is needed for inference.

On 2026-10-28, Fatima's later PR #1 added an archived model ZIP from that earlier completed-session methodology. The ZIP and extracted model checksums matched the reconciliation audit; the PR was professionally closed without merge, and her source branch/history remain intact. Its higher random-split scores are not a replacement for `anomaly-v1`. A separate [grid-load interval screen](GRID_LOAD_ANOMALY.md) now addresses the handbook's interval anomaly wording without changing the session-risk contract.

| Split | First to last source timestamp | Rows | Normal | Abnormal | Prevalence |
|---|---|---:|---:|---:|---:|
| Train | 2019-12-31 21:27 to 2021-07-30 14:43 | 308,753 | 252,267 | 56,486 | 18.29% |
| Validation | 2021-07-30 14:43 to 2021-10-16 12:01 | 66,162 | 52,018 | 14,144 | 21.38% |
| Test | 2021-10-16 12:02 to 2021-12-31 23:52 | 66,162 | 54,858 | 11,304 | 17.09% |

Split boundaries use exact timestamps in `bundle/anomaly-v1/config.json`; the table is abbreviated to minutes. CatBoost uses `auto_class_weights="Balanced"`. **SMOTE is intentionally not used; class imbalance is handled through training-time class weighting.** A location-prevalence rule trained only on the train split provides a meaningful reference. The selected threshold is **0.40**, maximizing validation F2 among thresholds with precision at least the validation base rate. The full precision, recall, F1, FP, and FN sweep is in `reports/integration_hardening/threshold_curve.csv`. It was frozen before test scoring.

| Held-out test | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---|
| CatBoost `anomaly-v1` | 0.215 | 0.888 | 0.346 | 0.642 | 0.238 | 18,147 / 36,711 / 1,267 / 10,037 |
| Location prevalence rule | 0.226 | 0.815 | 0.354 | 0.632 | 0.222 | 23,370 / 31,488 / 2,094 / 9,210 |

CatBoost was chosen for the recall objective and higher PR-AUC, despite lower precision and F1 than the rule. The 36,711 false positives are a major limitation. The UI labels the result as an investigation flag. The source `is_abnormal` label is derived from `end_cause`, not an independent production fault label. There is no Egyptian or real-time deployment validation.

The portable bundle contains the `.cbm` model, model SHA256, feature/config metadata, threshold, input/output JSON schemas, a real held-out test fixture, dependencies, and limitations. CatBoost's real model SHAP values produce global importance (`reports/integration_hardening/shap_global_importance.csv`) and a local reason (`real_xai_example.json`). Explanation failure causes an unavailable response; no placeholder reason is returned. The `chargesentinel-anomaly` MLflow run ID is recorded in `reports/integration_hardening/mlflow_run.json`; the local tracking store is supplementary to the committed bundle.

## Solar: modelled New Cairo PV reference

Mazen's latest branch includes distinct XGBoost 24h/48h joblib artifacts, LSTM artifacts, source code, plots, and an evaluation report. Its README still says training has not started, its report has Windows `file:///` links, and `src.solar` imports do not match its `team/mazen_solar/src/solar` location. The generic XGBoost file is an ambiguous default. Its June holdout report shows seasonal persistence outperforming XGBoost. These historical source results are retained as source evidence, not copied as acceptance metrics.

Integration evaluation uses Mazen's feature code under the actual `team.mazen_solar` namespace, with three safety fixes: missing PVGIS targets are excluded rather than interpolated; lead hours 24–48 use the earlier `_previous_day2` weather run; and PVGIS readings stamped at `HH:09` are never used as `HH:00` origin history. PVGIS values retain Mazen's ordinal alignment to Open-Meteo's UTC H1 labels; two target hours are missing. Historical PV is strictly **before** the requested origin. At lead hours 24 and 48, seasonal persistence steps back an additional day rather than reading the origin hour. Train, validation, and June test target times do not cross split boundaries. The validation comparison is chronological: train through April, select on May, evaluate the chosen model once on June.

| Horizon | May validation MAE (kW): persistence / Ridge / XGBoost | Selected champion | June test MAE (kW) | Model version |
|---|---|---|---:|---|
| 24h | 1.623 / 1.636 / 2.236 | Seasonal Persistence | 0.361 | `solar-seasonal-persistence-24h-v1` |
| 48h | 1.794 / 1.756 / 2.231 | Ridge Regression | 1.257 | `solar-ridge-regression-48h-v1` |

The 24h split contains 400/124/116 train/validation/test origins and 9,560/2,924/2,772 evaluated target rows. The 48h split contains 200/62/56 origins and 9,524/2,890/2,679 rows. The reduced counts reflect the two excluded missing target hours and origin spacing (6h for 24h, 12h for 48h).

The source report's XGBoost SHAP plots explain the XGBoost candidate; they are **not** explanations of the shipped champions. Persistence is directly interpretable as a prior observed PV hour. Ridge coefficients and scaling parameters are in the 48h JSON artifact. Solar output is bounded to 0–100 kW and zero at astronomical night. The public response retains the original scalar first-hour fields and adds `points` for the full curve and `model_family`.

The two horizon artifacts have different model versions and SHA256 digests. `bundle/solar-v1/checksums.json` verifies both JSON artifacts and the small, derived H1 PVGIS/Open-Meteo runtime assets. No raw source dataset or machine-specific path is in the Docker image. The bundle contains schemas, dependencies, and limitations. New local `chargesentinel-solar` MLflow run IDs, split row/origin counts, all family validation MAEs, and June results are in `reports/integration_hardening/selection.json`. The source `runs:/` URIs are not a serving dependency.

Forecasts are a historical 2024 H1 **modelled/reference PV** demonstration, not measured station output or a live weather service. Full documentation of limitations is in both bundle `limitations.md` files.

## Reproduction and contracts

```text
python -m team.fatima_anomaly.integration.train /absolute/path/to/Charging_Data.csv
python -m team.mazen_solar.integration.build_handoff /absolute/path/to/pvgis.json /absolute/path/to/openmeteo.json
```

Training dependencies include CatBoost, scikit-learn, XGBoost, and MLflow. Runtime inference needs only the pinned API dependencies in `requirements/api.txt`; the committed bundles are loaded from module-relative paths. Anomaly valid input:

```json
{"charger_id":"12","features":{"location":"Financial Industrial Park","district":"Nanhu","start_time":"2021-10-16T12:02:44"}}
```

Solar valid input:

```json
{"forecast_origin":"2024-06-15T00:00:00+00:00","horizon_hours":48}
```

Unsupported origins return 422. Missing or corrupt model/runtime assets return a sanitized 503. No fake predictions or explanations are produced. Demand, Digital Twin, simulator, availability, and recommendation behavior are unchanged.

Final gate on 2026-10-27: `RUN_DOCKER_TESTS=1 ... python -m pytest tests -q` reported **150 passed, 0 failed**. The Compose API, UI, and MLflow services were healthy; `deployment/smoke_test.py` passed real demand, both solar horizons, anomaly, Digital Twin, and internal UI-to-API checks. The API container loaded the portable artifacts from `/tmp` working directory without host data paths.
