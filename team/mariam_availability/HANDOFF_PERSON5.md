# Person 5 Handoff

> **Integration correction (2026-10-28):** The examples below were owner-branch synthetic fixtures, not real Jiaxing sessions: charger IDs `201001`/`201005` are absent from the source. The listed local `artifacts/` files were excluded from Git and were unavailable on review. The integrated runtime uses `bundle/availability-v1/` rebuilt from the local raw Jiaxing source, and `tests/fixtures/real_jiaxing_*` for acceptance. See `docs/MARIAM_AVAILABILITY_REVIEW.md` for provenance and revised metrics. Historical example outputs below are retained as owner records, not integrated acceptance evidence.

## Imports and calls

From the repository root (all default paths resolve from package files):

```python
from team.mariam_availability.src.availability.inference import predict_availability
from team.mariam_availability.src.recommendation.service import recommend

prediction = predict_availability(payload)
result = recommend(payload)
```

Signatures: `predict_availability(payload: Mapping[str, Any], bundle_path: str | Path = DEFAULT_BUNDLE_PATH) -> dict[str, Any]`; `recommend(payload: Mapping[str, Any], bundle_path: str | Path = DEFAULT_BUNDLE_PATH, predictor: AvailabilityPredictor | None = None, risk_table_path: str | Path = DEFAULT_RISK_TABLE_PATH) -> dict[str, Any]`.

## Actual recommendation example

This request was run against the saved bundle/risk table. The optional Person 3 score is ignored because config disables it:

```json
{"location":"Technology Park","user_eta":"2021-10-15T12:30:00","candidates":[{"charger_id":"201001","busy":true,"session_start":"2021-10-15T11:50:00","anomaly_prediction":{"score":0.99,"anomaly":true},"history_features":{"hist_abnormal_rate":0.12,"hist_median_normal":41.0,"sessions_before":120,"has_history":1,"site_n_chargers":8,"site_busy_others":2,"site_busy_share":0.2857,"site_starts_last_60min":3}},{"charger_id":"201005","busy":false,"history_features":{"hist_abnormal_rate":0.08,"sessions_before":100,"has_history":1}}]}
```

Actual response (also saved as `tests/fixtures/recommendation_response.json`):

```json
{"recommended_charger":"201005","location":"Technology Park","expected_free_time":"2021-10-15T12:30:00","expected_wait_minutes":0.0,"recommended_slot":"2021-10-15T12:30:00","model_version":"recommendation-score-v3","bundle_version":"xgboost_default@e518e9c6d140","config_version":"recommendation-score-v3","schema_version":"recommendation-output-v1","risk_source":"historical_abnormal_rate","score":0.09542729424138838,"score_components":{"wait":0.0,"anomaly_risk":0.181709,"demand_pressure":0.5,"solar_benefit":0.5},"explanation":{"terms":{"wait":0.0,"anomaly_risk":0.1817091769655535,"demand_pressure":0.5,"solar_benefit":0.5},"weights":{"wait":0.45,"anomaly_risk":0.25,"demand_pressure":0.2,"solar_benefit":0.1},"weighted_terms":{"wait":0.0,"anomaly_risk":0.045427294241388375,"demand_pressure":0.1,"solar_benefit":-0.05},"score":0.09542729424138838,"dominant_factor":"demand_pressure","reason":"Free now; Lowest fault risk at this site; Highest solar benefit; Demand pressure has the largest weighted contribution."},"comparison":{"second_charger_id":"201006","decisive_term":"demand_pressure","reason":"demand pressure and solar benefit contributed equally to the difference versus the second charger."},"mock_signals_used":true,"free_pool_restricted":true,"reason":["free_chargers_preferred","available_at_driver_eta"],"alternatives":[{"charger_id":"201006","location":"Technology Park","expected_free_time":"2021-10-15T12:30:00","expected_wait_minutes":0.0,"recommended_slot":"2021-10-15T12:30:00","score":0.09542729424138839,"score_components":{"wait":0.0,"anomaly_risk":0.181709,"demand_pressure":0.4,"solar_benefit":0.3},"risk_source":"historical_abnormal_rate","mock_signals_used":false,"free_pool_restricted":true,"reason":["free_chargers_preferred","available_at_driver_eta"],"explanation":{"terms":{"wait":0.0,"anomaly_risk":0.1817091769655535,"demand_pressure":0.4,"solar_benefit":0.3},"weights":{"wait":0.45,"anomaly_risk":0.25,"demand_pressure":0.2,"solar_benefit":0.1},"weighted_terms":{"wait":0.0,"anomaly_risk":0.045427294241388375,"demand_pressure":0.08000000000000002,"solar_benefit":-0.03},"score":0.09542729424138839,"dominant_factor":"demand_pressure","reason":"Free now; Lowest fault risk at this site; Demand pressure has the largest weighted contribution."}}],"limitations":["Duration is predicted from the session start and is not adjusted for time already elapsed.","Demand and solar inputs are placeholders until Persons 1 and 2 deliver.","Anomaly risk is a historical abnormal rate, not a model prediction.","Labels are operator recorded, not verified hardware diagnoses."]}
```

Availability request used for the actual output fixture: `{"charger_id":"201001","location":"Technology Park","district":"Nanhu","session_start":"2021-10-15T12:00:00","as_of":"2021-10-15T12:30:00","temp_c":22.5,"history_features":{"hist_abnormal_rate":0.12,"hist_median_normal":41.0,"sessions_before":120,"has_history":1,"site_n_chargers":8,"site_busy_others":2,"site_busy_share":0.2857,"site_starts_last_60min":3}}`. The actual response is `tests/fixtures/availability_response.json`: 35.49 predicted total minutes, free at `12:35:29`, with 5.49 minutes remaining as of `12:30`.

## Runtime files to copy

All paths below are relative to `team/mariam_availability/`.

| File | Bytes | Purpose |
|---|---:|---|
| `artifacts/availability_duration_bundle_optuna_mlflow.joblib` | 2,687,667 | Saved XGBoost bundle/encoder |
| `artifacts/charger_risk_table.json` | 7,728 | Training-only rates; generated from `features_all.pkl` if absent |
| `config.yaml` | 626 | Ranker weights/version; `use_person3_score: false` |
| `schemas/availability_input.schema.json` | 1,969 | Availability request |
| `schemas/availability_output.schema.json` | 1,172 | Versioned availability response |
| `schemas/recommendation_input.schema.json` | 3,481 | Optional Person 3 payload; risk is not required |
| `schemas/recommendation_output.schema.json` | 3,200 | Versioned recommendation response |

Runtime code: `src/availability/{features,inference}.py`, `src/recommendation/{ranker,service,risk}.py`. `.gitignore` excludes all team artifacts (and `*.joblib`), so share the model bundle and risk JSON separately. `artifacts/features_all.pkl` is 95,022,278 bytes (>50 MB), ignored, and only needed to regenerate a missing risk table; sharing the risk JSON avoids needing it. Root `schemas/` was left untouched per the edit boundary; use these folder-local additive schemas.

## Risk and limits

The risk table uses sessions with `start < 2021-07-01`, contains per-charger rate/count plus the global rate, and falls back to global for unseen chargers. `risk_source` identifies the selected candidate's risk. Person 3 is not used by default; enabling it requires `use_person3_score: true`. Demand/solar are neutral `0.5` placeholders. Other limits: duration is based on session start, not elapsed-time-adjusted; abnormal labels are operator-recorded, not verified hardware diagnoses.

Tests: from this folder run `python -m pytest tests -q -p no:cacheprovider`.

## XAI fields

`recommend()` now returns an `explanation` with normalized terms, config weights, signed weighted contributions, exact score, dominant contribution, and a plain-language reason. When a second eligible charger exists, `comparison` identifies the term with the largest weighted difference. Each alternative has its own score breakdown. Example fields below are taken from the real response fixture: `dominant_factor` is `demand_pressure`; the comparison with charger `201006` shows demand pressure and solar benefit contributed equally to the score difference.

Availability requests may add `"explain": true`. The saved XGBoost model then returns the base value and top five local feature contributions in log-minute units. The complete contributions (not just the displayed top five) add up to the model's raw output. For other estimator types, the response has `explanation: null` with an explanation note. The default (`explain` omitted/false) response is unchanged. See `tests/fixtures/availability_explain_response.json` for a real result.

Notebook `notebooks/08_xai_shap.ipynb` performs global and local SHAP analysis on a deterministic sample of validation-normal sessions without training or saving a model. It needs the optional `shap` package in the notebook environment. SHAP is not in runtime requirements.
