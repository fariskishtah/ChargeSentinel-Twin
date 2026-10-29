# Explainability report

## Recommendation explanation

For each candidate, the service reports four normalized input terms and their configured weights. Its displayed score is the sum of the signed weighted contributions: wait, historical abnormal rate, and demand add to the score; solar benefit subtracts. The dominant factor is the largest contribution by absolute value. If at least two eligible candidates remain after the free-charger preference, the response compares the winner with the runner-up and names the term with the largest absolute difference in weighted contribution. Example output is in `tests/fixtures/recommendation_response.json`; its winner is charger `201005`, with demand pressure as the largest single weighted contribution. Demand pressure and solar benefit contribute equally to the winner/runner-up difference in this example.

The current response also includes `candidate_eligibility` (including location/status exclusions, free-pool filtering, and estimated wait), `decision_explanation` (selection rule, score formula, and winner-versus-runner-up summary), and `signal_sources` (demand, solar, and risk provenance). These fields explain the full decision path; SHAP in notebook 8 remains specific to the availability-duration model and does not attribute the ranker score.

## Availability explanation

With `explain: true`, the saved XGBoost bundle returns its base value and the five strongest local feature contributions. Contributions are in the model target space (`log_minutes`), before conversion to minutes. The complete TreeSHAP vector is checked for additivity against the model's raw output using the saved estimator's best iteration. The normal response is unchanged when explanation is omitted or false. Non-XGBoost bundles return a null explanation and note. A real output is in `tests/fixtures/availability_explain_response.json`.

## Notebook analysis

`notebooks/08_xai_shap.ipynb` is prepared to load saved validation artifacts and the exported XGBoost model, sample at most 2,000 validation-normal sessions, verify SHAP/native contribution agreement, and produce global, dependence, local waterfall, feature-group and gain-comparison views. It does not train or export a model.

Execution status: **not completed in this environment**. The optional `shap` package is unavailable and package installation is blocked by this network-restricted environment. Notebook execution was also blocked when the Jupyter kernel hit a Windows `SetFileSecurity: Access is denied` error. No SHAP values or plots are claimed here. Run the first notebook cell in an environment with `shap` installed, then execute all cells from top to bottom. SHAP is intentionally not added to runtime requirements.

## Limitations

- Attributions explain model behavior; they do not establish causal effects.
- Integer-coded charger/location identifiers can produce feature attributions that are artifacts of encoding; correlated features can share attribution credit.
- Local availability attributions are for the saved model output in log minutes; they do not explain post-transform effects in minutes.
- Recommendation anomaly risk defaults to a training-period historical abnormal rate, not a Person 3 model prediction.
- Demand pressure now consumes Person 2's expected-session output when supplied, or forecasts from the bundled Jiaxing hourly history during its 2020-2021 coverage. Solar benefit consumes Person 1's modelled New Cairo forecast when supplied; this is explicitly simulated for Jiaxing and is not local station measurement. Missing or out-of-coverage inputs still use disclosed neutral placeholders. The prior example fixture in this report predates that integration.
- Abnormal-session labels are operator recorded and are not verified hardware diagnoses.
