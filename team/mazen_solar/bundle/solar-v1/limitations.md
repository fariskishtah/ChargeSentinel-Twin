# Solar v1 limitations

The source target is PVGIS modelled power for a 100 kWp New Cairo reference system, not measured station generation. Runtime history and weather are small derived 2024 H1 assets. Forecast origins are limited to the period where complete historical PV and required previous-run weather inputs exist; missing input is rejected without a synthetic replacement. The two missing PVGIS target hours are excluded from evaluation and never interpolated. Raw PVGIS readings are stamped at HH:09; the HH:00-labelled reading is unavailable at an HH:00 forecast origin and is excluded from historical features.

At 24 hours the May validation winner is seasonal persistence. At 48 hours Ridge Regression wins the May validation comparison. The 48-hour model uses the `_previous_day2` weather run for lead hours 24–48; day-ahead values for those hours would not yet be available at origin. Seasonal persistence uses an additional prior day at lead hours 24 and 48 to keep all PV history strictly before origin. June results are reporting only. The previous Mazen report's XGBoost SHAP plots explain the source XGBoost candidate, not these shipped champions.

Weather-run issue time is inferred from Open-Meteo previous-run field semantics; the archived data is a historical demonstration, not a live weather feed. No Egyptian production plant has been used for validation.
