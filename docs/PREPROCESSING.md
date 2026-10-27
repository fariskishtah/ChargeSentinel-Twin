# Preprocessing and leakage controls

## Demand

`team/waseem_demand/demand/build_hourly.py` validates Jiaxing transactions and builds continuous hourly history. Completed-session energy is credited to **End Time hour**; session arrivals to **Start Time hour**. An hour with no event is zero only under the complete source-capture assumption. `demand/features.py` shifts before computing lags (1/2/3/24/48/168h) and rolling means/std (6/24h), so the target hour is never an input. Scaling and fitting use training data only. The runtime bundle and 192h history contract remain frozen.

## Solar

`team/mazen_solar/integration/source_features.py` aligns PVGIS HH:09 modelled/reference readings ordinally to UTC H1 Open-Meteo labels. The HH:09 reading is unavailable at an HH:00 forecast origin; only earlier PV history is used. Forecast weather comes from available `_previous_day1` or `_previous_day2` runs at each lead time, not realized future weather. Two missing PVGIS target hours are excluded rather than filled. Output is bounded to 0–100 kW and zero at astronomical night. May selects 24h/48h champions; June is held out.

## EV session risk

`team/fatima_anomaly/integration/features.py` constructs only charger ID, location, district, and session-start hour/day/month/weekend. End Time, duration, final energy/rate, payment, end cause, and `is_*` outcomes are excluded. `integration/train.py` uses chronological splits, CatBoost Balanced weighting without SMOTE, and a validation-only threshold sweep. The 0.40 threshold is frozen before the test. `integration/inference.py` verifies artifact and feature checksums, then returns real CatBoost SHAP contributions; explanation failure is unavailable rather than fabricated.

## Grid-load interval screen

`src/grid_load/detector.py` compares each observed hour with eight **strictly earlier** matching hour-of-week records. Median and MAD-based scales avoid a learned preprocessing binary. The score is the largest absolute standardized energy or arrival deviation. The rule threshold is fixed at the validation 99th percentile; July–December 2021 is test only. `src/grid_load/train.py` trains the Isolation Forest comparison on 2020 causal features, evaluates both on the same chronological windows, and writes separately labeled injected sensitivity cases. The production endpoint serves only recorded historical intervals; it cannot fabricate a grid-fault diagnosis.
