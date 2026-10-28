# Duration model held-out evaluation

Data: `Charging_Data_clean.csv` (441,064 cleaned rows; SHA-256 `e518e9c6d140aa7844b2f7d77ed572acfe7b970b2a62d977f759148f7703a6bb`).
Chronological train and validation boundaries come from `src/features.py`; the held-out partition starts at 2021-10-01 and contains 80,072 rows.

| Model | Normal test MAE (min) | Normal test RMSE (min) | All-session test MAE (min) |
|---|---:|---:|---:|
| Median baseline | 35.520 | 84.781 | 36.445 |
| xgboost_default | 34.948 | 83.049 | 35.172 |

The saved champion beats the median baseline on normal-session held-out MAE.

The test partition was scored before the Optuna tuning work. These results are retained for handoff transparency and are not a pristine untouched final estimate. Normal-session metrics exclude sessions later labelled abnormal; all-session MAE is included to show the effect of early fault-terminated sessions.

Detailed metrics are in `availability_duration_test_comparison.csv`.
