# Time-aware split definitions

All listed tests are held out after model or threshold selection. Temporal order matters because nearby charging and weather records are correlated; random temporal splits can inflate results.

| Component | Train/reference | Validation/selection | Held-out test | Source evidence |
|---|---|---|---|---|
| Demand `demand-completion-v1` | 2020-01-08 20:00–2021-06-30 23:00; 12,940 rows | 2021-07-01–2021-09-30; 2,208 rows | 2021-10-01–2021-12-31; 2,208 rows | `team/waseem_demand/bundle/v1/metadata.json`; baseline, XGBoost, LSTM comparison. |
| Solar 24h | 2024-01-22–2024-04-30; 400 forecast origins | May 2024; 124 origins | June 2024; 116 origins | `team/mazen_solar/reports/integration_hardening/selection.json`; May chooses champion, June reports once. |
| Solar 48h | 2024-01-22–2024-04-30; 200 origins | May 2024; 62 origins | June 2024; 56 origins | Same selection report; origin spacing prevents target crossing. |
| EV session risk `anomaly-v1` | 2019-12-31 21:27:03–2021-07-30 14:43:40; 308,753, abnormal 18.29% | 2021-07-30 14:43:55–2021-10-16 12:01:57; 66,162, abnormal 21.38% | 2021-10-16 12:02:44–2021-12-31 23:52:14; 66,162, abnormal 17.09% | `team/fatima_anomaly/bundle/anomaly-v1/config.json`; 0.40 threshold selected on validation only. |
| Grid-load interval rule | 2020-02-25 21:00–2020-12-31 23:00; 7,443 scorable hours | 2021-01-01–2021-06-30; 4,344 hours | 2021-07-01–2021-12-31; 4,416 hours | `team/fatima_anomaly/reports/grid_load/comparison.json`. Eight earlier same-week hours can cross a split boundary, but only backward in time, as a live historical lookup would. |

Solar train/validation/test target row counts and exact origin bounds are in the selection report. The grid-load injected validation and held-out test windows are enumerated separately in `team/fatima_anomaly/reports/grid_load/*_injections.csv`; those rows do not alter the real hourly history. No real hourly grid-fault labels exist in this evidence.
