# Availability recommendation offline replay

## Method

The replay evaluated 39,204 real session starts from the held-out period 2021-10-01 00:01:34 to 2021-12-31 23:52:14, restricted to 11:00–14:59 and 19:00–22:59 local time. The clean dataset SHA-256 is `e518e9c6d140aa7844b2f7d77ed572acfe7b970b2a62d977f759148f7703a6bb`. At least one charger was free at 100.0% of those arrivals.

Ranker version `recommendation-score-v4` uses weights {'wait': 0.45, 'anomaly_risk': 0.25, 'demand_pressure': 0.2, 'solar_benefit': 0.1} for normalized wait, train-period charger anomaly rate, train-period charger/hour demand pressure, and neutral solar placeholder. If any candidate is actually free, it scores only free candidates. First-free chooses the lexicographically first currently free charger. Random-free chooses a currently free charger using seed 2026. When every charger is busy, the two baselines select the soonest model-predicted release, with charger ID as tie-break. Realized wait and subsequent fault labels are scoring-only.

## Results

| Policy | Mean wait (min) | Median wait (min) | P95 wait (min) | Selected free | Assignment CV | Assignment Gini | Later-fault share* |
|---|---:|---:|---:|---:|---:|---:|---:|
| ranker | 0.00 | 0.00 | 0.00 | 100.0% | 1.387 | 0.653 | 23.2% |
| first_free | 0.00 | 0.00 | 0.00 | 100.0% | 1.376 | 0.652 | 16.5% |
| random_free | 0.00 | 0.00 | 0.00 | 100.0% | 0.433 | 0.241 | 19.9% |

\*For a charger busy at arrival, this is the active session's later abnormal label. For a free charger, it is the next observed session's label. The latter is a proxy outcome, not a real counterfactual session.

Utilization balance is reported as assignment concentration (CV and Gini) across chargers during this arrival replay; it is not a measured change in physical utilization.

## Limitations

- Counterfactual replay: observed real starts are used as arrivals, but choices can differ from actual drivers.
- There is no queue data; drivers who found every charger busy and left are unobserved.
- Busy state at arrival is treated as observable. Predicted free time uses duration inference; actual end time is scoring-only.
- Historical demand-by-hour is a proxy, solar benefit is a neutral configured mock, and risk is a training-period charger rate.
- For a free charger, later-fault scoring uses its next observed session as a proxy outcome, not the counterfactual session created by this arrival.
- Only held-out test-period arrivals in the specified peak windows are included.

This is a policy comparison under observed occupancy and a duration model. It does not estimate demand from drivers who abandoned a busy site and does not establish causal effects of recommendations.
