# Deterministic Digital Twin

The Digital Twin is a versioned what-if calculator for explicit user-provided EV hub state. It is not an ML model, live grid controller, charger controller, or source of real forecasts.

## Inputs and formulas

Inputs are baseline solar/demand, total and active charger counts, average session duration, demand/solar multipliers, and unique offline charger IDs.

- Scenario solar = baseline solar × solar multiplier.
- Scenario demand = baseline demand × demand multiplier.
- Active chargers = baseline active chargers − offline charger count.
- Renewable balance = scenario solar − scenario demand. A negative value means grid energy may cover the renewable deficit; it does **not** mean the hub loses power.
- Renewable coverage = scenario solar ÷ scenario demand; it is `null` when demand is zero.
- Normalized utilization = `0.50 × demand pressure ÷ remaining-fleet ratio`, bounded to 0–1. The 0.50 reference is a transparent neutral heuristic, not measured charger utilization.
- Simulated wait impact = average session minutes × `max(demand pressure − remaining-fleet ratio, 0)`. It is a relative scenario estimate, not a predicted queue time.

## Severity

Thresholds live in `src/digital_twin/config.py`. Utilization of 0.65/0.80/0.95 and renewable-deficit ratios of 0.50/0.75/0.90 determine MEDIUM/HIGH/CRITICAL severity. Any reduction from full fleet availability is at least MEDIUM, 50% remaining is HIGH, 25% remaining is CRITICAL, and any zero-active-charger scenario is CRITICAL. Each severity maps to a fixed operational simulation recommendation.

## Limitations and future flow

All outputs are labeled simulations and must not be interpreted as live grid-control decisions. PVGIS solar is modeled/reference data, not real metered station generation. The present endpoint uses only explicit request values; it does not pretend they came from models.

```text
Solar Forecast ─┐
Demand Forecast ├─> Digital Twin
Availability ───┤
Anomaly State ──┘
```

These four inputs may be connected only after their owners deliver validated, versioned artifacts and schemas.
