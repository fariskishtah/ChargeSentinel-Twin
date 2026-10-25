# EV Charging Station Simulator v1

## Purpose

The simulator is a deterministic demonstration layer for a configurable EV charging hub. It models arrivals, charger assignment, charging progress, queueing, completion, simulated faults, and offline chargers. It does not control hardware, implement OCPP, consume live telemetry, or generate ML predictions.

It complements the existing Digital Twin:

- Charging Station Simulator → produces simulated station and session state.
- Digital Twin → evaluates explicit what-if operating scenarios.

## State and configuration

The default hub has six chargers rated at an explicit simulation default of 22 kW each. Reset accepts charger count, charger power, manual solar context, and manual station-demand context. These values are demonstration inputs and are not hardware specifications or forecasts.

Charger states and allowed transitions are centralized in `src/digital_twin/charging_station.py`:

```text
AVAILABLE -> RESERVED | CHARGING | OFFLINE
RESERVED  -> CHARGING
CHARGING  -> AVAILABLE | FAULT
FAULT     -> AVAILABLE
OFFLINE   -> AVAILABLE
```

Each EV session records its ID, arrival time, initial/current/target SOC, battery capacity, maximum charging power, assigned charger, start/end time, delivered energy, and status (`WAITING`, `CHARGING`, `COMPLETED`, or `CANCELLED`).

## Deterministic charging and time

The engine uses a manual simulation clock. State changes only through an explicit `advance_time(minutes)` call; there are no timers, background threads, random values, or wall-clock dependencies.

```text
energy needed (kWh) = capacity × (target SOC − initial SOC) / 100
charging power (kW) = min(EV maximum power, charger maximum power)
duration (minutes) = energy needed / charging power × 60
```

SOC and delivered energy advance from these formulas. When a session reaches its target SOC, it becomes `COMPLETED`, the charger becomes `AVAILABLE`, and the next waiting EV is assigned during the same simulation step.

## Queue, assignment, and faults

The queue is FIFO. The oldest waiting EV is assigned to the first compatible available charger in configured charger order. This simple assignment method is the future override point for Mariam's recommendation ranker.

A simulated fault can be injected only while a charger is `CHARGING`. Charging stops, the charger moves to `FAULT`, and the interrupted EV keeps its delivered energy and returns to the front of the queue. If another charger is available, normal queue assignment may resume it immediately. Clearing the fault restores availability and serves the queue. Only an `AVAILABLE` charger can be taken offline; restoring an offline charger also serves the queue.

Every fault is labeled `SIMULATED FAULT`; it is not output from Fatima's anomaly model.

## API

All routes use the existing FastAPI service:

- `POST /api/v1/simulator/reset`
- `GET /api/v1/simulator/state`
- `POST /api/v1/simulator/ev`
- `POST /api/v1/simulator/advance`
- `POST /api/v1/simulator/chargers/{charger_id}/fault`
- `POST /api/v1/simulator/chargers/{charger_id}/clear-fault`
- `POST /api/v1/simulator/chargers/{charger_id}/offline`
- `POST /api/v1/simulator/chargers/{charger_id}/restore`

Mutation endpoints return the complete updated state. Invalid request data returns `422`, an unknown charger returns `404`, and an impossible transition returns `409`.

## Limitations

- State is in memory, ephemeral, and resets when the API process restarts.
- v1 is one shared demo instance; it does not isolate multiple users or demos.
- There is no persistence, authentication boundary, OCPP integration, hardware control, live telemetry, automatic clock, or ML inference.
- Manual solar and demand values are context only and do not alter charging progression in v1.
- `RESERVED` and `CANCELLED` are defined for the state model but have no user action in v1.

## Future model integration

- Mazen: solar forecast → renewable context.
- Waseem: demand forecast → arrival and demand-pressure context.
- Fatima: anomaly output → anomaly/fault status after validation.
- Mariam: availability output → estimated free time; recommendation ranker → charger selection.

No unavailable model is connected or simulated in v1.
