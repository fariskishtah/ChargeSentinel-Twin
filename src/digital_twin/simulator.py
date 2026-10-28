"""Deterministic what-if simulation for the EV charging hub."""

from collections.abc import Mapping

from app.api.schemas import DigitalTwinRequest, DigitalTwinResponse
from src.digital_twin.config import (
    CRITICAL_CHARGER_AVAILABILITY_FLOOR,
    CRITICAL_RENEWABLE_DEFICIT,
    CRITICAL_UTILIZATION,
    DIGITAL_TWIN_VERSION,
    HIGH_CHARGER_AVAILABILITY_FLOOR,
    HIGH_RENEWABLE_DEFICIT,
    HIGH_UTILIZATION,
    MEDIUM_CHARGER_AVAILABILITY_FLOOR,
    MEDIUM_RENEWABLE_DEFICIT,
    MEDIUM_UTILIZATION,
    NORMALIZED_BASELINE_UTILIZATION,
    OUTPUT_PRECISION,
    RECOMMENDATIONS,
    Severity,
)


def _bounded_utilization(
    scenario_demand_kw: float,
    baseline_demand_kw: float,
    charger_availability_ratio: float,
) -> float:
    """Return a 0–1 normalized load index without assuming charger power ratings."""

    if scenario_demand_kw == 0:
        return 0.0
    if charger_availability_ratio == 0:
        return 1.0

    demand_pressure = scenario_demand_kw / baseline_demand_kw
    utilization = (
        NORMALIZED_BASELINE_UTILIZATION
        * demand_pressure
        / charger_availability_ratio
    )
    return min(max(utilization, 0.0), 1.0)


def _wait_impact_minutes(
    average_session_minutes: float,
    demand_pressure: float,
    charger_availability_ratio: float,
) -> float:
    """Return a relative simulated delay heuristic, never a predicted wait time."""

    pressure_gap = max(demand_pressure - charger_availability_ratio, 0.0)
    return average_session_minutes * pressure_gap


def _severity(
    *,
    scenario_active_chargers: int,
    utilization: float,
    renewable_deficit_ratio: float,
    charger_availability_ratio: float,
) -> Severity:
    """Classify a scenario using the centralized deterministic thresholds."""

    if (
        scenario_active_chargers == 0
        or utilization >= CRITICAL_UTILIZATION
        or renewable_deficit_ratio >= CRITICAL_RENEWABLE_DEFICIT
        or charger_availability_ratio <= CRITICAL_CHARGER_AVAILABILITY_FLOOR
    ):
        return "CRITICAL"
    if (
        utilization >= HIGH_UTILIZATION
        or renewable_deficit_ratio >= HIGH_RENEWABLE_DEFICIT
        or charger_availability_ratio <= HIGH_CHARGER_AVAILABILITY_FLOOR
    ):
        return "HIGH"
    if (
        utilization >= MEDIUM_UTILIZATION
        or renewable_deficit_ratio >= MEDIUM_RENEWABLE_DEFICIT
        or charger_availability_ratio < MEDIUM_CHARGER_AVAILABILITY_FLOOR
    ):
        return "MEDIUM"
    return "LOW"


def simulate(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate and evaluate one explicit, deterministic simulation scenario."""

    request = DigitalTwinRequest.model_validate(payload)

    scenario_solar_kw = request.baseline_solar_kw * request.solar_multiplier
    scenario_demand_kw = request.baseline_demand_kw * request.demand_multiplier
    scenario_active_chargers = request.active_chargers - len(
        request.offline_chargers
    )
    supply_demand_balance_kw = scenario_solar_kw - scenario_demand_kw

    renewable_coverage_ratio = (
        None
        if scenario_demand_kw == 0
        else scenario_solar_kw / scenario_demand_kw
    )
    renewable_deficit_ratio = (
        0.0
        if scenario_demand_kw == 0
        else max(scenario_demand_kw - scenario_solar_kw, 0.0)
        / scenario_demand_kw
    )
    charger_availability_ratio = (
        scenario_active_chargers / request.total_chargers
    )
    demand_pressure = (
        0.0
        if request.baseline_demand_kw == 0
        else scenario_demand_kw / request.baseline_demand_kw
    )

    utilization = _bounded_utilization(
        scenario_demand_kw,
        request.baseline_demand_kw,
        charger_availability_ratio,
    )
    wait_impact = _wait_impact_minutes(
        request.average_session_minutes,
        demand_pressure,
        charger_availability_ratio,
    )
    severity = _severity(
        scenario_active_chargers=scenario_active_chargers,
        utilization=utilization,
        renewable_deficit_ratio=renewable_deficit_ratio,
        charger_availability_ratio=charger_availability_ratio,
    )

    response = DigitalTwinResponse(
        scenario_solar_kw=round(scenario_solar_kw, OUTPUT_PRECISION),
        scenario_demand_kw=round(scenario_demand_kw, OUTPUT_PRECISION),
        supply_demand_balance_kw=round(
            supply_demand_balance_kw, OUTPUT_PRECISION
        ),
        renewable_coverage_ratio=(
            None
            if renewable_coverage_ratio is None
            else round(renewable_coverage_ratio, OUTPUT_PRECISION)
        ),
        scenario_active_chargers=scenario_active_chargers,
        estimated_utilization=round(utilization, OUTPUT_PRECISION),
        estimated_wait_impact_minutes=round(wait_impact, OUTPUT_PRECISION),
        severity=severity,
        recommendation=RECOMMENDATIONS[severity],
        simulation_version=DIGITAL_TWIN_VERSION,
        is_simulation=True,
    )
    return response.model_dump(mode="json")
