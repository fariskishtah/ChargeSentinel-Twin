"""Dedicated API routes for the deterministic charging station simulator."""

from collections.abc import Callable

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    SimulatorAdvanceRequest,
    SimulatorEVRequest,
    SimulatorResetRequest,
    SimulatorStateResponse,
)
from src.digital_twin.charging_station import (
    ChargingStationSimulator,
    SimulatorConflictError,
    SimulatorNotFoundError,
    SimulatorValidationError,
)

router = APIRouter(prefix="/simulator", tags=["simulator"])
station_simulator = ChargingStationSimulator()


def _execute(action: Callable[[], dict[str, object]]) -> SimulatorStateResponse:
    try:
        state = action()
    except SimulatorNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SimulatorConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SimulatorValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SimulatorStateResponse.model_validate(state)


@router.post("/reset", response_model=SimulatorStateResponse)
def reset_simulator(request: SimulatorResetRequest) -> SimulatorStateResponse:
    return _execute(
        lambda: station_simulator.reset(
            charger_count=request.charger_count,
            charger_power_kw=request.charger_power_kw,
            current_solar_kw=request.current_solar_kw,
            current_station_demand_kw=request.current_station_demand_kw,
        )
    )


@router.get("/state", response_model=SimulatorStateResponse)
def get_simulator_state() -> SimulatorStateResponse:
    return _execute(station_simulator.get_state)


@router.post("/ev", response_model=SimulatorStateResponse)
def add_simulated_ev(request: SimulatorEVRequest) -> SimulatorStateResponse:
    return _execute(
        lambda: station_simulator.add_ev(
            ev_id=request.ev_id,
            initial_soc_percent=request.initial_soc_percent,
            target_soc_percent=request.target_soc_percent,
            battery_capacity_kwh=request.battery_capacity_kwh,
            max_charge_power_kw=request.max_charge_power_kw,
        )
    )


@router.post("/advance", response_model=SimulatorStateResponse)
def advance_simulation(request: SimulatorAdvanceRequest) -> SimulatorStateResponse:
    return _execute(lambda: station_simulator.advance_time(request.minutes))


@router.post(
    "/chargers/{charger_id}/fault",
    response_model=SimulatorStateResponse,
)
def inject_charger_fault(charger_id: str) -> SimulatorStateResponse:
    return _execute(lambda: station_simulator.inject_fault(charger_id))


@router.post(
    "/chargers/{charger_id}/clear-fault",
    response_model=SimulatorStateResponse,
)
def clear_charger_fault(charger_id: str) -> SimulatorStateResponse:
    return _execute(lambda: station_simulator.clear_fault(charger_id))


@router.post(
    "/chargers/{charger_id}/offline",
    response_model=SimulatorStateResponse,
)
def take_charger_offline(charger_id: str) -> SimulatorStateResponse:
    return _execute(lambda: station_simulator.take_offline(charger_id))


@router.post(
    "/chargers/{charger_id}/restore",
    response_model=SimulatorStateResponse,
)
def restore_charger(charger_id: str) -> SimulatorStateResponse:
    return _execute(lambda: station_simulator.restore_charger(charger_id))
