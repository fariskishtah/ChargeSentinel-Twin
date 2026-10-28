"""Deterministic tests for the EV charging station simulator engine."""

import pytest

from src.digital_twin.charging_station import (
    SIMULATOR_VERSION,
    ChargerStatus,
    ChargingStationSimulator,
    SimulatorConflictError,
    SimulatorValidationError,
)


def add_ev(
    simulator: ChargingStationSimulator,
    ev_id: str,
    *,
    initial_soc: float = 20,
    target_soc: float = 80,
    battery_capacity: float = 60,
    max_power: float = 11,
) -> dict[str, object]:
    return simulator.add_ev(
        ev_id=ev_id,
        initial_soc_percent=initial_soc,
        target_soc_percent=target_soc,
        battery_capacity_kwh=battery_capacity,
        max_charge_power_kw=max_power,
    )


def short_session(simulator: ChargingStationSimulator, ev_id: str) -> None:
    add_ev(
        simulator,
        ev_id,
        initial_soc=0,
        target_soc=10,
        battery_capacity=10,
        max_power=10,
    )


def session(state: dict[str, object], ev_id: str) -> dict[str, object]:
    sessions = state["sessions"]
    assert isinstance(sessions, list)
    return next(item for item in sessions if item["ev_id"] == ev_id)


def charger(state: dict[str, object], charger_id: str) -> dict[str, object]:
    chargers = state["chargers"]
    assert isinstance(chargers, list)
    return next(item for item in chargers if item["charger_id"] == charger_id)


def test_reset_creates_expected_charger_count() -> None:
    simulator = ChargingStationSimulator(charger_count=3, charger_power_kw=50)

    state = simulator.reset(charger_count=6, charger_power_kw=22)

    assert state["station_summary"]["total_chargers"] == 6
    assert state["station_summary"]["available_chargers"] == 6
    assert [item["charger_id"] for item in state["chargers"]] == [
        "CHG-01",
        "CHG-02",
        "CHG-03",
        "CHG-04",
        "CHG-05",
        "CHG-06",
    ]


def test_add_ev_to_available_charger_starts_charging() -> None:
    simulator = ChargingStationSimulator()

    state = add_ev(simulator, "EV-1")

    assert session(state, "EV-1")["status"] == "CHARGING"
    assert charger(state, "CHG-01")["current_ev_id"] == "EV-1"


def test_second_ev_uses_second_charger() -> None:
    simulator = ChargingStationSimulator()
    add_ev(simulator, "EV-1")

    state = add_ev(simulator, "EV-2")

    assert charger(state, "CHG-02")["current_ev_id"] == "EV-2"


def test_ev_queues_when_all_chargers_are_busy() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    add_ev(simulator, "EV-1")

    state = add_ev(simulator, "EV-2")

    assert state["queue"] == ["EV-2"]
    assert session(state, "EV-2")["status"] == "WAITING"


def test_advancing_time_increases_soc() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    state = add_ev(simulator, "EV-1")
    initial_soc = session(state, "EV-1")["current_soc_percent"]

    advanced = simulator.advance_time(15)

    assert session(advanced, "EV-1")["current_soc_percent"] > initial_soc


def test_advancing_time_increases_delivered_energy() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    add_ev(simulator, "EV-1")

    state = simulator.advance_time(15)

    assert charger(state, "CHG-01")["energy_delivered_kwh"] == 2.75
    assert session(state, "EV-1")["energy_delivered_kwh"] == 2.75


def test_session_completes_at_target_soc() -> None:
    simulator = ChargingStationSimulator(charger_count=1, charger_power_kw=10)
    short_session(simulator, "EV-1")

    state = simulator.advance_time(6)

    completed = session(state, "EV-1")
    assert completed["status"] == "COMPLETED"
    assert completed["current_soc_percent"] == 10.0
    assert completed["session_end_time"] == 6.0


def test_completed_session_releases_charger() -> None:
    simulator = ChargingStationSimulator(charger_count=1, charger_power_kw=10)
    short_session(simulator, "EV-1")

    state = simulator.advance_time(6)

    assert charger(state, "CHG-01")["status"] == "AVAILABLE"
    assert charger(state, "CHG-01")["current_ev_id"] is None


def test_queued_ev_auto_assigns_after_charger_frees() -> None:
    simulator = ChargingStationSimulator(charger_count=1, charger_power_kw=10)
    short_session(simulator, "EV-1")
    short_session(simulator, "EV-2")

    state = simulator.advance_time(6)

    assert state["queue"] == []
    assert charger(state, "CHG-01")["current_ev_id"] == "EV-2"
    assert session(state, "EV-2")["status"] == "CHARGING"


def test_fault_stops_charging_and_preserves_session_energy() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    add_ev(simulator, "EV-1")
    state_before_fault = simulator.advance_time(5)
    delivered = session(state_before_fault, "EV-1")["energy_delivered_kwh"]

    faulted = simulator.inject_fault("CHG-01")
    advanced = simulator.advance_time(5)

    assert charger(faulted, "CHG-01")["status"] == "FAULT"
    assert charger(faulted, "CHG-01")["anomaly_active"] is True
    assert session(advanced, "EV-1")["energy_delivered_kwh"] == delivered
    assert session(advanced, "EV-1")["status"] == "WAITING"


def test_faulted_charger_is_not_assignable() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    add_ev(simulator, "EV-1")
    simulator.inject_fault("CHG-01")

    state = add_ev(simulator, "EV-2")

    assert state["queue"] == ["EV-1", "EV-2"]
    assert state["station_summary"]["charging_chargers"] == 0


def test_clear_fault_restores_charger_and_resumes_queue() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    add_ev(simulator, "EV-1")
    simulator.inject_fault("CHG-01")

    state = simulator.clear_fault("CHG-01")

    assert charger(state, "CHG-01")["status"] == "CHARGING"
    assert charger(state, "CHG-01")["anomaly_active"] is False
    assert charger(state, "CHG-01")["current_ev_id"] == "EV-1"


def test_offline_charger_is_not_assignable() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    simulator.take_offline("CHG-01")

    state = add_ev(simulator, "EV-1")

    assert charger(state, "CHG-01")["status"] == "OFFLINE"
    assert state["queue"] == ["EV-1"]


def test_restore_returns_charger_to_service() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    simulator.take_offline("CHG-01")

    state = simulator.restore_charger("CHG-01")

    assert charger(state, "CHG-01")["status"] == "AVAILABLE"


@pytest.mark.parametrize(
    ("initial_soc", "target_soc"),
    [(-1, 80), (80, 80), (90, 80), (20, 101)],
)
def test_invalid_soc_is_rejected(initial_soc: float, target_soc: float) -> None:
    simulator = ChargingStationSimulator()

    with pytest.raises(SimulatorValidationError):
        add_ev(
            simulator,
            "EV-1",
            initial_soc=initial_soc,
            target_soc=target_soc,
        )


def test_negative_charge_power_is_rejected() -> None:
    simulator = ChargingStationSimulator()

    with pytest.raises(SimulatorValidationError):
        add_ev(simulator, "EV-1", max_power=-1)


@pytest.mark.parametrize("minutes", [0, -1, float("nan"), float("inf")])
def test_invalid_advance_time_is_rejected(minutes: float) -> None:
    simulator = ChargingStationSimulator()

    with pytest.raises(SimulatorValidationError):
        simulator.advance_time(minutes)


def test_non_finite_configuration_is_rejected() -> None:
    with pytest.raises(SimulatorValidationError):
        ChargingStationSimulator(current_solar_kw=float("inf"))


def test_queue_is_fifo() -> None:
    simulator = ChargingStationSimulator(charger_count=1, charger_power_kw=10)
    short_session(simulator, "EV-1")
    short_session(simulator, "EV-2")
    short_session(simulator, "EV-3")

    first_completion = simulator.advance_time(6)
    second_completion = simulator.advance_time(6)

    assert charger(first_completion, "CHG-01")["current_ev_id"] == "EV-2"
    assert charger(second_completion, "CHG-01")["current_ev_id"] == "EV-3"


def test_same_inputs_and_steps_produce_same_state() -> None:
    first = ChargingStationSimulator(charger_count=2, charger_power_kw=22)
    second = ChargingStationSimulator(charger_count=2, charger_power_kw=22)
    for simulator in (first, second):
        add_ev(simulator, "EV-1")
        add_ev(simulator, "EV-2", initial_soc=40, target_soc=90)
        simulator.advance_time(15)

    assert first.get_state() == second.get_state()


def test_state_is_labeled_as_simulation() -> None:
    state = ChargingStationSimulator().get_state()

    assert state["is_simulation"] is True
    assert state["simulation_version"] == SIMULATOR_VERSION


def test_manual_context_is_preserved_without_generating_forecasts() -> None:
    simulator = ChargingStationSimulator(
        current_solar_kw=35,
        current_station_demand_kw=48,
    )

    summary = simulator.get_state()["station_summary"]

    assert summary["current_solar_kw"] == 35.0
    assert summary["current_station_demand_kw"] == 48.0


def test_impossible_charger_transition_is_rejected() -> None:
    simulator = ChargingStationSimulator(charger_count=1)
    add_ev(simulator, "EV-1")

    with pytest.raises(SimulatorConflictError):
        simulator.take_offline("CHG-01")


def test_transition_rules_include_only_documented_states() -> None:
    assert set(ChargerStatus) == {
        ChargerStatus.AVAILABLE,
        ChargerStatus.RESERVED,
        ChargerStatus.CHARGING,
        ChargerStatus.FAULT,
        ChargerStatus.OFFLINE,
    }
