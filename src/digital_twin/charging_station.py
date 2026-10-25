"""Deterministic, in-memory EV charging station simulation engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from threading import RLock

DEFAULT_CHARGER_COUNT = 6
DEFAULT_CHARGER_POWER_KW = 22.0
SIMULATOR_VERSION = "1.0.0"
OUTPUT_PRECISION = 4
_EPSILON = 1e-9


class ChargerStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    RESERVED = "RESERVED"
    CHARGING = "CHARGING"
    FAULT = "FAULT"
    OFFLINE = "OFFLINE"


class SessionStatus(str, Enum):
    WAITING = "WAITING"
    CHARGING = "CHARGING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


ALLOWED_CHARGER_TRANSITIONS: dict[ChargerStatus, frozenset[ChargerStatus]] = {
    ChargerStatus.AVAILABLE: frozenset(
        {
            ChargerStatus.RESERVED,
            ChargerStatus.CHARGING,
            ChargerStatus.OFFLINE,
        }
    ),
    ChargerStatus.RESERVED: frozenset({ChargerStatus.CHARGING}),
    ChargerStatus.CHARGING: frozenset(
        {ChargerStatus.AVAILABLE, ChargerStatus.FAULT}
    ),
    ChargerStatus.FAULT: frozenset({ChargerStatus.AVAILABLE}),
    ChargerStatus.OFFLINE: frozenset({ChargerStatus.AVAILABLE}),
}


class SimulatorError(ValueError):
    """Base class for safe, expected simulator errors."""


class SimulatorValidationError(SimulatorError):
    """Raised when an engine input violates a simulator invariant."""


class SimulatorNotFoundError(SimulatorError):
    """Raised when a requested simulated resource does not exist."""


class SimulatorConflictError(SimulatorError):
    """Raised when a valid request conflicts with the current simulator state."""


@dataclass
class EVSession:
    ev_id: str
    arrival_time: float
    initial_soc_percent: float
    target_soc_percent: float
    battery_capacity_kwh: float
    max_charge_power_kw: float
    assigned_charger_id: str | None = None
    session_start_time: float | None = None
    session_end_time: float | None = None
    status: SessionStatus = SessionStatus.WAITING
    energy_delivered_kwh: float = 0.0
    charging_elapsed_minutes: float = 0.0

    @property
    def energy_needed_kwh(self) -> float:
        return self.battery_capacity_kwh * (
            self.target_soc_percent - self.initial_soc_percent
        ) / 100

    @property
    def energy_remaining_kwh(self) -> float:
        return max(self.energy_needed_kwh - self.energy_delivered_kwh, 0.0)

    @property
    def current_soc_percent(self) -> float:
        gained_percent = (
            self.energy_delivered_kwh / self.battery_capacity_kwh * 100
        )
        return min(self.initial_soc_percent + gained_percent, self.target_soc_percent)


@dataclass
class Charger:
    charger_id: str
    max_power_kw: float
    status: ChargerStatus = ChargerStatus.AVAILABLE
    current_ev_id: str | None = None
    current_power_kw: float = 0.0
    energy_delivered_kwh: float = 0.0
    session_elapsed_minutes: float = 0.0
    estimated_minutes_remaining: float | None = None
    anomaly_active: bool = False


class ChargingStationSimulator:
    """Simulate one configurable charging hub without wall-clock time or randomness."""

    def __init__(
        self,
        *,
        charger_count: int = DEFAULT_CHARGER_COUNT,
        charger_power_kw: float = DEFAULT_CHARGER_POWER_KW,
        current_solar_kw: float = 0.0,
        current_station_demand_kw: float = 0.0,
    ) -> None:
        self._lock = RLock()
        self.reset(
            charger_count=charger_count,
            charger_power_kw=charger_power_kw,
            current_solar_kw=current_solar_kw,
            current_station_demand_kw=current_station_demand_kw,
        )

    def reset(
        self,
        *,
        charger_count: int = DEFAULT_CHARGER_COUNT,
        charger_power_kw: float = DEFAULT_CHARGER_POWER_KW,
        current_solar_kw: float = 0.0,
        current_station_demand_kw: float = 0.0,
    ) -> dict[str, object]:
        """Replace all state with one explicit simulation configuration."""

        self._validate_configuration(
            charger_count,
            charger_power_kw,
            current_solar_kw,
            current_station_demand_kw,
        )
        with self._lock:
            self.simulation_time = 0.0
            self.current_solar_kw = float(current_solar_kw)
            self.current_station_demand_kw = float(current_station_demand_kw)
            self.chargers = [
                Charger(
                    charger_id=f"CHG-{index:02d}",
                    max_power_kw=float(charger_power_kw),
                )
                for index in range(1, charger_count + 1)
            ]
            self.sessions: dict[str, EVSession] = {}
            self.waiting_ev_ids: list[str] = []
            self.completed_ev_ids: list[str] = []
            return self._state_unlocked()

    def get_state(self) -> dict[str, object]:
        """Return a validated-shape snapshot without mutating simulation state."""

        with self._lock:
            return self._state_unlocked()

    def add_ev(
        self,
        *,
        ev_id: str,
        initial_soc_percent: float,
        target_soc_percent: float,
        battery_capacity_kwh: float,
        max_charge_power_kw: float,
    ) -> dict[str, object]:
        """Add one EV at the current simulation time and assign it if possible."""

        normalized_ev_id = ev_id.strip()
        self._validate_ev(
            normalized_ev_id,
            initial_soc_percent,
            target_soc_percent,
            battery_capacity_kwh,
            max_charge_power_kw,
        )
        with self._lock:
            if normalized_ev_id in self.sessions:
                raise SimulatorConflictError(
                    f"EV ID '{normalized_ev_id}' already exists in this simulation."
                )
            session = EVSession(
                ev_id=normalized_ev_id,
                arrival_time=self.simulation_time,
                initial_soc_percent=float(initial_soc_percent),
                target_soc_percent=float(target_soc_percent),
                battery_capacity_kwh=float(battery_capacity_kwh),
                max_charge_power_kw=float(max_charge_power_kw),
            )
            self.sessions[normalized_ev_id] = session
            self.waiting_ev_ids.append(normalized_ev_id)
            self._assign_waiting_unlocked()
            return self._state_unlocked()

    def advance_time(self, minutes: float) -> dict[str, object]:
        """Advance only the deterministic simulation clock and charging sessions."""

        if not isinstance(minutes, (int, float)) or isinstance(minutes, bool):
            raise SimulatorValidationError("Advance minutes must be a number.")
        if not isfinite(minutes) or minutes <= 0:
            raise SimulatorValidationError("Advance minutes must be greater than zero.")

        with self._lock:
            remaining_minutes = float(minutes)
            while remaining_minutes > _EPSILON:
                self._assign_waiting_unlocked()
                charging = self._charging_chargers_unlocked()
                if not charging:
                    self.simulation_time += remaining_minutes
                    remaining_minutes = 0.0
                    break

                next_completion = min(
                    self._minutes_to_completion_unlocked(charger)
                    for charger in charging
                )
                step = min(remaining_minutes, next_completion)

                if step <= _EPSILON:
                    self._complete_finished_sessions_unlocked()
                    continue

                for charger in charging:
                    self._advance_charger_unlocked(charger, step)
                self.simulation_time += step
                remaining_minutes -= step
                self._complete_finished_sessions_unlocked()

            self._assign_waiting_unlocked()
            return self._state_unlocked()

    def inject_fault(self, charger_id: str) -> dict[str, object]:
        """Inject a simulated fault and return an interrupted EV to the FIFO queue."""

        with self._lock:
            charger = self._charger_unlocked(charger_id)
            self._transition_unlocked(charger, ChargerStatus.FAULT)
            interrupted_ev_id = charger.current_ev_id
            charger.anomaly_active = True
            if interrupted_ev_id is not None:
                session = self.sessions[interrupted_ev_id]
                session.status = SessionStatus.WAITING
                session.assigned_charger_id = None
                self.waiting_ev_ids.insert(0, interrupted_ev_id)
            self._clear_active_charger_fields(charger)
            self._assign_waiting_unlocked()
            return self._state_unlocked()

    def clear_fault(self, charger_id: str) -> dict[str, object]:
        """Clear an explicit simulated fault and restore charger availability."""

        with self._lock:
            charger = self._charger_unlocked(charger_id)
            self._transition_unlocked(charger, ChargerStatus.AVAILABLE)
            charger.anomaly_active = False
            self._assign_waiting_unlocked()
            return self._state_unlocked()

    def take_offline(self, charger_id: str) -> dict[str, object]:
        """Take an available simulated charger offline."""

        with self._lock:
            charger = self._charger_unlocked(charger_id)
            self._transition_unlocked(charger, ChargerStatus.OFFLINE)
            return self._state_unlocked()

    def restore_charger(self, charger_id: str) -> dict[str, object]:
        """Restore an offline simulated charger and serve the FIFO queue."""

        with self._lock:
            charger = self._charger_unlocked(charger_id)
            self._transition_unlocked(charger, ChargerStatus.AVAILABLE)
            self._assign_waiting_unlocked()
            return self._state_unlocked()

    @staticmethod
    def _validate_configuration(
        charger_count: int,
        charger_power_kw: float,
        current_solar_kw: float,
        current_station_demand_kw: float,
    ) -> None:
        if not isinstance(charger_count, int) or isinstance(charger_count, bool):
            raise SimulatorValidationError("Charger count must be an integer.")
        if charger_count <= 0:
            raise SimulatorValidationError("Charger count must be greater than zero.")
        for label, value in (
            ("Charger power", charger_power_kw),
            ("Current solar", current_solar_kw),
            ("Current station demand", current_station_demand_kw),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SimulatorValidationError(f"{label} must be a number.")
            if not isfinite(value):
                raise SimulatorValidationError(f"{label} must be finite.")
        if charger_power_kw <= 0:
            raise SimulatorValidationError("Charger power must be greater than zero.")
        if current_solar_kw < 0 or current_station_demand_kw < 0:
            raise SimulatorValidationError(
                "Solar and station demand context values cannot be negative."
            )

    @staticmethod
    def _validate_ev(
        ev_id: str,
        initial_soc_percent: float,
        target_soc_percent: float,
        battery_capacity_kwh: float,
        max_charge_power_kw: float,
    ) -> None:
        if not ev_id:
            raise SimulatorValidationError("EV ID is required.")
        values = (
            initial_soc_percent,
            target_soc_percent,
            battery_capacity_kwh,
            max_charge_power_kw,
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in values
        ):
            raise SimulatorValidationError("EV energy inputs must be numeric.")
        if not all(isfinite(value) for value in values):
            raise SimulatorValidationError("EV energy inputs must be finite.")
        if not 0 <= initial_soc_percent < target_soc_percent <= 100:
            raise SimulatorValidationError(
                "SOC must satisfy 0 ≤ initial SOC < target SOC ≤ 100."
            )
        if battery_capacity_kwh <= 0:
            raise SimulatorValidationError(
                "Battery capacity must be greater than zero."
            )
        if max_charge_power_kw <= 0:
            raise SimulatorValidationError(
                "Maximum charge power must be greater than zero."
            )

    def _charger_unlocked(self, charger_id: str) -> Charger:
        normalized = charger_id.strip()
        for charger in self.chargers:
            if charger.charger_id == normalized:
                return charger
        raise SimulatorNotFoundError(f"Charger '{normalized}' was not found.")

    @staticmethod
    def _transition_unlocked(charger: Charger, target: ChargerStatus) -> None:
        allowed = ALLOWED_CHARGER_TRANSITIONS[charger.status]
        if target not in allowed:
            raise SimulatorConflictError(
                f"Cannot transition {charger.charger_id} from "
                f"{charger.status.value} to {target.value}."
            )
        charger.status = target

    def _assign_waiting_unlocked(self) -> None:
        while self.waiting_ev_ids:
            charger = next(
                (
                    item
                    for item in self.chargers
                    if item.status is ChargerStatus.AVAILABLE
                ),
                None,
            )
            if charger is None:
                return
            ev_id = self.waiting_ev_ids.pop(0)
            session = self.sessions[ev_id]
            self._transition_unlocked(charger, ChargerStatus.CHARGING)
            session.status = SessionStatus.CHARGING
            session.assigned_charger_id = charger.charger_id
            if session.session_start_time is None:
                session.session_start_time = self.simulation_time
            charger.current_ev_id = ev_id
            charger.current_power_kw = min(
                session.max_charge_power_kw,
                charger.max_power_kw,
            )
            charger.energy_delivered_kwh = session.energy_delivered_kwh
            charger.session_elapsed_minutes = session.charging_elapsed_minutes
            self._refresh_charger_estimate_unlocked(charger)

    def _charging_chargers_unlocked(self) -> list[Charger]:
        return [
            charger
            for charger in self.chargers
            if charger.status is ChargerStatus.CHARGING
        ]

    def _minutes_to_completion_unlocked(self, charger: Charger) -> float:
        if charger.current_ev_id is None or charger.current_power_kw <= 0:
            raise SimulatorConflictError(
                f"Charging state is incomplete for {charger.charger_id}."
            )
        session = self.sessions[charger.current_ev_id]
        return session.energy_remaining_kwh / charger.current_power_kw * 60

    def _advance_charger_unlocked(self, charger: Charger, minutes: float) -> None:
        if charger.current_ev_id is None:
            raise SimulatorConflictError(
                f"Charging state is incomplete for {charger.charger_id}."
            )
        session = self.sessions[charger.current_ev_id]
        delivered = min(
            charger.current_power_kw * minutes / 60,
            session.energy_remaining_kwh,
        )
        session.energy_delivered_kwh += delivered
        session.charging_elapsed_minutes += minutes
        charger.energy_delivered_kwh = session.energy_delivered_kwh
        charger.session_elapsed_minutes = session.charging_elapsed_minutes
        self._refresh_charger_estimate_unlocked(charger)

    def _complete_finished_sessions_unlocked(self) -> None:
        for charger in self._charging_chargers_unlocked():
            if charger.current_ev_id is None:
                continue
            session = self.sessions[charger.current_ev_id]
            if session.energy_remaining_kwh > _EPSILON:
                continue
            session.energy_delivered_kwh = session.energy_needed_kwh
            session.status = SessionStatus.COMPLETED
            session.session_end_time = self.simulation_time
            session.assigned_charger_id = charger.charger_id
            self.completed_ev_ids.append(session.ev_id)
            self._transition_unlocked(charger, ChargerStatus.AVAILABLE)
            self._clear_active_charger_fields(charger)
        self._assign_waiting_unlocked()

    def _refresh_charger_estimate_unlocked(self, charger: Charger) -> None:
        if charger.current_ev_id is None or charger.current_power_kw <= 0:
            charger.estimated_minutes_remaining = None
            return
        session = self.sessions[charger.current_ev_id]
        charger.estimated_minutes_remaining = (
            session.energy_remaining_kwh / charger.current_power_kw * 60
        )

    @staticmethod
    def _clear_active_charger_fields(charger: Charger) -> None:
        charger.current_ev_id = None
        charger.current_power_kw = 0.0
        charger.energy_delivered_kwh = 0.0
        charger.session_elapsed_minutes = 0.0
        charger.estimated_minutes_remaining = None

    def _session_state_unlocked(self, session: EVSession) -> dict[str, object]:
        return {
            "ev_id": session.ev_id,
            "arrival_time": self._rounded(session.arrival_time),
            "initial_soc_percent": self._rounded(session.initial_soc_percent),
            "target_soc_percent": self._rounded(session.target_soc_percent),
            "current_soc_percent": self._rounded(session.current_soc_percent),
            "battery_capacity_kwh": self._rounded(session.battery_capacity_kwh),
            "max_charge_power_kw": self._rounded(session.max_charge_power_kw),
            "assigned_charger_id": session.assigned_charger_id,
            "session_start_time": self._optional_rounded(
                session.session_start_time
            ),
            "session_end_time": self._optional_rounded(session.session_end_time),
            "status": session.status.value,
            "energy_delivered_kwh": self._rounded(
                session.energy_delivered_kwh
            ),
        }

    def _charger_state_unlocked(self, charger: Charger) -> dict[str, object]:
        session = (
            self.sessions[charger.current_ev_id]
            if charger.current_ev_id is not None
            else None
        )
        return {
            "charger_id": charger.charger_id,
            "status": charger.status.value,
            "max_power_kw": self._rounded(charger.max_power_kw),
            "current_ev_id": charger.current_ev_id,
            "current_power_kw": self._rounded(charger.current_power_kw),
            "energy_delivered_kwh": self._rounded(
                charger.energy_delivered_kwh
            ),
            "session_elapsed_minutes": self._rounded(
                charger.session_elapsed_minutes
            ),
            "soc_percent": (
                None
                if session is None
                else self._rounded(session.current_soc_percent)
            ),
            "estimated_minutes_remaining": self._optional_rounded(
                charger.estimated_minutes_remaining
            ),
            "anomaly_active": charger.anomaly_active,
        }

    def _state_unlocked(self) -> dict[str, object]:
        status_counts = {
            status: sum(
                charger.status is status for charger in self.chargers
            )
            for status in ChargerStatus
        }
        sessions = [
            self._session_state_unlocked(session)
            for session in self.sessions.values()
        ]
        completed = [
            self._session_state_unlocked(self.sessions[ev_id])
            for ev_id in self.completed_ev_ids
        ]
        return {
            "simulation_time": self._rounded(self.simulation_time),
            "chargers": [
                self._charger_state_unlocked(charger)
                for charger in self.chargers
            ],
            "queue": list(self.waiting_ev_ids),
            "sessions": sessions,
            "completed_sessions": completed,
            "station_summary": {
                "total_chargers": len(self.chargers),
                "available_chargers": status_counts[ChargerStatus.AVAILABLE],
                "charging_chargers": status_counts[ChargerStatus.CHARGING],
                "reserved_chargers": status_counts[ChargerStatus.RESERVED],
                "faulted_chargers": status_counts[ChargerStatus.FAULT],
                "offline_chargers": status_counts[ChargerStatus.OFFLINE],
                "queue_length": len(self.waiting_ev_ids),
                "current_station_power_kw": self._rounded(
                    sum(charger.current_power_kw for charger in self.chargers)
                ),
                "current_solar_kw": self._rounded(self.current_solar_kw),
                "current_station_demand_kw": self._rounded(
                    self.current_station_demand_kw
                ),
            },
            "simulation_version": SIMULATOR_VERSION,
            "is_simulation": True,
        }

    @staticmethod
    def _rounded(value: float) -> float:
        return round(value, OUTPUT_PRECISION)

    @classmethod
    def _optional_rounded(cls, value: float | None) -> float | None:
        return None if value is None else cls._rounded(value)
