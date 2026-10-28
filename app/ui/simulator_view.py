"""Streamlit presentation for the deterministic charging station simulator."""

from __future__ import annotations

from html import escape
from textwrap import dedent
from typing import Any

import streamlit as st

from app.ui.api_client import APIResult, ChargeSentinelAPIClient, ResultStatus


def _number(value: object, *, digits: int = 1) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return "—"


def _render_issue(result: APIResult) -> None:
    if result.status is ResultStatus.INVALID_INPUT:
        detail = result.message or "Review the submitted simulation values."
        st.error(f"INVALID ACTION — {detail}")
    elif result.status is ResultStatus.API_OFFLINE:
        st.error("API OFFLINE — Start FastAPI and try the simulation action again.")
    else:
        st.error(
            "BACKEND ERROR — The simulator request could not be completed. "
            "Try again or check the API logs."
        )


def _complete_action(result: APIResult, message: str) -> None:
    if result.status is ResultStatus.READY and result.data is not None:
        st.session_state["simulator_flash"] = message
        st.rerun()
    _render_issue(result)


def _charger_card(charger: dict[str, Any]) -> str:
    charger_id = escape(str(charger.get("charger_id", "—")))
    status = str(charger.get("status", "OFFLINE"))
    safe_status = escape(status)
    status_class = status.lower() if status in {
        "AVAILABLE",
        "RESERVED",
        "CHARGING",
        "FAULT",
        "OFFLINE",
    } else "offline"
    current_ev = escape(str(charger.get("current_ev_id") or "—"))
    soc = charger.get("soc_percent")
    soc_text = "—" if soc is None else f"{_number(soc)}%"
    power = f"{_number(charger.get('current_power_kw'))} kW"
    energy = f"{_number(charger.get('energy_delivered_kwh'), digits=2)} kWh"
    estimate = charger.get("estimated_minutes_remaining")
    if isinstance(estimate, (int, float)) and not isinstance(estimate, bool):
        free_time = f"{estimate:.1f} min"
    elif status == "AVAILABLE":
        free_time = "Available now"
    else:
        free_time = "Unavailable"
    anomaly_text = (
        "SIMULATED FAULT"
        if charger.get("anomaly_active") is True
        else "No simulated fault"
    )
    anomaly_class = "cs-fault-active" if charger.get("anomaly_active") else ""

    return dedent(
        f"""
        <article class="cs-charger-card cs-charger-{status_class}"
            aria-label="{charger_id}, {safe_status}">
            <div class="cs-charger-card-header">
                <strong>{charger_id}</strong>
                <span class="cs-charger-state">{safe_status}</span>
            </div>
            <dl>
                <div><dt>Current EV</dt><dd>{current_ev}</dd></div>
                <div><dt>SOC</dt><dd>{soc_text}</dd></div>
                <div><dt>Power</dt><dd>{power}</dd></div>
                <div><dt>Energy delivered</dt><dd>{energy}</dd></div>
                <div><dt>Estimated free</dt><dd>{free_time}</dd></div>
            </dl>
            <p class="cs-fault-label {anomaly_class}">{anomaly_text}</p>
        </article>
        """
    ).strip()


def _render_state(data: dict[str, object]) -> None:
    raw_summary = data.get("station_summary")
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    raw_completed = data.get("completed_sessions")
    completed = raw_completed if isinstance(raw_completed, list) else []

    top_row = st.columns(3)
    top_row[0].metric("Available Chargers", summary.get("available_chargers", "—"))
    top_row[1].metric("Charging Chargers", summary.get("charging_chargers", "—"))
    top_row[2].metric("Faulted Chargers", summary.get("faulted_chargers", "—"))
    bottom_row = st.columns(3)
    bottom_row[0].metric("Offline Chargers", summary.get("offline_chargers", "—"))
    bottom_row[1].metric("Queue Length", summary.get("queue_length", "—"))
    bottom_row[2].metric(
        "Station Power",
        f"{_number(summary.get('current_station_power_kw'))} kW",
    )

    simulation_time = _number(data.get("simulation_time"))
    st.caption(
        f"Simulation time: {simulation_time} min · Completed sessions: "
        f"{len(completed)} · Manual context: "
        f"{_number(summary.get('current_solar_kw'))} kW solar / "
        f"{_number(summary.get('current_station_demand_kw'))} kW station demand"
    )

    st.markdown("#### Charger Grid")
    raw_chargers = data.get("chargers")
    chargers = raw_chargers if isinstance(raw_chargers, list) else []
    charger_cards = "".join(
        _charger_card(charger)
        for charger in chargers
        if isinstance(charger, dict)
    )
    st.markdown(
        f'<div class="cs-charger-grid">{charger_cards}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Waiting Queue")
    raw_queue = data.get("queue")
    queue = raw_queue if isinstance(raw_queue, list) else []
    if queue:
        items = "".join(f"<li>{escape(str(ev_id))}</li>" for ev_id in queue)
        st.markdown(
            f'<ol class="cs-queue-list">{items}</ol>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Queue empty — all waiting vehicles have been assigned.")


def _render_advance_controls(client: ChargeSentinelAPIClient) -> None:
    st.markdown("#### Advance Simulation Time")
    st.caption("Time changes only when you select an advance action.")
    columns = st.columns(3)
    for column, minutes in zip(columns, (1, 5, 15), strict=True):
        if column.button(
            f"Advance {minutes} min",
            key=f"advance_simulator_{minutes}",
            use_container_width=True,
        ):
            result = client.advance_simulator(minutes)
            _complete_action(result, f"Simulation advanced by {minutes} minute(s).")


def _render_add_ev(client: ChargeSentinelAPIClient) -> None:
    with st.expander("Add EV", expanded=True):
        st.caption(
            "Editable simulation defaults are provided for demo convenience; no "
            "vehicle values are generated automatically."
        )
        with st.form("simulator_add_ev_form"):
            ev_id = st.text_input("EV ID", value="EV-001")
            soc_columns = st.columns(2)
            initial_soc = soc_columns[0].number_input(
                "Initial SOC (%)",
                min_value=0.0,
                max_value=99.9,
                value=20.0,
                step=1.0,
            )
            target_soc = soc_columns[1].number_input(
                "Target SOC (%)",
                min_value=0.1,
                max_value=100.0,
                value=80.0,
                step=1.0,
            )
            energy_columns = st.columns(2)
            battery_capacity = energy_columns[0].number_input(
                "Battery Capacity (kWh)",
                min_value=0.1,
                value=60.0,
                step=1.0,
            )
            max_charge_power = energy_columns[1].number_input(
                "Max Charge Power (kW)",
                min_value=0.1,
                value=11.0,
                step=1.0,
            )
            submitted = st.form_submit_button("Add EV")
        if submitted:
            if target_soc <= initial_soc:
                st.error(
                    "INVALID INPUT — Target SOC must be greater than initial SOC."
                )
            else:
                result = client.add_simulated_ev(
                    {
                        "ev_id": ev_id.strip(),
                        "initial_soc_percent": initial_soc,
                        "target_soc_percent": target_soc,
                        "battery_capacity_kwh": battery_capacity,
                        "max_charge_power_kw": max_charge_power,
                    }
                )
                _complete_action(result, f"{ev_id.strip()} added to the simulation.")


def _charger_status(data: dict[str, object], charger_id: str) -> str:
    raw_chargers = data.get("chargers")
    if not isinstance(raw_chargers, list):
        return ""
    for charger in raw_chargers:
        if isinstance(charger, dict) and charger.get("charger_id") == charger_id:
            return str(charger.get("status", ""))
    return ""


def _render_charger_controls(
    client: ChargeSentinelAPIClient,
    data: dict[str, object],
) -> None:
    raw_chargers = data.get("chargers")
    chargers = raw_chargers if isinstance(raw_chargers, list) else []
    charger_ids = [
        str(charger["charger_id"])
        for charger in chargers
        if isinstance(charger, dict) and "charger_id" in charger
    ]
    with st.expander("Charger Controls"):
        if not charger_ids:
            st.warning("No simulated chargers are available to control.")
            return
        charger_id = st.selectbox("Charger", charger_ids)
        status = _charger_status(data, charger_id)
        st.caption(
            "Fault injection applies to a charging unit. Offline control applies "
            "to an available unit. Current state: "
            f"{status or 'UNKNOWN'}."
        )
        fault_columns = st.columns(2)
        if fault_columns[0].button(
            "Inject Fault",
            disabled=status != "CHARGING",
            use_container_width=True,
        ):
            result = client.inject_simulated_fault(charger_id)
            _complete_action(result, f"SIMULATED FAULT injected on {charger_id}.")
        if fault_columns[1].button(
            "Clear Fault",
            disabled=status != "FAULT",
            use_container_width=True,
        ):
            result = client.clear_simulated_fault(charger_id)
            _complete_action(result, f"Simulated fault cleared on {charger_id}.")

        availability_columns = st.columns(2)
        if availability_columns[0].button(
            "Take Charger Offline",
            disabled=status != "AVAILABLE",
            use_container_width=True,
        ):
            result = client.take_simulated_charger_offline(charger_id)
            _complete_action(result, f"{charger_id} taken offline in the simulation.")
        if availability_columns[1].button(
            "Restore Charger",
            disabled=status != "OFFLINE",
            use_container_width=True,
        ):
            result = client.restore_simulated_charger(charger_id)
            _complete_action(result, f"{charger_id} restored in the simulation.")


def _render_reset_controls(client: ChargeSentinelAPIClient) -> None:
    with st.expander("Reset and Configure Simulation"):
        st.warning(
            "Resetting clears all simulated vehicles, queue entries, and completed "
            "sessions."
        )
        with st.form("simulator_reset_form"):
            config_columns = st.columns(2)
            charger_count = config_columns[0].number_input(
                "Charger Count", min_value=1, max_value=100, value=6, step=1
            )
            charger_power = config_columns[1].number_input(
                "Charger Max Power (kW)",
                min_value=0.1,
                max_value=1000.0,
                value=22.0,
                step=1.0,
            )
            context_columns = st.columns(2)
            current_solar = context_columns[0].number_input(
                "Manual Solar Context (kW)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
            current_demand = context_columns[1].number_input(
                "Manual Station Demand Context (kW)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
            confirmed = st.checkbox(
                "Confirm reset of all simulated station state"
            )
            submitted = st.form_submit_button("Reset Simulation")
        if submitted:
            if not confirmed:
                st.warning("Select the confirmation checkbox before resetting.")
            else:
                result = client.reset_simulator(
                    {
                        "charger_count": charger_count,
                        "charger_power_kw": charger_power,
                        "current_solar_kw": current_solar,
                        "current_station_demand_kw": current_demand,
                    }
                )
                _complete_action(result, "Simulation reset with the selected setup.")


def render_station_simulator(client: ChargeSentinelAPIClient) -> None:
    """Render the single-demo-instance charging station simulator."""

    with st.container(border=True):
        st.subheader("Charging Station Live Simulator")
        st.markdown(
            '<span class="cs-simulation-badge">SIMULATION ONLY</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Add a vehicle and advance simulation time to observe charger occupancy, "
            "queueing, SOC progression, and session completion. This demo does not "
            "control real charging hardware."
        )
        st.caption(
            "State is ephemeral and shared by this single demo instance; it resets "
            "when the API container restarts."
        )

        state_result = client.get_simulator_state()
        if state_result.status is not ResultStatus.READY or state_result.data is None:
            _render_issue(state_result)
            return

        flash = st.session_state.pop("simulator_flash", None)
        if isinstance(flash, str):
            st.success(flash)

        _render_state(state_result.data)
        _render_advance_controls(client)
        _render_add_ev(client)
        _render_charger_controls(client, state_result.data)
        _render_reset_controls(client)
