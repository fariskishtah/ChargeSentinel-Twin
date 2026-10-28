"""Operator and EV driver dashboard views."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui.api_client import APIResult, ChargeSentinelAPIClient, ResultStatus
from app.ui.simulator_view import render_station_simulator

NOT_READY_MESSAGES = {
    "solar": "Solar service is temporarily unavailable.",
    "demand": "Demand service is temporarily unavailable.",
    "anomaly": "Anomaly service is temporarily unavailable.",
    "availability": "The portable availability bundle is unavailable.",
    "recommendation": "The availability bundle or historical risk table is unavailable.",
    "digital_twin": "Digital Twin service is temporarily unavailable.",
}


def _iso_utc(selected_date: date, selected_time: time) -> str:
    return datetime.combine(selected_date, selected_time, tzinfo=UTC).isoformat()


def _mariam_demo_input(name: str) -> dict[str, object]:
    path = Path(__file__).resolve().parent / "demo_inputs" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_feature_payload(raw: str) -> dict[str, object] | None:
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    if not all(isinstance(key, str) for key in parsed):
        return None
    return {str(key): value for key, value in parsed.items()}


def _value(data: dict[str, object], key: str) -> object:
    value = data.get(key)
    return value if value is not None else "—"


def _percentage(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "—"


def _render_result(
    result: APIResult,
    component: str,
    render_ready: Callable[[dict[str, object]], None],
) -> None:
    if result.status is ResultStatus.READY and result.data is not None:
        st.success("READY")
        render_ready(result.data)
    elif result.status is ResultStatus.MODEL_NOT_READY:
        message = NOT_READY_MESSAGES.get(
            result.component or component,
            "This component is not integrated yet.",
        )
        st.warning(f"MODEL NOT READY — {message}")
    elif result.status is ResultStatus.INVALID_INPUT:
        st.error("INVALID INPUT — Review the submitted values and try again.")
    elif result.status is ResultStatus.API_OFFLINE:
        st.error(
            "API OFFLINE — Unable to reach the service. Start FastAPI and try again."
        )
    else:
        st.error(
            "BACKEND ERROR — Unable to complete the request. Try again or check "
            "the service logs if the issue continues."
        )


def render_health(result: APIResult) -> None:
    with st.container(border=True):
        st.subheader("System Health")
        st.caption(
            "Live connectivity and version information for the FastAPI service."
        )
        if result.status is ResultStatus.READY and result.data is not None:
            st.success("API Online")
            first, second = st.columns(2)
            first.metric("API Status", "READY")
            second.metric("API Version", _value(result.data, "api_version"))
        elif result.status is ResultStatus.API_OFFLINE:
            st.error("API OFFLINE — Dashboard controls remain available.")
        else:
            st.error("BACKEND ERROR — Health response is unavailable.")


def _render_solar(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Solar Forecast")
        st.caption(
            "Reference/modelled PV generation forecast for a 100 kWp New Cairo system. "
            "2024 H1 historical demonstration; values are not measured station output."
        )
        with st.form("solar_form"):
            dcol, tcol = st.columns(2)
            selected_date = dcol.date_input("Forecast Origin — Date (UTC)", value=date(2024, 6, 15), key="solar_origin_date")
            selected_time = tcol.time_input("Forecast Origin — Hour (UTC)", value=time(0), step=3600, key="solar_origin_time")
            forecast_origin = _iso_utc(selected_date, selected_time)
            horizon_label = st.radio(
                "Forecast Horizon", ["24 Hours", "48 Hours"], horizontal=True
            )
            submitted = st.form_submit_button("Run Solar Forecast")
        if submitted:
            horizon = 24 if horizon_label == "24 Hours" else 48
            result = client.get_solar_forecast(
                {"forecast_origin": forecast_origin, "horizon_hours": horizon}
            )

            def show(data: dict[str, object]) -> None:
                cols = st.columns(3)
                cols[0].metric("Horizon", f"{data['horizon_hours']} h")
                cols[1].metric("Model Family", _value(data, "model_family"))
                cols[2].metric("Model Version", _value(data, "model_version"))
                points = data["points"]
                curve = pd.DataFrame(points).set_index("timestamp").rename(columns={"forecast_kw": "Reference PV (kW)"})
                curve.index = pd.to_datetime(curve.index, utc=True)
                st.line_chart(curve)
                st.caption("Forecast uses only source history and forecast runs available at the selected origin.")

            _render_result(result, "solar", show)


def _render_demand(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("EV Demand Forecast")
        st.caption(
            "Jiaxing historical demo: 2020-01-08 21:00 through 2022-01-01 00:00. "
            "Use source operational time; source timezone is unverified."
        )
        st.info("Energy is aggregated by session completion hour and is used as an EV demand proxy; "
                "it is not direct physical grid-load telemetry.")
        with st.form("demand_form"):
            date_col, time_col = st.columns(2)
            selected_date = date_col.date_input("Forecast Origin — Date (source operational time)",
                                               value=date(2021, 10, 1), key="demand_origin_date")
            selected_time = time_col.time_input("Forecast Origin — Hour (source operational time)",
                                               value=time(0), step=3600, key="demand_origin_time")
            forecast_origin = datetime.combine(selected_date, selected_time).isoformat()
            submitted = st.form_submit_button("Run Demand Forecast")
        if submitted:
            result = client.get_demand_forecast(
                {"forecast_origin": forecast_origin}
            )

            def show(data: dict[str, object]) -> None:
                cols = st.columns(3)
                cols[0].metric(
                    "Expected Completed-Session Energy (kWh)", f"{data['expected_energy_kwh']:.2f}"
                )
                cols[1].metric("Expected Sessions", f"{data['expected_sessions']:.1f}")
                cols[2].metric("Model Version", _value(data, "model_version"))
                st.caption(f"Forecast timestamp (source operational time): {data['timestamp']}")

            _render_result(result, "demand", show)


def _render_anomaly(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("EV Session Risk")
        st.caption(
            "Start-of-session estimate of abnormal end-cause risk on Jiaxing data. "
            "This is an investigation flag, not a confirmed live fault."
        )
        with st.form("anomaly_form"):
            charger_id = st.text_input("Charger ID", value="12", key="anomaly_charger")
            location = st.text_input("Location", value="Financial Industrial Park", key="anomaly_location")
            district = st.text_input("District", value="Nanhu", key="anomaly_district")
            dcol, tcol = st.columns(2)
            start_date = dcol.date_input("Session Start — Date (source operational time)", value=date(2021, 10, 16), key="anomaly_start_date")
            start_time = tcol.time_input("Session Start — Time (source operational time)", value=time(12, 2, 44), step=1, key="anomaly_start_time")
            submitted = st.form_submit_button("Check for Anomalies")
        if submitted:
            result = client.predict_anomaly(
                {"charger_id": charger_id.strip(), "features": {"location": location.strip(),
                 "district": district.strip(), "start_time": datetime.combine(start_date, start_time).isoformat()}}
            )

            def show(data: dict[str, object]) -> None:
                cols = st.columns(4)
                cols[0].metric("Status", "Investigate" if data["anomaly"] else "Lower risk")
                cols[1].metric("Probability", f"{data['score']:.1%}")
                cols[2].metric("Threshold", f"{data['threshold']:.1%}")
                cols[3].metric("Model Version", _value(data, "model_version"))
                st.write("Top contributing features", data["reason"])

            _render_result(result, "anomaly", show)


def _render_grid_load(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Grid-Load Interval Anomaly")
        st.caption("Historical Jiaxing completed-session energy proxy, not measured grid power. "
                   "An unusual interval is an investigation flag, not a confirmed equipment fault.")
        with st.form("grid_load_form"):
            day = st.date_input("Interval — Date (source operational time)",
                                value=date(2021, 10, 1), key="grid_load_date")
            hour = st.time_input("Interval — Hour (source operational time)",
                                 value=time(0), step=3600, key="grid_load_hour")
            submitted = st.form_submit_button("Check Historical Interval")
        if submitted:
            result = client.detect_grid_load({"timestamp": datetime.combine(day, hour).isoformat()})

            def show(data: dict[str, object]) -> None:
                cols = st.columns(4)
                cols[0].metric("Status", "Investigate" if data["anomaly"] else "Within range")
                cols[1].metric("Deviation score", f"{data['score']:.2f}")
                cols[2].metric("Threshold", f"{data['threshold']:.2f}")
                cols[3].metric("Completed energy (kWh)", f"{data['energy_kwh']:.1f}")
                st.write("Reason", data["reason"])
                st.caption(f"Detector: {data['detector_version']} · Recorded arrivals: {data['sessions']}")

            _render_result(result, "anomaly", show)


def _availability_form(client: ChargeSentinelAPIClient, *, form_key: str) -> None:
    request = _mariam_demo_input("availability")
    with st.form(form_key):
        st.caption(
            f"Real Jiaxing held-out session: charger {request['charger_id']} · "
            f"started {request['session_start']} (source operational time). "
            "The model receives start-known history, not the actual End Time."
        )
        with st.expander("View historical request context"):
            st.json(request)
        submitted = st.form_submit_button("Check Availability")
    result = None
    if submitted:
        result = client.get_availability(request)

    st.markdown("#### Result")
    if (
        result is not None
        and result.status is ResultStatus.READY
        and result.data is not None
    ):

        def show(data: dict[str, object]) -> None:
            cols = st.columns(3)
            cols[0].metric("Charger", _value(data, "charger_id"))
            cols[1].metric("Expected free time", _value(data, "expected_free_time"))
            cols[2].metric(
                "Expected wait", f"{_value(data, 'expected_wait_minutes')} min"
            )
            st.caption(
                f"Estimated total session duration: {_value(data, 'predicted_duration_minutes')} min · "
                f"Model: {_value(data, 'model_version')}"
            )
            st.info("Historical estimate only; no live charger telemetry or guaranteed release time.")

        _render_result(result, "availability", show)
    else:
        cols = st.columns(2)
        cols[0].metric("Expected Free Time", "—")
        cols[1].metric("Expected Wait", "—")
        if result is not None:
            _render_result(result, "availability", lambda data: None)


def _render_operator_availability(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Charger Availability")
        st.caption(
            "Check when a specific charging point is expected to become available."
        )
        _availability_form(client, form_key="operator_availability")


def _render_twin(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Digital Twin — What-If Simulator")
        st.markdown(
            '<span class="cs-simulation-badge">SIMULATION ONLY</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Scenario outputs are simulated estimates and are not live grid-control "
            "decisions. Inputs are user-provided, not live station measurements or "
            "model forecasts."
        )
        with st.form("digital_twin_form"):
            st.markdown(
                '<p class="cs-group-label">Baseline State</p>',
                unsafe_allow_html=True,
            )
            solar_col, demand_col = st.columns(2)
            baseline_solar_kw = solar_col.number_input(
                "Baseline Solar (kW)", min_value=0.0, value=0.0, step=1.0
            )
            baseline_demand_kw = demand_col.number_input(
                "Baseline Demand (kW)", min_value=0.0, value=0.0, step=1.0
            )
            total_col, active_col, session_col = st.columns(3)
            total_chargers = total_col.number_input(
                "Total Chargers", min_value=1, value=10, step=1
            )
            active_chargers = active_col.number_input(
                "Active Chargers",
                min_value=0,
                value=total_chargers,
                step=1,
            )
            average_session_minutes = session_col.number_input(
                "Average Session Duration (minutes)",
                min_value=1.0,
                value=60.0,
                step=5.0,
            )
            st.markdown(
                '<p class="cs-group-label">Scenario Modifiers</p>',
                unsafe_allow_html=True,
            )
            demand_col, solar_col = st.columns(2)
            demand_multiplier = demand_col.number_input(
                "Demand Multiplier", min_value=0.1, value=1.0, step=0.1
            )
            solar_multiplier = solar_col.number_input(
                "Solar Multiplier", min_value=0.0, value=1.0, step=0.1
            )
            offline_raw = st.text_input(
                "Offline Charger IDs",
                help="Optional comma-separated charger IDs.",
            )
            submitted = st.form_submit_button("Run Simulation")
        if submitted:
            offline_chargers = [
                item.strip() for item in offline_raw.split(",") if item.strip()
            ]
            result = client.simulate_digital_twin(
                {
                    "baseline_solar_kw": baseline_solar_kw,
                    "baseline_demand_kw": baseline_demand_kw,
                    "total_chargers": total_chargers,
                    "active_chargers": active_chargers,
                    "average_session_minutes": average_session_minutes,
                    "demand_multiplier": demand_multiplier,
                    "solar_multiplier": solar_multiplier,
                    "offline_chargers": offline_chargers,
                }
            )

            def show(data: dict[str, object]) -> None:
                st.info("SIMULATION — Deterministic what-if result")
                first_row = st.columns(3)
                first_row[0].metric(
                    "Scenario solar", f"{_value(data, 'scenario_solar_kw')} kW"
                )
                first_row[1].metric(
                    "Scenario demand", f"{_value(data, 'scenario_demand_kw')} kW"
                )
                first_row[2].metric(
                    "Renewable balance",
                    f"{_value(data, 'supply_demand_balance_kw')} kW",
                )
                second_row = st.columns(3)
                second_row[0].metric(
                    "Renewable coverage",
                    _percentage(data.get("renewable_coverage_ratio")),
                )
                second_row[1].metric(
                    "Utilization",
                    _percentage(data.get("estimated_utilization")),
                )
                second_row[2].metric(
                    "Wait impact",
                    f"{_value(data, 'estimated_wait_impact_minutes')} min",
                )
                severity = str(_value(data, "severity"))
                severity_class = (
                    severity.lower()
                    if severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
                    else "high"
                )
                st.markdown(
                    '<span class="cs-severity-badge '
                    f'cs-severity-{severity_class}">Severity: '
                    f"{escape(severity)}</span>",
                    unsafe_allow_html=True,
                )
                recommendation = escape(str(_value(data, "recommendation")))
                st.markdown(
                    '<div class="cs-recommendation"><strong>Recommendation</strong>'
                    f"<br>{recommendation}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Simulation version: {_value(data, 'simulation_version')} · "
                    "Scenario outputs are simulated estimates and are not live "
                    "grid-control decisions."
                )

            _render_result(result, "digital_twin", show)


def render_operator_view(
    client: ChargeSentinelAPIClient, health_result: APIResult
) -> None:
    st.header("Operator View")
    st.markdown(
        '<p class="cs-view-intro">Monitor service readiness, prepare model requests, '
        "and test deterministic operating scenarios.</p>",
        unsafe_allow_html=True,
    )
    render_health(health_result)
    _render_solar(client)
    _render_demand(client)
    _render_anomaly(client)
    _render_grid_load(client)
    _render_operator_availability(client)
    render_station_simulator(client)
    _render_twin(client)


def _render_driver_availability(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Check Charger Availability")
        st.caption("Historical Jiaxing example of when a charging session may finish.")
        _availability_form(client, form_key="driver_availability")


def _render_recommendation(client: ChargeSentinelAPIClient) -> None:
    request = _mariam_demo_input("recommendation")
    with st.container(border=True):
        st.subheader("Smart Charger Recommendation")
        st.caption(
            "Historical Jiaxing candidate states reconstructed at the shown arrival time. "
            "This is a ranked suggestion, not a live availability guarantee."
        )
        with st.form("recommendation_form"):
            st.caption(f"Location: {request['location']} · Arrival: {request['user_eta']} (source operational time)")
            with st.expander("View historical candidate context"):
                st.json(request)
            submitted = st.form_submit_button("Recommend Charger")
        if submitted:
            result = client.get_recommendation(request)

            def show(data: dict[str, object]) -> None:
                cols = st.columns(3)
                cols[0].metric(
                    "Recommended charger", _value(data, "recommended_charger")
                )
                cols[1].metric("Recommended slot", _value(data, "recommended_slot"))
                cols[2].metric(
                    "Expected wait", f"{_value(data, 'expected_wait_minutes')} min"
                )
                explanation = data.get("explanation") or {}
                if isinstance(explanation, dict):
                    st.write("Why this charger:", explanation.get("reason", "—"))
                st.write("Alternatives:", _value(data, "alternatives"))
                if data.get("mock_signals_used"):
                    st.warning("Demand and solar score inputs are neutral placeholders in this historical replay.")
                st.caption(f"Ranker: {_value(data, 'model_version')} · Availability bundle: {_value(data, 'bundle_version')}")
                st.info("Fault and offline candidates are excluded; historical abnormal-end rates are not hardware fault probabilities.")

            _render_result(result, "recommendation", show)


def render_driver_view(client: ChargeSentinelAPIClient) -> None:
    st.header("EV Driver View")
    st.markdown(
        '<p class="cs-view-intro">Plan a charging stop with clear availability and '
        "charger recommendations using documented historical Jiaxing context.</p>",
        unsafe_allow_html=True,
    )
    _render_driver_availability(client)
    _render_recommendation(client)
