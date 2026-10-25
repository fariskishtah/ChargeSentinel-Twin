"""Operator and EV driver dashboard views."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from html import escape

import streamlit as st

from app.ui.api_client import APIResult, ChargeSentinelAPIClient, ResultStatus

NOT_READY_MESSAGES = {
    "solar": "Solar forecasting model is awaiting integration.",
    "demand": "EV demand forecasting model is awaiting integration.",
    "anomaly": "Anomaly detection model is awaiting integration.",
    "availability": "Charger availability model is awaiting integration.",
    "recommendation": "Smart charger recommendation model is awaiting integration.",
    "digital_twin": "Digital Twin service is temporarily unavailable.",
}


def _iso_utc(selected_date: date, selected_time: time) -> str:
    return datetime.combine(selected_date, selected_time, tzinfo=UTC).isoformat()


def _datetime_fields(label: str, key: str) -> str:
    now = datetime.now(UTC)
    date_col, time_col = st.columns(2)
    selected_date = date_col.date_input(
        f"{label} — Date (UTC)", value=now.date(), key=f"{key}_date"
    )
    selected_time = time_col.time_input(
        f"{label} — Time (UTC)",
        value=now.time().replace(microsecond=0),
        key=f"{key}_time",
    )
    return _iso_utc(selected_date, selected_time)


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
            "Request an hourly renewable generation forecast for the selected horizon."
        )
        with st.form("solar_form"):
            forecast_origin = _datetime_fields("Forecast Origin", "solar_origin")
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
                cols[0].metric("Forecast timestamp", _value(data, "timestamp"))
                cols[1].metric("Horizon", f"{_value(data, 'horizon_hours')} h")
                cols[2].metric("Model version", _value(data, "model_version"))
                st.caption(
                    "Hourly forecast points will be visualized when the integrated "
                    "model supplies the selected horizon."
                )

            _render_result(result, "solar", show)


def _render_demand(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("EV Demand Forecast")
        st.caption(
            "Estimate charging energy and session demand from a UTC origin time."
        )
        with st.form("demand_form"):
            forecast_origin = _datetime_fields("Forecast Origin", "demand_origin")
            submitted = st.form_submit_button("Run Demand Forecast")
        if submitted:
            result = client.get_demand_forecast(
                {"forecast_origin": forecast_origin}
            )

            def show(data: dict[str, object]) -> None:
                cols = st.columns(3)
                cols[0].metric(
                    "Expected energy", f"{_value(data, 'expected_energy_kwh')} kWh"
                )
                cols[1].metric("Expected sessions", _value(data, "expected_sessions"))
                cols[2].metric("Model version", _value(data, "model_version"))

            _render_result(result, "demand", show)


def _render_anomaly(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Operational Charging Anomaly Detection")
        st.caption(
            "Evaluate session-level charging behavior using only leakage-audited "
            "features."
        )
        with st.form("anomaly_form"):
            charger_id = st.text_input("Charger ID", key="anomaly_charger")
            raw_features = st.text_area(
                "Approved Session Features (JSON)",
                value="{}",
                help=(
                    "Use only final feature names confirmed by the anomaly-model owner."
                ),
            )
            submitted = st.form_submit_button("Check for Anomalies")
        if submitted:
            features = _parse_feature_payload(raw_features)
            if not charger_id.strip() or features is None:
                st.error(
                    "INVALID INPUT — Enter a charger ID and a non-empty JSON object "
                    "containing only confirmed safe features."
                )
                return
            result = client.predict_anomaly(
                {"charger_id": charger_id.strip(), "features": features}
            )

            def show(data: dict[str, object]) -> None:
                cols = st.columns(4)
                cols[0].metric("Charger", _value(data, "charger_id"))
                cols[1].metric("Anomaly", _value(data, "anomaly"))
                cols[2].metric("Score", _value(data, "score"))
                cols[3].metric("Model version", _value(data, "model_version"))
                st.write("Reasons", _value(data, "reason"))

            _render_result(result, "anomaly", show)


def _availability_form(client: ChargeSentinelAPIClient, *, form_key: str) -> None:
    with st.form(form_key):
        charger_id = st.text_input("Charger ID", key=f"{form_key}_charger")
        query_time = _datetime_fields("Query Time", f"{form_key}_query")
        submitted = st.form_submit_button("Check Availability")
    result = None
    if submitted:
        result = client.get_availability(
            {"charger_id": charger_id.strip(), "query_time": query_time}
        )

    st.markdown("#### Result")
    if (
        result is not None
        and result.status is ResultStatus.READY
        and result.data is not None
    ):

        def show(data: dict[str, object]) -> None:
            cols = st.columns(2)
            cols[0].metric("Expected free time", _value(data, "expected_free_time"))
            cols[1].metric(
                "Expected wait", f"{_value(data, 'expected_wait_minutes')} min"
            )
            st.caption(f"Model version: {_value(data, 'model_version')}")

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
    _render_operator_availability(client)
    _render_twin(client)


def _render_driver_availability(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Check Charger Availability")
        st.caption("See when a selected charging point is expected to become free.")
        _availability_form(client, form_key="driver_availability")


def _render_recommendation(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Smart Charger Recommendation")
        st.caption("Find a suitable charger and time around your expected arrival.")
        with st.form("recommendation_form"):
            user_eta = _datetime_fields("User ETA", "recommendation_eta")
            preferred = st.text_input("Preferred charger ID (optional)")
            submitted = st.form_submit_button("Recommend Charger")
        if submitted:
            payload: dict[str, object] = {"user_eta": user_eta}
            if preferred.strip():
                payload["preferred_charger_id"] = preferred.strip()
            result = client.get_recommendation(payload)

            def show(data: dict[str, object]) -> None:
                cols = st.columns(3)
                cols[0].metric(
                    "Recommended charger", _value(data, "recommended_charger")
                )
                cols[1].metric("Recommended slot", _value(data, "recommended_slot"))
                cols[2].metric(
                    "Expected wait", f"{_value(data, 'expected_wait_minutes')} min"
                )
                st.write("Alternatives:", _value(data, "alternatives"))
                st.caption(f"Model version: {_value(data, 'model_version')}")

            _render_result(result, "recommendation", show)


def render_driver_view(client: ChargeSentinelAPIClient) -> None:
    st.header("EV Driver View")
    st.markdown(
        '<p class="cs-view-intro">Plan a charging stop with clear availability and '
        "charger recommendations when those services are ready.</p>",
        unsafe_allow_html=True,
    )
    _render_driver_availability(client)
    _render_recommendation(client)
