"""Operator and EV driver dashboard views."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time

import streamlit as st

from app.ui.api_client import APIResult, ChargeSentinelAPIClient, ResultStatus

NOT_READY_MESSAGES = {
    "solar": "Solar model is not integrated yet.",
    "demand": "Demand model is not integrated yet.",
    "anomaly": "Anomaly model is not integrated yet.",
    "availability": "Availability model is not integrated yet.",
    "recommendation": "Recommendation component is not integrated yet.",
    "digital_twin": "Digital Twin component is not integrated yet.",
}


def _iso_utc(selected_date: date, selected_time: time) -> str:
    return datetime.combine(selected_date, selected_time, tzinfo=UTC).isoformat()


def _datetime_fields(label: str, key: str) -> str:
    now = datetime.now(UTC)
    date_col, time_col = st.columns(2)
    selected_date = date_col.date_input(
        f"{label} date (UTC)", value=now.date(), key=f"{key}_date"
    )
    selected_time = time_col.time_input(
        f"{label} time (UTC)",
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
    return value if value is not None else "--"


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
        st.error("INVALID INPUT — Check the submitted values and try again.")
    elif result.status is ResultStatus.API_OFFLINE:
        st.error("API OFFLINE — Start the FastAPI service and try again.")
    else:
        st.error("BACKEND ERROR — The request could not be completed safely.")


def render_health(result: APIResult) -> None:
    st.subheader("System Health")
    if result.status is ResultStatus.READY and result.data is not None:
        st.success("READY — API Online")
        first, second = st.columns(2)
        first.metric("API status", str(_value(result.data, "status")).upper())
        second.metric("API version", _value(result.data, "api_version"))
    elif result.status is ResultStatus.API_OFFLINE:
        st.error("API OFFLINE — Dashboard controls remain available.")
    else:
        st.error("BACKEND ERROR — Health response is unavailable.")


def _render_solar(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Solar Forecast")
        with st.form("solar_form"):
            forecast_origin = _datetime_fields("Forecast origin", "solar_origin")
            horizon = st.radio("Horizon (hours)", [24, 48], horizontal=True)
            submitted = st.form_submit_button("Run Solar Forecast")
        if submitted:
            result = client.get_solar_forecast(
                {"forecast_origin": forecast_origin, "horizon_hours": horizon}
            )

            def show(data: dict[str, object]) -> None:
                cols = st.columns(4)
                cols[0].metric("Forecast timestamp", _value(data, "timestamp"))
                cols[1].metric("Horizon", f"{_value(data, 'horizon_hours')} h")
                cols[2].metric("Forecast", f"{_value(data, 'forecast_kw')} kW")
                cols[3].metric("Model version", _value(data, "model_version"))

            _render_result(result, "solar", show)


def _render_demand(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("EV Demand Forecast")
        with st.form("demand_form"):
            forecast_origin = _datetime_fields("Forecast origin", "demand_origin")
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
        st.subheader("Anomaly Detection")
        st.caption("Only leakage-audited features should be submitted.")
        with st.form("anomaly_form"):
            charger_id = st.text_input("Charger ID", key="anomaly_charger")
            raw_features = st.text_area(
                "Safe feature payload (JSON object)",
                value="{}",
                help="Fatima must confirm the final safe feature names.",
            )
            submitted = st.form_submit_button("Check Session")
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
                st.write("Reasons:", _value(data, "reason"))

            _render_result(result, "anomaly", show)


def _availability_form(
    client: ChargeSentinelAPIClient, *, form_key: str
) -> None:
    with st.form(form_key):
        charger_id = st.text_input("Charger ID", key=f"{form_key}_charger")
        query_time = _datetime_fields("Query time", f"{form_key}_query")
        submitted = st.form_submit_button("Check Availability")
    if submitted:
        result = client.get_availability(
            {"charger_id": charger_id.strip(), "query_time": query_time}
        )

        def show(data: dict[str, object]) -> None:
            cols = st.columns(3)
            cols[0].metric("Expected free time", _value(data, "expected_free_time"))
            cols[1].metric(
                "Expected wait", f"{_value(data, 'expected_wait_minutes')} min"
            )
            cols[2].metric("Model version", _value(data, "model_version"))

        _render_result(result, "availability", show)


def _render_operator_availability(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Charger Availability")
        _availability_form(client, form_key="operator_availability")


def _render_twin(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Digital Twin")
        st.warning("SIMULATION ONLY — No live charger or grid control.")
        with st.form("digital_twin_form"):
            demand_multiplier = st.number_input(
                "Demand multiplier", min_value=0.1, value=1.0, step=0.1
            )
            solar_multiplier = st.number_input(
                "Solar multiplier", min_value=0.0, value=1.0, step=0.1
            )
            offline_raw = st.text_input(
                "Offline charger IDs",
                help="Optional comma-separated charger IDs.",
            )
            submitted = st.form_submit_button("Run Simulation")
        if submitted:
            offline_chargers = [
                item.strip() for item in offline_raw.split(",") if item.strip()
            ]
            result = client.simulate_digital_twin(
                {
                    "demand_multiplier": demand_multiplier,
                    "solar_multiplier": solar_multiplier,
                    "offline_chargers": offline_chargers,
                }
            )

            def show(data: dict[str, object]) -> None:
                st.info("SIMULATION")
                cols = st.columns(3)
                cols[0].metric(
                    "Supply-demand balance", _value(data, "supply_demand_balance")
                )
                cols[1].metric(
                    "Estimated utilization", _value(data, "estimated_utilization")
                )
                cols[2].metric(
                    "Estimated wait impact", _value(data, "estimated_wait_impact")
                )
                st.write("Recommendation:", _value(data, "recommendation"))
                st.caption(f"Simulation version: {_value(data, 'simulation_version')}")

            _render_result(result, "digital_twin", show)


def render_operator_view(
    client: ChargeSentinelAPIClient, health_result: APIResult
) -> None:
    st.header("Operator View")
    render_health(health_result)
    _render_solar(client)
    _render_demand(client)
    _render_anomaly(client)
    _render_operator_availability(client)
    _render_twin(client)


def _render_driver_availability(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Charger Availability")
        _availability_form(client, form_key="driver_availability")


def _render_recommendation(client: ChargeSentinelAPIClient) -> None:
    with st.container(border=True):
        st.subheader("Smart Charger Recommendation")
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
    st.caption("Availability and recommendations appear only after model integration.")
    _render_driver_availability(client)
    _render_recommendation(client)
