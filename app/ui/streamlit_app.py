"""ChargeSentinel Streamlit dashboard entry point."""

import streamlit as st

from app.ui.api_client import ChargeSentinelAPIClient, ResultStatus
from app.ui.styles import apply_global_styles
from app.ui.views import render_driver_view, render_operator_view

st.set_page_config(
    page_title="ChargeSentinel Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_global_styles()

client = ChargeSentinelAPIClient()
health_result = client.health()

if health_result.status is ResultStatus.READY:
    status_class = "cs-status-ready"
    status_text = "● READY · API ONLINE"
elif health_result.status is ResultStatus.API_OFFLINE:
    status_class = "cs-status-offline"
    status_text = "● API OFFLINE"
else:
    status_class = "cs-status-error"
    status_text = "● BACKEND ERROR"

st.markdown(
    f"""
    <header class="cs-hero">
        <p class="cs-kicker">Samsung Innovation Campus — Smart Energy Project</p>
        <h1>ChargeSentinel Twin</h1>
        <p class="cs-subtitle">
            Renewable-Aware EV Charging &amp; Grid Intelligence Platform
        </p>
        <div class="cs-status-row" role="status" aria-label="{status_text}">
            <span class="cs-status-pill {status_class}">{status_text}</span>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)

operator_tab, driver_tab = st.tabs(["Operator View", "EV Driver View"])

with operator_tab:
    render_operator_view(client, health_result)

with driver_tab:
    render_driver_view(client)
