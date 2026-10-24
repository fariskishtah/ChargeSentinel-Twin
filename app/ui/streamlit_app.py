"""ChargeSentinel Streamlit dashboard entry point."""

import streamlit as st

from app.ui.api_client import ChargeSentinelAPIClient, ResultStatus
from app.ui.views import render_driver_view, render_operator_view

st.set_page_config(page_title="ChargeSentinel Twin", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; }
    h1, h2, h3 { color: #0b1f4b; }
    div[data-testid="stMetric"] {
        background: #f4f8ff;
        border: 1px solid #d8e6ff;
        border-radius: 0.65rem;
        padding: 0.75rem;
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: #1428a0;
        color: white;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

client = ChargeSentinelAPIClient()
health_result = client.health()

st.title("ChargeSentinel Twin")
st.caption("Renewable-aware EV charging and grid-load intelligence")

with st.sidebar:
    st.subheader("Service Status")
    if health_result.status is ResultStatus.READY:
        st.success("API Online")
    elif health_result.status is ResultStatus.API_OFFLINE:
        st.error("API Offline")
    else:
        st.error("Backend Error")

operator_tab, driver_tab = st.tabs(["Operator View", "EV Driver View"])

with operator_tab:
    render_operator_view(client, health_result)

with driver_tab:
    render_driver_view(client)
