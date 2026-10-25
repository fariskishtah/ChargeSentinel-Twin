"""Centralized visual styling for the ChargeSentinel dashboard."""

from __future__ import annotations

import streamlit as st


def apply_global_styles() -> None:
    """Apply the shared light theme and accessible component styling."""

    st.markdown(
        """
        <style>
        :root {
            --cs-navy: #0B1F4B;
            --cs-text: #1D2939;
            --cs-muted: #526174;
            --cs-blue: #1428A0;
            --cs-blue-hover: #0F1F7A;
            --cs-focus: #2563EB;
            --cs-page: #F6F8FC;
            --cs-surface: #FFFFFF;
            --cs-surface-blue: #F2F6FF;
            --cs-border: #D8E0EC;
            --cs-success: #176B3A;
            --cs-success-bg: #EAF7EF;
            --cs-warning: #7A4B00;
            --cs-warning-bg: #FFF7E5;
            --cs-error: #A62B2B;
            --cs-error-bg: #FDECEC;
            --cs-radius: 0.75rem;
        }

        .stApp {
            background: var(--cs-page);
            color: var(--cs-text);
        }

        [data-testid="stHeader"] {
            background: var(--cs-page);
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 76rem;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stHeadingWithActionElements"] {
            color: var(--cs-navy);
            letter-spacing: -0.015em;
        }

        h2 {
            margin-top: 0.35rem;
        }

        p, label, li, [data-testid="stMarkdownContainer"] {
            color: var(--cs-text);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--cs-muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .cs-hero {
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-top: 4px solid var(--cs-blue);
            border-radius: var(--cs-radius);
            box-shadow: 0 8px 24px rgba(11, 31, 75, 0.07);
            margin-bottom: 1.25rem;
            padding: 1.55rem 1.65rem 1.4rem;
        }

        .cs-kicker {
            color: var(--cs-blue);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.09em;
            margin: 0 0 0.35rem;
            text-transform: uppercase;
        }

        .cs-hero h1 {
            color: var(--cs-navy);
            font-size: clamp(2rem, 5vw, 3.15rem);
            font-weight: 760;
            letter-spacing: -0.04em;
            line-height: 1.04;
            margin: 0;
        }

        .cs-subtitle {
            color: var(--cs-text);
            font-size: 1.08rem;
            font-weight: 520;
            line-height: 1.5;
            margin: 0.65rem 0 1rem;
        }

        .cs-status-row {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }

        .cs-status-pill,
        .cs-simulation-badge,
        .cs-severity-badge {
            align-items: center;
            border: 1px solid;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.035em;
            line-height: 1;
            padding: 0.5rem 0.7rem;
        }

        .cs-status-ready {
            background: var(--cs-success-bg);
            border-color: #AED8BE;
            color: var(--cs-success);
        }

        .cs-status-offline,
        .cs-status-error {
            background: var(--cs-error-bg);
            border-color: #EABBBB;
            color: var(--cs-error);
        }

        .cs-simulation-badge {
            background: var(--cs-warning-bg);
            border-color: #E7CB8B;
            color: var(--cs-warning);
            margin: 0.1rem 0 0.35rem;
        }

        .cs-view-intro {
            color: var(--cs-muted);
            font-size: 1rem;
            line-height: 1.55;
            margin: -0.35rem 0 1rem;
            max-width: 48rem;
        }

        .cs-group-label {
            color: var(--cs-blue);
            font-size: 0.78rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            margin: 0.85rem 0 0.15rem;
            text-transform: uppercase;
        }

        .cs-recommendation {
            background: var(--cs-surface-blue);
            border-left: 4px solid var(--cs-blue);
            border-radius: 0.4rem;
            color: var(--cs-text);
            line-height: 1.55;
            margin-top: 0.8rem;
            padding: 0.85rem 1rem;
        }

        .cs-recommendation strong {
            color: var(--cs-navy);
        }

        .cs-severity-low {
            background: var(--cs-success-bg);
            border-color: #AED8BE;
            color: var(--cs-success);
        }

        .cs-severity-medium {
            background: var(--cs-warning-bg);
            border-color: #E7CB8B;
            color: var(--cs-warning);
        }

        .cs-severity-high,
        .cs-severity-critical {
            background: var(--cs-error-bg);
            border-color: #EABBBB;
            color: var(--cs-error);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--cs-surface);
            border-color: var(--cs-border);
            border-radius: var(--cs-radius);
            box-shadow: 0 4px 16px rgba(11, 31, 75, 0.045);
        }

        div[data-testid="stMetric"] {
            background: var(--cs-surface-blue);
            border: 1px solid #CFDBF2;
            border-radius: 0.65rem;
            min-height: 6.15rem;
            padding: 0.8rem 0.9rem;
        }

        div[data-testid="stMetricLabel"] p {
            color: #344258;
            font-size: 0.88rem;
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: var(--cs-navy);
            font-size: 1.65rem;
            font-weight: 720;
        }

        div[data-testid="stForm"] {
            background: #FBFCFE;
            border-color: var(--cs-border);
            border-radius: 0.65rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-radius: 0.65rem;
            gap: 0.35rem;
            padding: 0.28rem;
        }

        [data-testid="stTab"] {
            border-radius: 0.45rem;
            color: var(--cs-muted);
            font-size: 0.95rem;
            font-weight: 680;
            min-height: 2.75rem;
            padding: 0 1.1rem;
        }

        [data-testid="stTab"][aria-selected="true"] {
            background: var(--cs-blue);
            color: #FFFFFF;
        }

        [data-testid="stTab"][aria-selected="true"] p {
            color: #FFFFFF;
        }

        [data-testid="stTab"] .react-aria-SelectionIndicator {
            display: none;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background: var(--cs-blue);
            border: 1px solid var(--cs-blue);
            border-radius: 0.5rem;
            color: #FFFFFF;
            font-weight: 700;
            min-height: 2.75rem;
            padding: 0.55rem 1rem;
        }

        .stButton > button p,
        [data-testid="stFormSubmitButton"] > button p {
            color: #FFFFFF;
        }

        @media (hover: hover) {
            .stButton > button:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                background: var(--cs-blue-hover);
                border-color: var(--cs-blue-hover);
                color: #FFFFFF;
            }
        }

        .stButton > button:focus-visible,
        [data-testid="stFormSubmitButton"] > button:focus-visible,
        [data-testid="stTab"]:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="radio"]:focus-visible {
            outline: 3px solid var(--cs-focus) !important;
            outline-offset: 2px;
        }

        [data-testid="stAlert"] {
            border-radius: 0.55rem;
            font-size: 0.96rem;
        }

        [data-testid="stAlert"] p {
            color: inherit;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] input {
            color: var(--cs-text);
        }

        @media (max-width: 48rem) {
            [data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 4rem;
            }

            .cs-hero {
                padding: 1.25rem 1.1rem;
            }

            .cs-subtitle {
                font-size: 1rem;
            }

            [data-testid="stTab"] {
                flex: 1;
                justify-content: center;
                padding: 0 0.55rem;
            }

            input, textarea, select {
                font-size: 1rem !important;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
