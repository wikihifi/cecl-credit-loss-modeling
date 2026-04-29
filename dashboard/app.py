"""
CECL Credit Risk Modeling Dashboard
=====================================
Run with: streamlit run dashboard/app.py

Author: Saurabh Chavan
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CECL Credit Risk Dashboard",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eff6ff !important;
        border-color: #2563eb !important;
        color: #1e40af !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    '<h2 style="color:#1e40af; margin-bottom:0;">CECL Credit Risk</h2>'
    '<p style="color:#64748b; font-size:0.9rem; margin-top:0;">Modeling Dashboard</p>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

st.sidebar.markdown('<p style="color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.05em;">OVERVIEW</p>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "nav",
    [
        "Portfolio Overview",
        "PD Model",
        "LGD Model",
        "Stress Testing",
        "Monte Carlo",
        "Simulation Runs (Analytics)",
        "Simulation Runs (Classic)",
        "Loan Scorer",
    ],
    key="main_nav",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="color: #94a3b8; font-size: 0.78rem;">'
    "<strong>Saurabh Chavan</strong><br>"
    "CECL Credit Risk Modeling<br>"
    "3.8M loans | $694B portfolio<br>"
    "Fannie Mae 2005-2007 vintages"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page Routing (imports from views/ folder, NOT pages/)
# ---------------------------------------------------------------------------
if page == "Portfolio Overview":
    from views import portfolio_overview
    portfolio_overview.render()
elif page == "PD Model":
    from views import pd_model_page
    pd_model_page.render()
elif page == "LGD Model":
    from views import lgd_model_page
    lgd_model_page.render()
elif page == "Stress Testing":
    from views import stress_testing_page
    stress_testing_page.render()
elif page == "Monte Carlo":
    from views import monte_carlo_page
    monte_carlo_page.render()
elif page == "Simulation Runs (Analytics)":
    from views import simulation_runs_analytics_page
    simulation_runs_analytics_page.render()
elif page == "Simulation Runs (Classic)":
    from views import simulation_runs_page
    simulation_runs_page.render()
elif page == "Loan Scorer":
    from views import loan_scorer_page
    loan_scorer_page.render()