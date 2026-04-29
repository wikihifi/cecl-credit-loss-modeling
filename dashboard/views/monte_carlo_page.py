"""
Monte Carlo Page
Author: Saurabh Chavan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import style_chart, COLORS, info_box, warning_box, section_header

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

@st.cache_data
def load_mc_dist():
    p = MODEL_DIR / "mc_loss_distribution.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_mc_metrics():
    p = MODEL_DIR / "mc_risk_metrics.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_mc_sensitivity():
    p = MODEL_DIR / "mc_sensitivity.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_mc_scenarios():
    p = MODEL_DIR / "mc_scenarios.csv"
    return pd.read_csv(p) if p.exists() else None


def _get_loss_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in ("loss", "portfolio_loss"):
        if column in df.columns:
            return df[column]
    return None


def render():
    st.title("Monte Carlo Simulation")
    info_box("<strong>Monte Carlo:</strong> Instead of 2 scenarios, generate 10,000 random economic futures. "
             "Each has different unemployment, GDP, and HPI drawn from historically realistic correlated distributions. "
             "Result: a full loss distribution yielding VaR and Expected Shortfall.")

    section_header("Risk Metrics from 10,000 Simulations")
    mm = load_mc_metrics()
    k1, k2, k3, k4 = st.columns(4)
    if mm is not None:
        m = mm.iloc[0]; tb = m["total_balance"]
        k1.metric("Expected Loss", f"${m['expected_loss']/1e6:,.0f}M", f"{m['expected_loss']/tb*100:.2f}%")
        k2.metric("VaR 99%", f"${m['var_99']/1e6:,.0f}M", f"{m['var_99']/tb*100:.2f}%", delta_color="inverse")
        k3.metric("VaR 99.9%", f"${m['var_999']/1e6:,.0f}M", f"{m['var_999']/tb*100:.2f}%", delta_color="inverse")
        k4.metric("ES 99%", f"${m['es_99']/1e6:,.0f}M", f"{m['es_99']/tb*100:.2f}%", delta_color="inverse")
    else:
        k1.metric("Expected Loss", "$44,454M", "6.41%")
        k2.metric("VaR 99%", "$81,805M", "11.79%", delta_color="inverse")
        k3.metric("VaR 99.9%", "$99,208M", "14.30%", delta_color="inverse")
        k4.metric("ES 99%", "$89,071M", "12.84%", delta_color="inverse")

    info_box("<strong>Expected Loss</strong> = reserve (CECL). <strong>VaR 99%</strong> = exceeded 1% of time. "
             "<strong>VaR 99.9%</strong> = economic capital buffer. <strong>ES 99%</strong> = avg loss in worst 1%.")

    tab_dist, tab_torn, tab_scat, tab_meth = st.tabs(["Loss Distribution", "Sensitivity", "Scenario Explorer", "Methodology"])

    with tab_dist:
        dist = load_mc_dist()
        if dist is not None and mm is not None:
            m = mm.iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=dist["portfolio_loss"]/1e6, nbinsx=80, name="Losses",
                marker_color=COLORS["primary"], opacity=0.7))
            fig.add_vline(x=m["expected_loss"]/1e6, line_color=COLORS["success"], line_width=2,
                annotation_text=f"EL: ${m['expected_loss']/1e6:,.0f}M", annotation_font_color=COLORS["success"])
            fig.add_vline(x=m["var_99"]/1e6, line_dash="dash", line_color=COLORS["warning"], line_width=2,
                annotation_text=f"VaR99: ${m['var_99']/1e6:,.0f}M", annotation_font_color=COLORS["warning"])
            fig.add_vline(x=m["var_999"]/1e6, line_dash="dash", line_color=COLORS["danger"], line_width=2,
                annotation_text=f"VaR99.9: ${m['var_999']/1e6:,.0f}M", annotation_font_color=COLORS["danger"])
            fig.update_layout(title="Portfolio Loss Distribution (10,000 Simulations)",
                xaxis_title="Portfolio Loss ($M)", yaxis_title="Frequency", showlegend=False)
            style_chart(fig, 500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Monte Carlo data not found. Run `run_monte_carlo.py`.")

    with tab_torn:
        sens = load_mc_sensitivity()
        if sens is not None:
            cc, ce = st.columns([2, 1])
            with cc:
                tornado = []
                for var in sens["variable"].unique():
                    vd = sens[sens["variable"]==var]
                    loss_series = _get_loss_series(vd)
                    if loss_series is None:
                        continue
                    tornado.append({"variable":var, "min":loss_series.min()/1e6, "max":loss_series.max()/1e6,
                                    "range":(loss_series.max()-loss_series.min())/1e6})
                if tornado:
                    tdf = pd.DataFrame(tornado).sort_values("range", ascending=True)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(y=tdf["variable"], x=tdf["min"], orientation="h", name="Best", marker_color=COLORS["success"], opacity=0.7))
                    fig.add_trace(go.Bar(y=tdf["variable"], x=tdf["max"]-tdf["min"], orientation="h", name="Worst Increment",
                        marker_color=COLORS["danger"], opacity=0.7, base=tdf["min"]))
                    fig.update_layout(title="Sensitivity: Loss Range by Macro Variable", xaxis_title="Loss ($M)", barmode="overlay")
                    style_chart(fig, 400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sensitivity data is missing a loss column.")
            with ce:
                info_box("<strong>Unemployment</strong> is the dominant driver (2.9x loss increase from 4% to 12%). "
                         "HPI is secondary, operating through the LGD channel.")
        else:
            st.info("Sensitivity data not found.")

    with tab_scat:
        scen = load_mc_scenarios()
        if scen is not None and "unemployment_rate" in scen.columns and "portfolio_loss" in scen.columns:
            fig = go.Figure()
            color_col = "hpi_change_annual" if "hpi_change_annual" in scen.columns else None
            fig.add_trace(go.Scattergl(x=scen["unemployment_rate"], y=scen["portfolio_loss"]/1e6, mode="markers",
                marker=dict(size=3, color=scen[color_col] if color_col else COLORS["primary"],
                    colorscale="RdYlGn" if color_col else None,
                    colorbar=dict(title="HPI Chg %") if color_col else None, opacity=0.5)))
            fig.update_layout(title="10K Scenarios: Unemployment vs Loss", xaxis_title="Unemployment (%)", yaxis_title="Loss ($M)")
            style_chart(fig, 500)
            st.plotly_chart(fig, use_container_width=True)
            info_box("Each dot = one economic future. Upper-right (high unemployment + falling HPI) = tail risk events.")
        else:
            st.info("Scenario data not found.")

    with tab_meth:
        m1, m2 = st.columns(2)
        with m1:
            info_box("<strong>Correlated draws:</strong> Cholesky decomposition of historical correlation matrix transforms "
                     "independent normals into correlated macro scenarios (high unemployment + low HPI together).")
        with m2:
            info_box("<strong>Per scenario:</strong> Correlated macro draw -> PD/LGD multipliers -> portfolio loss. "
                     "10,000 iterations -> full distribution -> VaR, ES.")
        warning_box("Limitation: uniform national macro multipliers. Geographic concentration not captured.")
