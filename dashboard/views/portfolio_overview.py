"""
Portfolio Overview Page
Supports selectable portfolio datasets with a persistent per-dataset analysis cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import COLORS, info_box, section_header, style_chart, warning_box
from dataset_helpers import get_dataset_presets, get_portfolio_meta, portfolio_display_label
from portfolio_cache import (
    CACHE_ROOT,
    analyze_dataset,
    cache_is_complete,
    dataset_fingerprint,
    default_dataset_path,
    get_cache_dir,
    load_cache,
    seed_cache_from_legacy,
)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"


# ---------------------------------------------------------------------------
# Streamlit-cached loaders (keyed by file path so each dataset gets its own
# cache slot; TTL=0 means the entry lives until the Streamlit session ends or
# the file is explicitly invalidated)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=0)
def _load_csv(path: str):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def _load_ecl_summary():
    p = MODEL_DIR / "ecl_summary.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def _load_mc_metrics():
    p = MODEL_DIR / "mc_risk_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Dataset selector (lives OUTSIDE any form so conditional widgets re-render
# immediately when the user changes the source radio)
# ---------------------------------------------------------------------------

def _render_dataset_selector() -> tuple:
    """
    Render dataset source radio + conditional preset / custom-path widget.
    Returns (parquet_path: str | None, label: str, source: str).
    parquet_path=None means "use default dataset".
    """
    with st.expander("Dataset", expanded=True):
        ds_mode = st.radio(
            "Source",
            ["Default", "Preset", "Custom path"],
            horizontal=True,
            key="po_ds_mode",
            help="Default = the pre-built portfolio summary from models/dashboard_*.csv",
        )

        if ds_mode == "Default":
            return None, "Default", "default"

        if ds_mode == "Preset":
            presets = get_dataset_presets()
            if not presets:
                st.warning("No preset datasets found under `data/processed/`. Use Custom path.")
                return None, "Default", "default"
            preset_key = st.selectbox(
                "Dataset",
                options=list(presets.keys()),
                format_func=lambda k: presets[k]["label"],
                key="po_ds_preset",
            )
            info = presets[preset_key]
            st.caption(info["description"])
            return info["path"], info["label"], "preset"

        # Custom path
        custom = st.text_input(
            "Portfolio path",
            value=st.session_state.get("_po_last_custom", ""),
            placeholder="/absolute/path/to/dataset.parquet or directory",
            key="po_ds_custom",
        )
        if custom.strip():
            st.session_state["_po_last_custom"] = custom.strip()
            label = portfolio_display_label(custom.strip())
            return custom.strip(), label, "custom"
        st.info("Enter a path to a parquet file or directory.")
        return None, "Default", "default"


# ---------------------------------------------------------------------------
# Cache resolution — load from cache or build it
# ---------------------------------------------------------------------------

def _resolve_cache(parquet_path: str, ds_label: str) -> tuple:
    """
    Ensure a cache exists for the given dataset and return loaded artifacts.

    Returns:
        (artifacts: dict, cache_status: str)
        cache_status ∈ {"cached", "newly_analyzed", "seeded_from_legacy", "failed"}
    """
    fingerprint = dataset_fingerprint(parquet_path)
    cache_dir = get_cache_dir(fingerprint)

    if cache_is_complete(cache_dir):
        return load_cache(cache_dir), "cached"

    # First check if this is the default dataset and we can seed from legacy CSVs
    default_p = default_dataset_path()
    is_default_ds = (
        default_p is not None
        and Path(parquet_path).resolve() == Path(default_p).resolve()
    )
    if is_default_ds and seed_cache_from_legacy(cache_dir):
        return load_cache(cache_dir), "seeded_from_legacy"

    # Full analysis
    with st.spinner(f"Analyzing dataset — {ds_label} (first-time only, result is cached)…"):
        ok, msg = analyze_dataset(parquet_path, cache_dir)

    if not ok:
        return {}, "failed"

    return load_cache(cache_dir), "newly_analyzed"


# ---------------------------------------------------------------------------
# Dataset info card
# ---------------------------------------------------------------------------

def _render_dataset_info(parquet_path, ds_label: str, cache_status: str, artifacts: dict) -> None:
    meta_raw = artifacts.get("metadata") or {}
    pq_meta = get_portfolio_meta(parquet_path) if parquet_path else {}

    status_icon = {
        "cached": "🗃 Cached",
        "seeded_from_legacy": "🗃 Pre-built summary",
        "newly_analyzed": "✅ Freshly analyzed",
        "failed": "❌ Analysis failed",
    }.get(cache_status, cache_status)

    col_path, col_meta = st.columns([3, 2])
    with col_path:
        st.markdown(f"**{ds_label}**")
        if parquet_path:
            st.caption(parquet_path)
        else:
            st.caption("Pre-built summary from `models/dashboard_*.csv`")
    with col_meta:
        parts = [status_icon]
        if pq_meta.get("row_count") is not None:
            parts.append(f"{pq_meta['row_count']:,} rows")
        if pq_meta.get("file_count"):
            kind = "file" if pq_meta.get("is_file") else "directory"
            parts.append(f"{kind} · {pq_meta['file_count']} parquet file(s)")
        if pq_meta.get("row_group_count"):
            parts.append(f"{pq_meta['row_group_count']} row groups")
        st.caption("  ·  ".join(parts))


# ---------------------------------------------------------------------------
# Portfolio composition charts (shared by Default and cached paths)
# ---------------------------------------------------------------------------

def _render_fico_tab(fico_df) -> None:
    col_c, col_t = st.columns([2, 1])
    with col_c:
        if fico_df is not None and not fico_df.empty:
            bucket_col = "fico_bucket" if "fico_bucket" in fico_df.columns else fico_df.columns[0]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fico_df[bucket_col], y=fico_df["count"],
                name="Loan Count", marker_color=COLORS["primary"],
            ))
            if "default_rate" in fico_df.columns:
                fig.add_trace(go.Scatter(
                    x=fico_df[bucket_col], y=fico_df["default_rate"] * 100,
                    name="Default Rate (%)", mode="lines+markers",
                    marker=dict(color=COLORS["danger"], size=10),
                    line=dict(color=COLORS["danger"], width=2), yaxis="y2",
                ))
            fig.update_layout(
                title="Loan Distribution and Default Rate by FICO Score",
                xaxis_title="FICO Bucket",
                yaxis=dict(title="Loan Count"),
                yaxis2=dict(title="Default Rate (%)", side="right", overlaying="y", showgrid=False),
                legend=dict(x=0.01, y=0.99), bargap=0.3,
            )
            style_chart(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("FICO summary not available for this dataset.")
    with col_t:
        info_box(
            "<strong>FICO Score</strong> is the single strongest predictor of default. "
            "Borrowers below 620 (subprime) default at rates 3-5x higher than those above 740."
        )


def _render_ltv_tab(ltv_df) -> None:
    col_c, col_t = st.columns([2, 1])
    with col_c:
        if ltv_df is not None and not ltv_df.empty:
            bucket_col = "ltv_bucket" if "ltv_bucket" in ltv_df.columns else ltv_df.columns[0]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ltv_df[bucket_col], y=ltv_df["count"],
                name="Loan Count", marker_color=COLORS["success"],
            ))
            if "default_rate" in ltv_df.columns:
                fig.add_trace(go.Scatter(
                    x=ltv_df[bucket_col], y=ltv_df["default_rate"] * 100,
                    name="Default Rate (%)", mode="lines+markers",
                    marker=dict(color=COLORS["danger"], size=10),
                    line=dict(color=COLORS["danger"], width=2), yaxis="y2",
                ))
            fig.update_layout(
                title="Loan Distribution and Default Rate by LTV",
                xaxis_title="LTV Bucket",
                yaxis=dict(title="Loan Count"),
                yaxis2=dict(title="Default Rate (%)", side="right", overlaying="y", showgrid=False),
                legend=dict(x=0.01, y=0.99), bargap=0.3,
            )
            style_chart(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("LTV summary not available for this dataset.")
    with col_t:
        info_box(
            "<strong>Loan-to-Value (LTV)</strong> measures borrower leverage. "
            "Above 90% LTV, a small price decline puts the borrower underwater."
        )


def _render_vintage_tab(vintage_df) -> None:
    col_c, col_t = st.columns([2, 1])
    with col_c:
        if vintage_df is not None and not vintage_df.empty:
            data = vintage_df.copy()
            if "origination_year" in data.columns:
                data = data[data["origination_year"] >= 2004]
            fig = go.Figure()
            if "balance" in data.columns:
                fig.add_trace(go.Bar(
                    x=data["origination_year"].astype(str), y=data["balance"] / 1e9,
                    name="Balance ($B)", marker_color=COLORS["purple"],
                ))
            elif "count" in data.columns:
                fig.add_trace(go.Bar(
                    x=data["origination_year"].astype(str), y=data["count"],
                    name="Loan Count", marker_color=COLORS["purple"],
                ))
            if "default_rate" in data.columns:
                fig.add_trace(go.Scatter(
                    x=data["origination_year"].astype(str), y=data["default_rate"] * 100,
                    name="Default Rate (%)", mode="lines+markers",
                    marker=dict(color=COLORS["danger"], size=10),
                    line=dict(color=COLORS["danger"], width=2), yaxis="y2",
                ))
            yaxis_title = "Balance ($B)" if "balance" in data.columns else "Loan Count"
            fig.update_layout(
                title="Portfolio by Origination Year",
                xaxis_title="Year",
                yaxis=dict(title=yaxis_title),
                yaxis2=dict(title="Default Rate (%)", side="right", overlaying="y", showgrid=False),
                bargap=0.3,
            )
            style_chart(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Vintage summary not available for this dataset.")
    with col_t:
        info_box(
            "<strong>2007 vintages</strong> have the highest default rate (17.4%), "
            "originated at peak bubble with loosened standards."
        )


def _render_ecl_tab(fico_df) -> None:
    """ECL-by-FICO tab — only shown when ecl_rate is in the cached summary."""
    col_c, col_t = st.columns([2, 1])
    with col_c:
        if fico_df is not None and "ecl_rate" in fico_df.columns:
            bucket_col = "fico_bucket" if "fico_bucket" in fico_df.columns else fico_df.columns[0]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fico_df[bucket_col], y=fico_df["ecl_rate"] * 100,
                marker_color=[COLORS["danger"], COLORS["warning"], "#eab308",
                              COLORS["success"], COLORS["primary"]],
                text=[f"{r * 100:.1f}%" for r in fico_df["ecl_rate"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="CECL ECL Rate by FICO Bucket",
                xaxis_title="FICO Bucket",
                yaxis_title="ECL Rate (%)",
                showlegend=False, bargap=0.3,
            )
            style_chart(fig, 420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "ECL rate by FICO is not available for the selected dataset. "
                "It requires running the full PD+LGD scoring pipeline on this dataset first."
            )
    with col_t:
        info_box(
            "Subprime borrowers (FICO < 620) have ECL rates roughly 2x higher than super-prime (740+)."
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Portfolio Overview")
    info_box(
        "<strong>What is this dashboard?</strong> This is an end-to-end "
        "CECL (Current Expected Credit Loss) credit risk modeling framework "
        "applied to 3.8 million Fannie Mae mortgage loans. CECL (ASC 326) "
        "requires banks to estimate <strong>lifetime expected losses</strong> "
        "from day one of every loan, considering multiple economic scenarios."
    )

    # ── Dataset selection ────────────────────────────────────────────────────
    parquet_path, ds_label, ds_source = _render_dataset_selector()

    # ── Resolve data ─────────────────────────────────────────────────────────
    if parquet_path is None or ds_source == "default":
        # Default path: use pre-built models/dashboard_*.csv directly
        totals_df = _load_csv(str(MODEL_DIR / "dashboard_portfolio_totals.csv"))
        fico_df = _load_csv(str(MODEL_DIR / "dashboard_fico_summary.csv"))
        ltv_df = _load_csv(str(MODEL_DIR / "dashboard_ltv_summary.csv"))
        vintage_df = _load_csv(str(MODEL_DIR / "dashboard_vintage_summary.csv"))
        cache_status = "seeded_from_legacy"
        artifacts: dict = {
            "metadata": {"source": "pre_built_dashboard_csvs"},
            "totals": totals_df,
            "fico": fico_df,
            "ltv": ltv_df,
            "vintage": vintage_df,
        }
        parquet_path_for_info = None
    else:
        # Validate path exists before trying to analyze
        if not Path(parquet_path).exists():
            st.error(f"Dataset path not found: `{parquet_path}`")
            return

        artifacts, cache_status = _resolve_cache(parquet_path, ds_label)

        if cache_status == "failed":
            st.error(
                f"Could not analyze dataset: `{parquet_path}`. "
                "Check that it is a valid parquet file or directory."
            )
            return

        totals_df = artifacts.get("totals")
        fico_df = artifacts.get("fico")
        ltv_df = artifacts.get("ltv")
        vintage_df = artifacts.get("vintage")
        parquet_path_for_info = parquet_path

    # ── Dataset info card ────────────────────────────────────────────────────
    _render_dataset_info(parquet_path_for_info if parquet_path else None,
                         ds_label, cache_status, artifacts)

    st.markdown("---")

    # ── KPI Row 1: Portfolio Summary ─────────────────────────────────────────
    section_header("Portfolio Summary")
    k1, k2, k3, k4 = st.columns(4)
    if totals_df is not None and not totals_df.empty:
        t = totals_df.iloc[0]

        def _safe(col, default="—"):
            v = t.get(col)
            return v if (v is not None and str(v) != "nan") else None

        tl = _safe("total_loans")
        tb = _safe("total_balance")
        dr = _safe("default_rate")
        mf = _safe("mean_fico")

        k1.metric("Total Loans", f"{float(tl):,.0f}" if tl is not None else "—")
        k2.metric("Outstanding Balance", f"${float(tb)/1e9:,.1f}B" if tb is not None else "—")
        k3.metric("Default Rate", f"{float(dr)*100:.1f}%" if dr is not None else "—")
        k4.metric("Mean FICO", f"{float(mf):.0f}" if mf is not None else "—")
    else:
        k1.metric("Total Loans", "—")
        k2.metric("Outstanding Balance", "—")
        k3.metric("Default Rate", "—")
        k4.metric("Mean FICO", "—")

    # ── KPI Row 2: ECL / MC — labeled as latest available model outputs ──────
    section_header("Expected Credit Loss (CECL)")

    ecl_summary = _load_ecl_summary()
    mc_metrics = _load_mc_metrics()

    # Warn when the selected dataset is not the default (model outputs may not match)
    if ds_source != "default":
        st.warning(
            "**Model outputs below are the latest available results** and may not correspond "
            "to the currently selected dataset. ECL and Monte Carlo metrics are produced by "
            "the full scoring pipeline — re-run the pipeline on this dataset to obtain "
            "dataset-specific model outputs.",
            icon="⚠️",
        )

    e1, e2, e3, e4 = st.columns(4)
    if ecl_summary is not None:
        base_rows = ecl_summary[ecl_summary["scenario"] == "Baseline"]
        adv_rows = ecl_summary[ecl_summary["scenario"] == "Severely Adverse"]
        wt_rows = ecl_summary[ecl_summary["scenario"].str.contains("Weighted", na=False)]
        if not base_rows.empty:
            base = base_rows.iloc[0]
            e1.metric("Baseline ECL", f"${base['total_ecl']/1e6:,.0f}M",
                      f"{base['portfolio_ecl_rate']*100:.2f}%")
        else:
            e1.metric("Baseline ECL", "—")
        if not adv_rows.empty:
            adv = adv_rows.iloc[0]
            e2.metric("Severely Adverse ECL", f"${adv['total_ecl']/1e6:,.0f}M",
                      f"{adv['portfolio_ecl_rate']*100:.2f}%", delta_color="inverse")
        else:
            e2.metric("Severely Adverse ECL", "—")
        if not wt_rows.empty:
            wt = wt_rows.iloc[0]
            e3.metric("Weighted ECL (60/40)", f"${wt['total_ecl']/1e6:,.0f}M",
                      f"{wt['portfolio_ecl_rate']*100:.2f}%")
        else:
            e3.metric("Weighted ECL (60/40)", "—")
    else:
        e1.metric("Baseline ECL", "—")
        e2.metric("Severely Adverse ECL", "—")
        e3.metric("Weighted ECL (60/40)", "—")

    if mc_metrics is not None and not mc_metrics.empty:
        mc = mc_metrics.iloc[0]
        tb_val = mc.get("total_balance", 1)
        pct = f"{mc['var_999']/float(tb_val)*100:.2f}%" if float(tb_val) > 0 else ""
        e4.metric("VaR 99.9% (MC)", f"${mc['var_999']/1e6:,.0f}M", pct, delta_color="inverse")
    else:
        e4.metric("VaR 99.9% (MC)", "—")

    ecl_label = "Latest available model outputs" if ds_source != "default" else ""
    info_box(
        ("<strong>Latest available model outputs.</strong> " if ds_source != "default" else "") +
        "<strong>How to read:</strong> Baseline ECL = expected loss under normal conditions. "
        "Severely Adverse = Fed worst-case (unemployment 10%, HPI -33%). "
        "Weighted ECL (60/40) = reserve on financial statements. "
        "VaR 99.9% = loss exceeded only 0.1% of the time across Monte Carlo simulations."
    )

    # ── Portfolio Composition (dataset-aware via cache) ──────────────────────
    section_header("Portfolio Composition")

    has_data = any(
        df is not None and not df.empty
        for df in [fico_df, ltv_df, vintage_df]
    )

    if has_data:
        show_ecl_tab = (fico_df is not None and "ecl_rate" in (fico_df.columns if fico_df is not None else []))
        tabs_labels = ["By FICO Score", "By LTV Ratio", "By Vintage Year"]
        if show_ecl_tab:
            tabs_labels.append("ECL by Segment")
        tabs = st.tabs(tabs_labels)

        with tabs[0]:
            _render_fico_tab(fico_df)
        with tabs[1]:
            _render_ltv_tab(ltv_df)
        with tabs[2]:
            _render_vintage_tab(vintage_df)
        if show_ecl_tab:
            with tabs[3]:
                _render_ecl_tab(fico_df)
    else:
        st.info("Portfolio composition charts are not available for the selected dataset.")

    # ── Production Context ────────────────────────────────────────────────────
    section_header("How This Framework Operates in Production")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Scoring Pipeline")
        info_box(
            "<strong>Daily:</strong> New loans scored through PD + LGD models for day-one ECL reserve.<br>"
            "<strong>Monthly:</strong> Full portfolio re-scored with updated FRED macro data."
        )
    with p2:
        st.markdown("#### Governance Cycle")
        info_box(
            "<strong>Quarterly:</strong> PSI computed, calibration refreshed, stress tests updated.<br>"
            "<strong>Annually:</strong> Full re-development. Independent validation by challenge function."
        )
