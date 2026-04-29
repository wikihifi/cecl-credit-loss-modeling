"""
Simulation Runs (Analytics) — redesigned analytics-first view.

Reuses all data loading, chart, state, and launch logic from
simulation_runs_page (Classic). Only the layout and information
hierarchy are new.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import COLORS, info_box, section_header, style_chart

# ── Import every reusable piece from the Classic page ──────────────────────
from views.simulation_runs_page import (
    STEP_NAME_MAP,
    _LABEL_COLORS,
    _build_run_record,
    _chart_convergence,
    _chart_error_to_reference,
    _chart_runtime_vs_quality,
    _chart_stability,
    _chart_step_scaling,
    _get_loss_series,
    _poll_active_runs,
    _render_config_table,
    _render_launch_form,
    _render_raw_log,
    _render_run_charts,
    _render_sim_count_analytics,
    _render_step_metrics,
    _status_badge,
    build_run_analytics_df,
    load_run_dist,
    load_run_risk,
    load_run_scenarios,
    load_run_sensitivity,
    load_run_summary,
    parse_log_metrics,
    read_log_tail,
)
from run_state import MODEL_DIR, RunInfo, discover_runs, is_alive, is_run_complete, update_state

# ---------------------------------------------------------------------------
# Page-level CSS — layered on top of the app-level CSS in app.py
# ---------------------------------------------------------------------------

_PAGE_CSS = """
<style>
/* KPI strip cards */
.kpi-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.kpi-card .kpi-label {
    font-size: 0.73rem;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.kpi-card .kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.kpi-card .kpi-sub {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
}
/* Status pills */
.pill-running  { background:#dbeafe; color:#1e40af; border-radius:12px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.pill-complete { background:#dcfce7; color:#15803d; border-radius:12px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.pill-failed   { background:#fee2e2; color:#b91c1c; border-radius:12px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
/* Run history rows */
.run-row {
    background:#ffffff;
    border:1px solid #e2e8f0;
    border-radius:8px;
    padding:10px 14px;
    margin-bottom:6px;
    display:flex;
    align-items:center;
    gap:12px;
}
.run-row-prefix { font-family:monospace; font-size:0.82rem; color:#334155; flex:1; }
/* Section divider */
.section-rule { border:none; border-top:1px solid #e2e8f0; margin:20px 0; }
</style>
"""


# ---------------------------------------------------------------------------
# Top status strip
# ---------------------------------------------------------------------------

def _render_status_strip(all_runs: list[RunInfo], analytics_df: pd.DataFrame | None) -> None:
    n_running = sum(1 for r in all_runs if r.status == "running")
    n_complete = sum(1 for r in all_runs if r.status == "completed")
    n_failed = sum(1 for r in all_runs if r.status == "failed")

    best_el_str = "—"
    median_rt_str = "—"
    best_eff_str = "—"

    if analytics_df is not None:
        completed = analytics_df[analytics_df["status"] == "completed"]
        el_s = pd.to_numeric(completed.get("expected_loss", pd.Series(dtype=float)), errors="coerce").dropna()
        rt_s = pd.to_numeric(completed.get("total_runtime_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
        if not el_s.empty:
            best_el_str = f"${el_s.min() / 1e6:,.0f}M"
        if not rt_s.empty:
            median_rt_str = f"{rt_s.median():.0f}s"
        # Best efficiency = lowest EL error per second (from convergence metrics if present)
        if "error_per_second" in completed.columns:
            eff_s = pd.to_numeric(completed["error_per_second"], errors="coerce").dropna()
            if not eff_s.empty:
                best_run = completed.loc[eff_s.idxmin(), "prefix"] if not eff_s.empty else None
                if best_run:
                    best_eff_str = str(best_run)[-14:]

    cols = st.columns(6)

    def _kpi(col, label, value, sub=""):
        sub_part = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'{sub_part}'
            f'</div>',
            unsafe_allow_html=True,
        )

    _kpi(cols[0], "Active", str(n_running), "running now")
    _kpi(cols[1], "Completed", str(n_complete), "all time")
    _kpi(cols[2], "Failed", str(n_failed), "")
    _kpi(cols[3], "Median runtime", median_rt_str, "completed runs")
    _kpi(cols[4], "Best EL", best_el_str, "lowest expected loss")
    _kpi(cols[5], "Best efficiency", best_eff_str, "lowest err/sec run")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Active-run live strip (right panel companion to the launch form)
# ---------------------------------------------------------------------------

def _render_active_run_panel(all_runs: list[RunInfo]) -> None:
    running = [r for r in all_runs if r.status == "running"]
    if not running:
        st.markdown(
            '<div style="background:#f0fdf4; border:1px dashed #86efac; border-radius:10px; '
            'padding:20px; text-align:center; color:#15803d; font-size:0.9rem;">'
            '🟢 No active runs — ready to launch'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for run in running:
        with st.container():
            st.markdown(
                f'<div style="background:#eff6ff; border-left:4px solid #2563eb; '
                f'border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:8px;">'
                f'<span style="font-family:monospace;font-size:0.82rem;color:#1e40af;">{run.prefix}</span> '
                f'<span class="pill-running">RUNNING</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            tail = read_log_tail(run.log_path or "", n=12)
            st.code(tail or "(log initializing…)", language=None)


# ---------------------------------------------------------------------------
# Compact outcome analytics
# ---------------------------------------------------------------------------

def _render_analytics_dashboard(analytics_df: pd.DataFrame) -> None:
    completed = analytics_df[analytics_df["status"] == "completed"].copy()
    if completed.empty:
        st.info("Analytics available once at least one run completes.")
        return

    st.markdown(
        '<p style="font-size:0.8rem;color:#64748b;margin-bottom:12px;">'
        'Baseline = median loss_rate across completed runs. '
        'Improved = loss rate &lt; baseline − 5 %.'
        '</p>',
        unsafe_allow_html=True,
    )

    tab_outcomes, tab_quality, tab_table = st.tabs(
        ["📊 Outcomes", "🎯 Forecast Quality", "📋 Run Table"]
    )

    with tab_outcomes:
        _render_outcomes_tab(completed)

    with tab_quality:
        _render_sim_count_analytics(completed)

    with tab_table:
        _render_run_table(completed)


def _render_outcomes_tab(completed: pd.DataFrame) -> None:
    row1_l, row1_r = st.columns(2)
    row2_l, row2_r = st.columns(2)

    palette = [COLORS["primary"], COLORS["warning"], COLORS["success"],
               COLORS["danger"], COLORS["purple"], COLORS["cyan"]]

    # Chart A: Simulations vs Expected Loss by backend
    with row1_l:
        plot = completed.dropna(subset=["n_simulations", "expected_loss"]).copy()
        plot["n_simulations"] = pd.to_numeric(plot["n_simulations"], errors="coerce")
        if not plot.empty:
            fig = go.Figure()
            for i, be in enumerate(plot["backend"].fillna("unknown").unique()):
                sub = plot[plot["backend"].fillna("unknown") == be]
                fig.add_trace(go.Scatter(
                    x=sub["n_simulations"],
                    y=sub["expected_loss"] / 1e6,
                    mode="markers+text",
                    text=sub["prefix"].str[-7:],
                    textposition="top center",
                    textfont=dict(size=8),
                    marker=dict(size=11, color=palette[i % len(palette)]),
                    name=str(be),
                    hovertemplate="%{text}<br>Sims: %{x:,}<br>EL: $%{y:,.0f}M<extra></extra>",
                ))
            fig.update_layout(title="Simulations vs Expected Loss",
                              xaxis_title="N Simulations", yaxis_title="EL ($M)")
            style_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True, key="rds-sims-vs-el")
        else:
            st.info("Insufficient data.")

    # Chart B: Runtime vs Expected Loss (efficiency view)
    with row1_r:
        plot = completed.dropna(subset=["total_runtime_seconds", "expected_loss"]).copy()
        if not plot.empty:
            fig = go.Figure()
            for i, be in enumerate(plot["backend"].fillna("unknown").unique()):
                sub = plot[plot["backend"].fillna("unknown") == be]
                fig.add_trace(go.Scatter(
                    x=pd.to_numeric(sub["total_runtime_seconds"], errors="coerce"),
                    y=pd.to_numeric(sub["expected_loss"], errors="coerce") / 1e6,
                    mode="markers",
                    marker=dict(size=12, color=palette[i % len(palette)], opacity=0.85,
                                symbol="circle"),
                    name=str(be),
                    text=sub["prefix"].str[-7:],
                    hovertemplate="%{text}<br>Runtime: %{x:.0f}s<br>EL: $%{y:,.0f}M<extra></extra>",
                ))
            fig.update_layout(title="Runtime vs Expected Loss",
                              xaxis_title="Runtime (s)", yaxis_title="EL ($M)")
            style_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True, key="rds-rt-vs-el")
        else:
            st.info("Insufficient data.")

    # Chart C: Backend avg EL bar
    with row2_l:
        plot = completed.dropna(subset=["backend", "expected_loss"]).copy()
        if not plot.empty:
            grp = plot.groupby("backend").agg(
                el_mean=("expected_loss", "mean"),
                rt_mean=("total_runtime_seconds", "mean"),
                count=("prefix", "count"),
            ).reset_index()
            fig = go.Figure(go.Bar(
                x=grp["backend"],
                y=grp["el_mean"] / 1e6,
                marker_color=COLORS["primary"],
                text=(grp["el_mean"] / 1e6).map(lambda v: f"${v:,.0f}M"),
                textposition="outside",
                customdata=grp[["rt_mean", "count"]].values,
                hovertemplate=(
                    "%{x}<br>Avg EL: $%{y:,.0f}M<br>"
                    "Avg runtime: %{customdata[0]:.0f}s<br>"
                    "Runs: %{customdata[1]}<extra></extra>"
                ),
            ))
            fig.update_layout(title="Avg Expected Loss by Backend",
                              xaxis_title="Backend", yaxis_title="Avg EL ($M)")
            style_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True, key="rds-backend-el")
        else:
            st.info("Insufficient data.")

    # Chart D: Outcome classification donut
    with row2_r:
        label_counts = completed["improvement_label"].value_counts()
        if not label_counts.empty:
            labels = label_counts.index.tolist()
            values = label_counts.values.tolist()
            colors = [_LABEL_COLORS.get(lbl, "#d1d5db") for lbl in labels]
            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                hole=0.55,
                textinfo="label+percent",
                textfont_size=12,
                hovertemplate="%{label}: %{value} runs (%{percent})<extra></extra>",
            ))
            fig.update_layout(
                title="Run Outcome Classification",
                showlegend=False,
                annotations=[dict(text=f"{sum(values)}<br>runs", x=0.5, y=0.5,
                                  font_size=14, showarrow=False, font_color="#0f172a")],
            )
            style_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True, key="rds-outcome-donut")
        else:
            st.info("No outcome data.")


def _render_run_table(completed: pd.DataFrame) -> None:
    display = {
        "prefix": "Run",
        "launch_ts": "Launched",
        "backend": "Backend",
        "n_simulations": "Sims",
        "dtype": "dtype",
        "total_runtime_seconds": "Runtime (s)",
        "expected_loss": "EL ($M)",
        "var_99": "VaR 99% ($M)",
        "loss_rate": "Loss Rate (%)",
        "improvement_label": "Outcome",
        "source": "Source",
    }
    tbl = completed[[c for c in display if c in completed.columns]].copy()
    if "expected_loss" in tbl.columns:
        tbl["expected_loss"] = (tbl["expected_loss"] / 1e6).round(1)
    if "var_99" in tbl.columns:
        tbl["var_99"] = (tbl["var_99"] / 1e6).round(1)
    if "loss_rate" in tbl.columns:
        tbl["loss_rate"] = (tbl["loss_rate"] * 100).round(3)
    if "total_runtime_seconds" in tbl.columns:
        tbl["total_runtime_seconds"] = tbl["total_runtime_seconds"].round(1)
    tbl.columns = [display.get(c, c) for c in tbl.columns]
    st.dataframe(tbl, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Compact run history
# ---------------------------------------------------------------------------

_STATUS_ICON = {"completed": "✓", "running": "⏳", "failed": "✗"}
_STATUS_COLOR = {"completed": "#15803d", "running": "#1e40af", "failed": "#b91c1c"}
_STATUS_BG = {"completed": "#f0fdf4", "running": "#eff6ff", "failed": "#fef2f2"}
_STATUS_BORDER = {"completed": "#86efac", "running": "#93c5fd", "failed": "#fca5a5"}


def _render_compact_run_card(run: RunInfo, risk_df: pd.DataFrame | None, summary_df: pd.DataFrame | None) -> None:
    """Single compact card: key fields visible, all detail inside expander."""
    legacy = run.config.get("_legacy", False)
    cfg = run.config or {}

    # Pull key metrics for the summary line
    el_str = "—"
    rt_str = "—"
    sims_str = str(cfg.get("n_simulations") or "—")
    be_str = str(cfg.get("backend") or "—")

    if risk_df is not None and not risk_df.empty:
        r = risk_df.iloc[0]
        el = r.get("expected_loss")
        if el is not None and pd.notna(el):
            el_str = f"${float(el)/1e6:,.0f}M"

    if summary_df is not None and not summary_df.empty:
        rt = summary_df.iloc[0].get("total_runtime_seconds")
        if rt is not None and pd.notna(rt):
            rt_str = f"{float(rt):.0f}s"
        nsims = summary_df.iloc[0].get("n_simulations")
        if nsims is not None and pd.notna(nsims):
            sims_str = f"{int(nsims):,}"

    status = run.status
    icon = _STATUS_ICON.get(status, "?")
    bg = _STATUS_BG.get(status, "#f8fafc")
    border = _STATUS_BORDER.get(status, "#e2e8f0")
    color = _STATUS_COLOR.get(status, "#334155")
    src_badge = f"[{run.source.upper()}]"
    legacy_note = " · legacy" if legacy else ""

    label_html = (
        f'<span style="color:{color};font-weight:700;">{icon}</span> '
        f'<span style="font-family:monospace;font-size:0.82rem;">{run.prefix}</span> '
        f'<span style="color:#94a3b8;font-size:0.75rem;">{src_badge}{legacy_note}</span>'
        f' &nbsp;·&nbsp; <span style="font-size:0.82rem;color:#475569;">{be_str} · {sims_str} sims · EL {el_str} · {rt_str}</span>'
    )

    with st.expander(
        f"{icon} {run.prefix}  {src_badge}{legacy_note}  ·  {be_str}  ·  {sims_str} sims  ·  EL {el_str}  ·  {rt_str}",
        expanded=(status == "running"),
    ):
        if legacy:
            st.caption("⚠ Legacy prefix — may have been overwritten by a subsequent CLI run.")

        head_l, head_r = st.columns([3, 1])
        with head_r:
            _status_badge(status)

        if status == "running":
            tail = read_log_tail(run.log_path or "", n=30)
            st.code(tail or "(log initializing…)", language=None)
            st.caption("In progress — page auto-refreshes every ~3 seconds.")
            return

        if status == "completed":
            tab_m, tab_c, tab_cfg = st.tabs(["Metrics", "Charts", "Config"])
            with tab_m:
                _render_step_metrics(run)
            with tab_c:
                _render_run_charts(run)
            with tab_cfg:
                _render_config_table(run)
            _render_raw_log(run)
            return

        # failed
        if run.exit_code == -1:
            st.error("Run terminated — process gone before completion (or server restart).")
        else:
            st.error(f"Run exited with code {run.exit_code}.")
        _render_step_metrics(run)
        _render_raw_log(run)


def _render_run_history(all_runs: list[RunInfo]) -> None:
    if not all_runs:
        st.info(
            "No runs yet. Use the launch form above, or run "
            "`src/run_monte_carlo_custom_backend.py` from the CLI."
        )
        return

    running = [r for r in all_runs if r.status == "running"]
    completed = [r for r in all_runs if r.status == "completed"]
    failed = [r for r in all_runs if r.status == "failed"]

    for group_label, group_runs in [
        ("⏳ Active", running),
        ("✓ Completed", completed),
        ("✗ Failed", failed),
    ]:
        if not group_runs:
            continue
        st.markdown(
            f'<div style="font-size:0.85rem;font-weight:600;color:#475569;'
            f'margin:14px 0 6px;">{group_label} ({len(group_runs)})</div>',
            unsafe_allow_html=True,
        )
        for run in group_runs:
            risk_path = str(MODEL_DIR / f"{run.prefix}_risk_metrics.csv")
            summary_path = str(MODEL_DIR / f"{run.prefix}_runtime_summary.csv")
            risk_df = load_run_risk(risk_path)
            summary_df = load_run_summary(summary_path)
            _render_compact_run_card(run, risk_df, summary_df)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)
    st.title("Simulation Runs · Analytics")

    st.session_state.setdefault("procs", {})
    st.session_state.setdefault("active_prefixes", [])

    all_runs = discover_runs()

    if "_just_submitted" in st.session_state:
        submitted_prefix = st.session_state.pop("_just_submitted")
        st.success(f"✓ Simulation submitted: **{submitted_prefix}**")

    # Build analytics DF once so status strip + dashboard share the same data
    analytics_df = build_run_analytics_df(all_runs) if all_runs else None

    # ── Status strip ────────────────────────────────────────────────────────
    _render_status_strip(all_runs, analytics_df)

    # ── Launch + live status (two-column zone) ──────────────────────────────
    launch_col, live_col = st.columns([5, 4], gap="large")

    with launch_col:
        _render_launch_form(all_runs)

    with live_col:
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:600;color:#1e40af;'
            'padding-bottom:6px;border-bottom:2px solid #dbeafe;margin-bottom:12px;">'
            'Active Runs</div>',
            unsafe_allow_html=True,
        )
        _render_active_run_panel(all_runs)

    # ── Cross-run analytics ─────────────────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#1e40af;font-size:1.15rem;font-weight:600;'
        'margin-bottom:8px;">Cross-Run Analytics</div>',
        unsafe_allow_html=True,
    )

    if analytics_df is not None:
        _render_analytics_dashboard(analytics_df)
    else:
        st.info("Complete at least one run to unlock cross-run analytics.")

    # ── Run history ─────────────────────────────────────────────────────────
    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#1e40af;font-size:1.15rem;font-weight:600;'
        'margin-bottom:8px;">Run History</div>',
        unsafe_allow_html=True,
    )
    _render_run_history(all_runs)

    # ── Polling ─────────────────────────────────────────────────────────────
    _poll_active_runs(all_runs)
