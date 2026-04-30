"""
Simulation Runs Screen
Lets analysts launch run_monte_carlo_custom_backend.py, watch live logs,
and browse all completed runs (UI-launched and CLI-launched).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import COLORS, info_box, section_header, style_chart, warning_box
from run_state import MODEL_DIR, RunInfo, discover_runs, is_alive, is_run_complete, read_state, update_state, write_state
from dataset_helpers import (
    get_dataset_presets,
    get_portfolio_meta as _get_portfolio_meta,
    portfolio_display_label as _portfolio_display_label,
)
from portfolio_cache import dataset_fingerprint as _dataset_fingerprint
from model_bundle import (
    find_compatible_bundles as _find_compatible_bundles,
    latest_mc_ready_bundle as _latest_mc_ready_bundle,
    legacy_is_mc_ready as _legacy_is_mc_ready,
    legacy_bundle_info as _legacy_bundle_info,
)

REPO_ROOT = Path(__file__).parent.parent.parent
RUNNER_SCRIPT = REPO_ROOT / "src" / "run_monte_carlo_custom_backend.py"

STEP_NAME_MAP = {
    "baseline_scoring_seconds": "Step 1: Baseline scoring",
    "loan_specific_sensitivity_seconds": "Step 2: Loan sensitivities",
    "scored_portfolio_write_seconds": "Step 2b: Write portfolio",
    "macro_stats_seconds": "Step 3: Macro stats",
    "monte_carlo_seconds": "Step 4: Monte Carlo",
    "risk_metrics_seconds": "Step 5: Risk metrics",
    "sensitivity_seconds": "Step 6: Sensitivity",
    "save_seconds": "Step 7: Save results",
}

RUN_INDEX_PATH = MODEL_DIR / "simulation_run_index.csv"

# Relative-change tolerance for improvement/deterioration classification
_IMPROVEMENT_TOLERANCE = 0.05  # 5%

_INDEX_COLUMNS = [
    "prefix", "status", "source", "launch_ts",
    "backend", "n_simulations", "dtype", "execution_mode",
    "cpu_workers", "torch_threads_per_worker", "pyarrow_threads",
    "scenario_batch_size", "loan_chunk_size",
    "expected_loss", "var_99", "var_999", "es_99", "total_balance",
    "total_runtime_seconds",
    "baseline_scoring_seconds", "loan_specific_sensitivity_seconds",
    "monte_carlo_seconds", "sensitivity_seconds",
    "loss_rate", "improvement_label",
    "portfolio_path", "portfolio_label",
]

# ---------------------------------------------------------------------------
# Cached data loaders (keyed on path so each prefix gets its own cache entry)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_run_summary(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_run_risk(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_run_dist(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_run_sensitivity(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(ttl=300)
def load_run_scenarios(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


def _get_loss_series(df: pd.DataFrame) -> pd.Series | None:
    for column in ("loss", "portfolio_loss"):
        if column in df.columns:
            return df[column]
    return None


# ---------------------------------------------------------------------------
# Cross-run analytics helpers
# ---------------------------------------------------------------------------

def _build_run_record(run: RunInfo) -> dict | None:
    """Assemble a single analytics row from a run's artifacts. Returns None if data unavailable."""
    if run.status != "completed":
        return None

    risk_path = MODEL_DIR / f"{run.prefix}_risk_metrics.csv"
    summary_path = MODEL_DIR / f"{run.prefix}_runtime_summary.csv"

    risk_row: dict = {}
    summary_row: dict = {}

    try:
        if risk_path.exists():
            risk_row = pd.read_csv(risk_path).iloc[0].to_dict()
    except Exception:
        pass

    try:
        if summary_path.exists():
            summary_row = pd.read_csv(summary_path).iloc[0].to_dict()
    except Exception:
        pass

    if not risk_row and not summary_row:
        return None

    cfg = run.config or {}

    def _get(d: dict, key: str):
        v = d.get(key)
        return None if (v is None or (isinstance(v, float) and pd.isna(v))) else v

    record: dict = {
        "prefix": run.prefix,
        "status": run.status,
        "source": run.source,
        "launch_ts": run.launch_ts,
        "backend": _get(cfg, "backend") or _get(risk_row, "backend") or _get(summary_row, "backend"),
        "n_simulations": _get(cfg, "n_simulations") or _get(risk_row, "n_simulations") or _get(summary_row, "n_simulations"),
        "dtype": _get(cfg, "dtype") or _get(risk_row, "dtype") or _get(summary_row, "dtype"),
        "execution_mode": _get(cfg, "execution_mode") or _get(risk_row, "execution_mode") or _get(summary_row, "execution_mode"),
        "cpu_workers": _get(cfg, "cpu_workers") or _get(risk_row, "cpu_workers") or _get(summary_row, "cpu_workers"),
        "torch_threads_per_worker": _get(cfg, "torch_threads_per_worker") or _get(risk_row, "torch_threads_per_worker") or _get(summary_row, "torch_threads_per_worker"),
        "pyarrow_threads": _get(cfg, "pyarrow_threads") or _get(risk_row, "pyarrow_threads") or _get(summary_row, "pyarrow_threads"),
        "scenario_batch_size": _get(cfg, "scenario_batch_size") or _get(risk_row, "scenario_batch_size"),
        "loan_chunk_size": _get(cfg, "loan_chunk_size") or _get(risk_row, "loan_chunk_size"),
        "expected_loss": _get(risk_row, "expected_loss"),
        "var_99": _get(risk_row, "var_99"),
        "var_999": _get(risk_row, "var_999"),
        "es_99": _get(risk_row, "es_99"),
        "total_balance": _get(risk_row, "total_balance"),
        "total_runtime_seconds": _get(summary_row, "total_runtime_seconds"),
        "baseline_scoring_seconds": _get(summary_row, "baseline_scoring_seconds"),
        "loan_specific_sensitivity_seconds": _get(summary_row, "loan_specific_sensitivity_seconds"),
        "monte_carlo_seconds": _get(summary_row, "monte_carlo_seconds"),
        "sensitivity_seconds": _get(summary_row, "sensitivity_seconds"),
        "loss_rate": None,
        "improvement_label": "unknown",
        "portfolio_path": _get(cfg, "portfolio_path"),
        "portfolio_label": _get(cfg, "portfolio_label"),
    }

    el = record["expected_loss"]
    tb = record["total_balance"]
    if el is not None and tb is not None and tb > 0:
        record["loss_rate"] = el / tb

    return record


def _classify_improvement(df: pd.DataFrame) -> pd.DataFrame:
    """Add improvement_label column based on loss_rate vs median baseline.

    Baseline = median loss_rate across all completed runs with available data.
    Labels:
      improved     — loss_rate < baseline × (1 - tolerance)
      deteriorated — loss_rate > baseline × (1 + tolerance)
      neutral      — within ±5% of baseline
      baseline     — only one run (no comparison possible)
      unknown      — loss_rate not available
    """
    df = df.copy()
    valid = df["loss_rate"].dropna()
    if valid.empty:
        return df

    if len(valid) == 1:
        df.loc[df["loss_rate"].notna(), "improvement_label"] = "baseline"
        return df

    baseline_median = valid.median()
    lo = baseline_median * (1 - _IMPROVEMENT_TOLERANCE)
    hi = baseline_median * (1 + _IMPROVEMENT_TOLERANCE)

    def _label(lr):
        if pd.isna(lr):
            return "unknown"
        if lr < lo:
            return "improved"
        if lr > hi:
            return "deteriorated"
        return "neutral"

    df["improvement_label"] = df["loss_rate"].map(_label)
    return df


def build_run_analytics_df(all_runs: list[RunInfo]) -> pd.DataFrame | None:
    """Build a cross-run analytics DataFrame from completed runs.

    Reads from existing artifacts, merges with any previously persisted rows
    for prefixes that are no longer in all_runs (historical), classifies
    improvement/deterioration, persists to simulation_run_index.csv, and
    returns the combined DataFrame.
    """
    records = []
    seen_prefixes: set = set()

    for run in all_runs:
        rec = _build_run_record(run)
        if rec is not None:
            records.append(rec)
            seen_prefixes.add(run.prefix)

    # Load previously persisted rows for historical runs not in current all_runs
    if RUN_INDEX_PATH.exists():
        try:
            old = pd.read_csv(RUN_INDEX_PATH)
            for _, row in old.iterrows():
                if row.get("prefix") not in seen_prefixes:
                    records.append(row.to_dict())
        except Exception:
            pass

    if not records:
        return None

    df = pd.DataFrame(records)
    for col in _INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[_INDEX_COLUMNS]
    df = _classify_improvement(df)

    # Sort newest first before saving
    df = df.sort_values("launch_ts", ascending=False, na_position="last")

    try:
        df.to_csv(RUN_INDEX_PATH, index=False)
    except Exception:
        pass

    return df


# ---------------------------------------------------------------------------
# Cross-run charts and comparison table
# ---------------------------------------------------------------------------

_LABEL_COLORS = {
    "improved": "#22c55e",
    "neutral": "#94a3b8",
    "deteriorated": "#ef4444",
    "baseline": "#3b82f6",
    "unknown": "#d1d5db",
}


# ---------------------------------------------------------------------------
# Forecast quality vs simulation count
# ---------------------------------------------------------------------------

def _compute_convergence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add reference-error and efficiency columns to a completed-runs DataFrame.

    Reference = row with the highest n_simulations within each (backend, dtype) group.
    Groups with only one row get zero error (the row IS the reference).

    Added columns:
      el_pct_error, var99_pct_error, var999_pct_error, es99_pct_error,
      runtime_per_sim, error_per_second
    """
    df = df.copy()
    _NEW = [
        "el_pct_error", "var99_pct_error", "var999_pct_error", "es99_pct_error",
        "runtime_per_sim", "error_per_second",
    ]
    for c in _NEW:
        df[c] = float("nan")

    df["n_simulations"] = pd.to_numeric(df["n_simulations"], errors="coerce")

    rt = pd.to_numeric(df.get("total_runtime_seconds"), errors="coerce")
    ns = df["n_simulations"]
    valid_rt = rt.notna() & ns.notna() & (ns > 0)
    df.loc[valid_rt, "runtime_per_sim"] = rt[valid_rt] / ns[valid_rt]

    if df.empty:
        return df

    df["_be"] = df["backend"].fillna("unknown").astype(str)
    df["_dt"] = df["dtype"].fillna("unknown").astype(str)

    metric_map = {
        "expected_loss": "el_pct_error",
        "var_99": "var99_pct_error",
        "var_999": "var999_pct_error",
        "es_99": "es99_pct_error",
    }

    for (_be, _dt), grp in df.groupby(["_be", "_dt"]):
        ns_grp = pd.to_numeric(grp["n_simulations"], errors="coerce")
        if ns_grp.isna().all():
            continue
        ref_idx = ns_grp.idxmax()
        ref = grp.loc[ref_idx]
        for idx in grp.index:
            row = grp.loc[idx]
            for metric, err_col in metric_map.items():
                try:
                    rv = float(ref[metric])
                    cv = float(row[metric])
                    if not (pd.isna(rv) or pd.isna(cv)) and rv != 0:
                        df.at[idx, err_col] = abs(cv - rv) / abs(rv) * 100.0
                except (TypeError, ValueError):
                    pass

    ep = df["el_pct_error"]
    rt2 = pd.to_numeric(df.get("total_runtime_seconds"), errors="coerce")
    valid_eff = ep.notna() & rt2.notna() & (rt2 > 0)
    df.loc[valid_eff, "error_per_second"] = ep[valid_eff] / rt2[valid_eff]

    df = df.drop(columns=["_be", "_dt"])
    return df


def _render_sim_count_recommendation(df: pd.DataFrame) -> None:
    """Narrative interpretation box: sweet spot, tail convergence, diminishing returns."""
    el_err = df["el_pct_error"].dropna()
    v99_err = df["var99_pct_error"].dropna()
    has_errors = not el_err.empty and not v99_err.empty

    st.markdown("#### Interpretation")
    cols = st.columns(3)

    # Sweet spot
    sweet_n = None
    if has_errors:
        cand = df.dropna(subset=["el_pct_error", "var99_pct_error", "n_simulations"])
        cand = cand.sort_values("n_simulations")
        good = cand[(cand["el_pct_error"] <= 1.0) & (cand["var99_pct_error"] <= 2.0)]
        if not good.empty:
            sweet_n = int(good["n_simulations"].min())

    if sweet_n is not None:
        cols[0].success(
            f"**Sweet spot: {sweet_n:,} sims**\n\n"
            "EL error ≤ 1% and VaR 99 error ≤ 2% vs the highest-sim reference run."
        )
    elif has_errors:
        cols[0].warning(
            f"**No sweet spot found**\n\n"
            f"Best EL error: {el_err.min():.1f}%. "
            "Add runs with more simulations to find the convergence threshold."
        )
    else:
        cols[0].info(
            "**Sweet spot: insufficient data**\n\n"
            "Need ≥ 2 completed runs with different n_simulations in the same backend/dtype group."
        )

    # Tail convergence
    if has_errors and "var999_pct_error" in df.columns and df["var999_pct_error"].notna().any():
        avg_el = el_err.mean()
        avg_v999 = df["var999_pct_error"].dropna().mean()
        if avg_v999 > avg_el * 1.5:
            cols[1].warning(
                f"**Tail converges slower**\n\n"
                f"VaR 99.9% avg error ({avg_v999:.1f}%) exceeds EL avg error ({avg_el:.1f}%). "
                "Tail metrics need more simulations to stabilize."
            )
        else:
            cols[1].success(
                f"**Tail tracks EL closely**\n\n"
                f"VaR 99.9% error ({avg_v999:.1f}%) ≈ EL error ({avg_el:.1f}%)."
            )
    else:
        cols[1].info(
            "**Tail convergence:** insufficient data\n\n"
            "Need multiple sim counts to compare tail vs EL convergence speed."
        )

    # Diminishing returns
    cand_dr = df.dropna(subset=["n_simulations", "el_pct_error"]).sort_values("n_simulations")
    if len(cand_dr) >= 3:
        ns_arr = cand_dr["n_simulations"].values
        err_arr = cand_dr["el_pct_error"].values
        dr_n = None
        for i in range(1, len(ns_arr)):
            if ns_arr[i] > ns_arr[i - 1] > 0:
                reduction = err_arr[i - 1] - err_arr[i]
                ratio = ns_arr[i] / ns_arr[i - 1]
                if reduction / max(err_arr[i - 1], 0.001) < 0.10 and ratio > 1.5:
                    dr_n = int(ns_arr[i])
                    break
        if dr_n:
            cols[2].info(
                f"**Diminishing returns ~{dr_n:,} sims**\n\n"
                "Error reduction < 10% despite ≥ 50% more simulations."
            )
        else:
            cols[2].info("**Diminishing returns:** not yet visible from available data.")
    else:
        cols[2].info(
            "**Diminishing returns:** need ≥ 3 distinct sim counts to identify the inflection point."
        )


def _chart_convergence(df: pd.DataFrame) -> None:
    """Chart 1: Risk metrics vs n_simulations."""
    metrics = [
        ("expected_loss", "Expected Loss", COLORS.get("primary", "#3b82f6")),
        ("var_99", "VaR 99%", COLORS.get("warning", "#f59e0b")),
        ("var_999", "VaR 99.9%", COLORS.get("danger", "#ef4444")),
        ("es_99", "ES 99%", COLORS.get("success", "#22c55e")),
    ]
    plot_df = df.dropna(subset=["n_simulations"]).copy()
    plot_df["n_simulations"] = pd.to_numeric(plot_df["n_simulations"], errors="coerce")
    plot_df = plot_df.dropna(subset=["n_simulations"])

    if plot_df.empty:
        st.info("Not enough data for convergence chart.")
        return

    has_any = any(col in plot_df.columns and plot_df[col].notna().any() for col, *_ in metrics)
    if not has_any:
        st.info("Risk metric columns not available.")
        return

    fig = go.Figure()
    for col, label, color in metrics:
        sub = plot_df.dropna(subset=[col]).sort_values("n_simulations")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["n_simulations"],
            y=sub[col] / 1e6,
            mode="lines+markers",
            name=label,
            line=dict(color=color),
            marker=dict(size=8),
            hovertemplate=f"{label}: $%{{y:,.0f}}M at %{{x:,}} sims<extra></extra>",
        ))

    ns_vals = plot_df["n_simulations"].dropna()
    xtype = "log" if (ns_vals.max() / max(ns_vals.min(), 1)) >= 10 else "linear"
    fig.update_layout(
        title="Risk Metric Convergence",
        xaxis_title="N Simulations" + (" (log)" if xtype == "log" else ""),
        yaxis_title="Metric ($M)",
        xaxis_type=xtype,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    style_chart(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key="sq-convergence")

    if len(ns_vals.unique()) == 1:
        st.caption(
            "All runs use the same simulation count — launch runs with different n_simulations "
            "to observe convergence."
        )


def _chart_error_to_reference(df: pd.DataFrame) -> None:
    """Chart 2: Percent error vs n_simulations (selectable metric)."""
    err_opts = {
        "el_pct_error": "EL % Error",
        "var99_pct_error": "VaR 99% Error",
        "var999_pct_error": "VaR 99.9% Error",
        "es99_pct_error": "ES 99% Error",
    }
    available = {k: v for k, v in err_opts.items()
                 if k in df.columns and df[k].notna().any()}

    if not available:
        st.info(
            "Error-to-reference chart requires ≥ 2 completed runs in the same backend/dtype "
            "group at different simulation counts."
        )
        return

    sel = st.selectbox(
        "Error metric",
        options=list(available.keys()),
        format_func=lambda k: available[k],
        key="sq-err-metric-sel",
    )

    plot_df = df.dropna(subset=["n_simulations", sel]).copy()
    plot_df["n_simulations"] = pd.to_numeric(plot_df["n_simulations"], errors="coerce")
    plot_df = plot_df.dropna(subset=["n_simulations"]).sort_values("n_simulations")

    if plot_df.empty:
        st.info("No data for the selected error metric.")
        return

    palette = ["#3b82f6", "#f59e0b", "#22c55e", "#ef4444", "#8b5cf6"]
    fig = go.Figure()
    for i, be in enumerate(plot_df["backend"].fillna("unknown").unique()):
        sub = plot_df[plot_df["backend"].fillna("unknown") == be]
        fig.add_trace(go.Scatter(
            x=sub["n_simulations"],
            y=sub[sel],
            mode="lines+markers",
            name=str(be),
            line=dict(color=palette[i % len(palette)]),
            marker=dict(size=9),
            text=sub["prefix"].str[-10:],
            hovertemplate=f"{be} — %{{text}}<br>Sims: %{{x:,}}<br>Error: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="#22c55e",
                  annotation_text="1% threshold", annotation_position="right")
    fig.add_hline(y=2.0, line_dash="dot", line_color="#f59e0b",
                  annotation_text="2% threshold", annotation_position="right")

    ns_vals = plot_df["n_simulations"].dropna()
    xtype = "log" if (ns_vals.max() / max(ns_vals.min(), 1)) >= 10 else "linear"
    fig.update_layout(
        title=f"{available[sel]} vs N Simulations",
        xaxis_title="N Simulations" + (" (log)" if xtype == "log" else ""),
        yaxis_title="% Error vs Reference",
        xaxis_type=xtype,
    )
    style_chart(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key="sq-error-to-ref")
    st.caption(
        "Reference = highest-n_simulations run in the same backend/dtype group. "
        "Zero error = this run IS the reference."
    )


def _chart_runtime_vs_quality(df: pd.DataFrame) -> None:
    """Chart 3: Total runtime vs EL error (efficiency frontier view)."""
    plot_df = df.dropna(subset=["total_runtime_seconds", "el_pct_error"]).copy()
    plot_df["total_runtime_seconds"] = pd.to_numeric(
        plot_df["total_runtime_seconds"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["total_runtime_seconds"])

    if plot_df.empty:
        st.info(
            "Runtime vs quality chart requires runs with both timing data and error vs reference. "
            "Need ≥ 2 runs in the same backend/dtype group."
        )
        return

    ns_series = pd.to_numeric(plot_df["n_simulations"], errors="coerce").fillna(1000)
    ns_max = ns_series.max()
    marker_sizes = ((ns_series / max(ns_max, 1)) * 20 + 6).clip(6, 26)

    palette = ["#3b82f6", "#f59e0b", "#22c55e", "#ef4444", "#8b5cf6"]
    fig = go.Figure()
    for i, be in enumerate(plot_df["backend"].fillna("unknown").unique()):
        mask = plot_df["backend"].fillna("unknown") == be
        sub = plot_df[mask]
        fig.add_trace(go.Scatter(
            x=sub["total_runtime_seconds"],
            y=sub["el_pct_error"],
            mode="markers",
            name=str(be),
            marker=dict(
                size=marker_sizes[mask].tolist(),
                color=palette[i % len(palette)],
                opacity=0.8,
            ),
            text=sub.apply(
                lambda r: (
                    f"{str(r.get('prefix', ''))[-10:]} ({int(r['n_simulations']):,} sims)"
                    if pd.notna(r.get("n_simulations")) else str(r.get("prefix", ""))[-10:]
                ),
                axis=1,
            ),
            hovertemplate="%{text}<br>Runtime: %{x:.1f}s<br>EL Error: %{y:.2f}%<extra></extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="#22c55e",
                  annotation_text="1% quality threshold")
    fig.update_layout(
        title="Runtime vs Forecast Quality (EL error)",
        xaxis_title="Total Runtime (s)",
        yaxis_title="EL % Error vs Reference",
    )
    style_chart(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key="sq-runtime-vs-quality")
    st.caption("Marker size scales with n_simulations. Bottom-left corner = best efficiency.")


def _chart_stability(df: pd.DataFrame) -> None:
    """Chart 4: EL spread across repeated runs at same sim count + config."""
    work = df.copy()
    work["n_simulations"] = pd.to_numeric(work["n_simulations"], errors="coerce")
    work["_be"] = work["backend"].fillna("unknown").astype(str)
    work["_dt"] = work["dtype"].fillna("unknown").astype(str)
    work["expected_loss"] = pd.to_numeric(work.get("expected_loss"), errors="coerce")

    grp = work.dropna(subset=["n_simulations", "expected_loss"]).groupby(
        ["_be", "_dt", "n_simulations"]
    )["expected_loss"]
    counts = grp.count()

    if (counts > 1).sum() == 0:
        st.info(
            "Stability analysis requires repeated runs at the same simulation count and backend/dtype. "
            "No such duplicates found in current data.\n\n"
            "Launch multiple runs with identical configuration to observe variance bands."
        )
        return

    stats = grp.agg(["mean", "std", "min", "max", "count"]).reset_index()
    stats = stats[stats["count"] > 1].copy()
    stats["cv"] = (stats["std"] / stats["mean"].abs()).fillna(0) * 100
    stats["label"] = (
        stats["_be"] + "/" + stats["_dt"]
        + " @ " + stats["n_simulations"].astype(int).astype(str) + " sims"
    )

    fig = go.Figure()
    for _, row in stats.iterrows():
        mean_m = row["mean"] / 1e6
        std_m = float(row["std"]) / 1e6 if pd.notna(row["std"]) else 0.0
        fig.add_trace(go.Bar(
            x=[row["label"]],
            y=[mean_m],
            error_y=dict(type="data", array=[std_m], visible=True),
            name=str(row["label"]),
            text=[f"CV: {row['cv']:.1f}% (n={int(row['count'])})"],
            textposition="outside",
            marker_color=COLORS.get("primary", "#3b82f6"),
        ))

    fig.update_layout(
        title="EL Stability Across Repeated Runs",
        xaxis_title="Config Group",
        yaxis_title="Expected Loss ($M)",
        showlegend=False,
    )
    style_chart(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key="sq-stability")
    st.caption("Error bars = ±1 std dev. CV = coefficient of variation. Lower CV means more stable output.")


def _chart_step_scaling(df: pd.DataFrame) -> None:
    """Chart 5: Selected step runtime vs n_simulations."""
    step_opts = {
        "monte_carlo_seconds": "Monte Carlo",
        "baseline_scoring_seconds": "Baseline Scoring",
        "loan_specific_sensitivity_seconds": "Loan Sensitivities",
        "sensitivity_seconds": "Sensitivity",
        "total_runtime_seconds": "Total Runtime",
    }
    available = {k: v for k, v in step_opts.items()
                 if k in df.columns and df[k].notna().any()}

    if not available:
        st.info("Step scaling chart requires runtime_summary.csv data. No step timing available yet.")
        return

    sel = st.selectbox(
        "Step",
        options=list(available.keys()),
        format_func=lambda k: available[k],
        key="sq-step-sel",
    )

    plot_df = df.dropna(subset=["n_simulations", sel]).copy()
    plot_df["n_simulations"] = pd.to_numeric(plot_df["n_simulations"], errors="coerce")
    plot_df = plot_df.dropna(subset=["n_simulations"]).sort_values("n_simulations")

    if plot_df.empty:
        st.info("No data for the selected step.")
        return

    palette = ["#3b82f6", "#f59e0b", "#22c55e", "#ef4444"]
    fig = go.Figure()
    for i, be in enumerate(plot_df["backend"].fillna("unknown").unique()):
        sub = plot_df[plot_df["backend"].fillna("unknown") == be]
        fig.add_trace(go.Scatter(
            x=sub["n_simulations"],
            y=pd.to_numeric(sub[sel], errors="coerce"),
            mode="lines+markers",
            name=str(be),
            line=dict(color=palette[i % len(palette)]),
            marker=dict(size=9),
            hovertemplate=f"{be}<br>Sims: %{{x:,}}<br>Time: %{{y:.1f}}s<extra></extra>",
        ))

    ns_vals = plot_df["n_simulations"].dropna()
    xtype = "log" if (ns_vals.max() / max(ns_vals.min(), 1)) >= 10 else "linear"
    fig.update_layout(
        title=f"{available[sel]} Scaling",
        xaxis_title="N Simulations" + (" (log)" if xtype == "log" else ""),
        yaxis_title="Duration (s)",
        xaxis_type=xtype,
    )
    style_chart(fig, 380)
    st.plotly_chart(fig, use_container_width=True, key="sq-step-scaling")


def _render_sim_count_analytics(df: pd.DataFrame) -> None:
    """Render the 'Forecast Quality vs Simulation Count' analytics section."""
    work = df.copy()
    work["n_simulations"] = pd.to_numeric(work["n_simulations"], errors="coerce")

    if not work["n_simulations"].notna().any():
        st.info(
            "Forecast quality analytics require n_simulations data. "
            "No completed runs with simulation count info found."
        )
        return

    work = _compute_convergence_metrics(work)

    _render_sim_count_recommendation(work)
    st.markdown("---")

    tab_conv, tab_err, tab_rt, tab_stab, tab_step = st.tabs([
        "Convergence", "Error to Reference", "Runtime vs Quality",
        "Stability", "Step Scaling",
    ])
    with tab_conv:
        _chart_convergence(work)
    with tab_err:
        _chart_error_to_reference(work)
    with tab_rt:
        _chart_runtime_vs_quality(work)
    with tab_stab:
        _chart_stability(work)
    with tab_step:
        _chart_step_scaling(work)


def _render_cross_run_analytics(df: pd.DataFrame) -> None:
    """Render the cross-run analytics section: summary KPIs, charts, table."""
    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        st.info("Cross-run analytics available once at least one run completes.")
        return

    section_header("Cross-Run Analytics")
    st.caption(
        "Baseline = median loss_rate across all completed runs. "
        "Improved = loss_rate < baseline − 5%. Deteriorated = loss_rate > baseline + 5%."
    )

    # --- Summary KPIs ---
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Completed runs", len(completed))
    el_series = completed["expected_loss"].dropna()
    if not el_series.empty:
        best_el = el_series.min()
        kpi_cols[1].metric("Best EL", f"${best_el/1e6:,.0f}M")
        kpi_cols[2].metric("Latest EL", f"${el_series.iloc[0]/1e6:,.0f}M")
    rt_series = completed["total_runtime_seconds"].dropna()
    if not rt_series.empty:
        kpi_cols[3].metric("Avg runtime", f"{rt_series.mean():.0f}s")

    tab_charts, tab_forecast, tab_table = st.tabs(["Charts", "Forecast Quality", "Comparison Table"])

    with tab_forecast:
        _render_sim_count_analytics(completed)

    with tab_charts:
        # Four charts in a 2×2 grid
        row1_left, row1_right = st.columns(2)
        row2_left, row2_right = st.columns(2)

        # Chart 1: n_simulations vs expected_loss, colored by backend
        with row1_left:
            plot_df = completed.dropna(subset=["n_simulations", "expected_loss"])
            if not plot_df.empty:
                backends = plot_df["backend"].fillna("unknown").unique()
                fig = go.Figure()
                for be in backends:
                    sub = plot_df[plot_df["backend"].fillna("unknown") == be]
                    fig.add_trace(go.Scatter(
                        x=sub["n_simulations"], y=sub["expected_loss"] / 1e6,
                        mode="markers+text",
                        text=sub["prefix"].str[-6:],
                        textposition="top center",
                        textfont=dict(size=9),
                        marker=dict(size=10),
                        name=str(be),
                    ))
                fig.update_layout(title="Simulations vs Expected Loss",
                                  xaxis_title="N Simulations", yaxis_title="Expected Loss ($M)")
                style_chart(fig, 320)
                st.plotly_chart(fig, use_container_width=True, key="xr-sims-vs-el")
            else:
                st.info("Not enough data for this chart.")

        # Chart 2: n_simulations vs total runtime, colored by dtype
        with row1_right:
            plot_df = completed.dropna(subset=["n_simulations", "total_runtime_seconds"])
            if not plot_df.empty:
                dtypes = plot_df["dtype"].fillna("unknown").unique()
                fig = go.Figure()
                for dt in dtypes:
                    sub = plot_df[plot_df["dtype"].fillna("unknown") == dt]
                    fig.add_trace(go.Scatter(
                        x=sub["n_simulations"], y=sub["total_runtime_seconds"],
                        mode="markers+text",
                        text=sub["prefix"].str[-6:],
                        textposition="top center",
                        textfont=dict(size=9),
                        marker=dict(size=10),
                        name=str(dt),
                    ))
                fig.update_layout(title="Simulations vs Total Runtime",
                                  xaxis_title="N Simulations", yaxis_title="Runtime (s)")
                style_chart(fig, 320)
                st.plotly_chart(fig, use_container_width=True, key="xr-sims-vs-rt")
            else:
                st.info("Not enough data for this chart.")

        # Chart 3: backend vs avg expected_loss (bar)
        with row2_left:
            plot_df = completed.dropna(subset=["backend", "expected_loss"])
            if not plot_df.empty:
                grp = plot_df.groupby("backend")["expected_loss"].mean().reset_index()
                grp["expected_loss_m"] = grp["expected_loss"] / 1e6
                fig = go.Figure(go.Bar(
                    x=grp["backend"], y=grp["expected_loss_m"],
                    marker_color=COLORS["primary"], text=grp["expected_loss_m"].map(lambda v: f"${v:,.0f}M"),
                    textposition="outside",
                ))
                fig.update_layout(title="Avg Expected Loss by Backend",
                                  xaxis_title="Backend", yaxis_title="Avg EL ($M)")
                style_chart(fig, 320)
                st.plotly_chart(fig, use_container_width=True, key="xr-backend-el")
            else:
                st.info("Not enough data for this chart.")

        # Chart 4: outcome classification count
        with row2_right:
            label_counts = completed["improvement_label"].value_counts().reset_index()
            label_counts.columns = ["label", "count"]
            colors = [_LABEL_COLORS.get(lbl, "#94a3b8") for lbl in label_counts["label"]]
            fig = go.Figure(go.Bar(
                x=label_counts["label"], y=label_counts["count"],
                marker_color=colors,
                text=label_counts["count"], textposition="outside",
            ))
            fig.update_layout(title="Run Outcome Classification",
                              xaxis_title="Outcome", yaxis_title="Count")
            style_chart(fig, 320)
            st.plotly_chart(fig, use_container_width=True, key="xr-outcome-class")

    with tab_table:
        display_cols = {
            "prefix": "Run",
            "launch_ts": "Launched",
            "backend": "Backend",
            "n_simulations": "N Sims",
            "dtype": "dtype",
            "cpu_workers": "CPU Workers",
            "total_runtime_seconds": "Runtime (s)",
            "expected_loss": "Expected Loss ($M)",
            "var_99": "VaR 99% ($M)",
            "loss_rate": "Loss Rate (%)",
            "improvement_label": "Outcome",
            "source": "Source",
        }
        tbl = completed[[c for c in display_cols if c in completed.columns]].copy()
        if "expected_loss" in tbl.columns:
            tbl["expected_loss"] = tbl["expected_loss"] / 1e6
        if "var_99" in tbl.columns:
            tbl["var_99"] = tbl["var_99"] / 1e6
        if "loss_rate" in tbl.columns:
            tbl["loss_rate"] = tbl["loss_rate"] * 100
        if "total_runtime_seconds" in tbl.columns:
            tbl["total_runtime_seconds"] = tbl["total_runtime_seconds"].round(1)
        tbl.columns = [display_cols.get(c, c) for c in tbl.columns]
        st.dataframe(tbl, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Log helpers (NOT cached — must reflect live state)
# ---------------------------------------------------------------------------

def read_log_tail(log_path: str, n: int = 50) -> str:
    p = Path(log_path)
    if not p.exists():
        return ""
    try:
        with open(p, "rb") as f:
            try:
                f.seek(-16384, 2)  # read at most 16 KB from end
            except OSError:
                f.seek(0)
            content = f.read().decode("utf-8", errors="replace")
        lines = content.splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def parse_log_metrics(log_path: str) -> dict:
    """Extract 5 portfolio/baseline fields from captured log using regex."""
    result = {}
    if not log_path or not Path(log_path).exists():
        return result
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return result

    patterns = {
        "n_loans": r"Portfolio:\s+([\d,]+) loans",
        "total_balance": r"Portfolio:.*?\$([\d.]+)B",
        "baseline_pd": r"Baseline PD:\s+([\d.]+)%",
        "baseline_lgd": r"Baseline LGD:\s+([\d.]+)%",
        "baseline_el_m": r"Baseline Annual EL:\s+\$([\d,]+)M",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                result[key] = float(raw)
            except ValueError:
                pass
    return result


# ---------------------------------------------------------------------------
# Launch form (U3)
# ---------------------------------------------------------------------------

def _render_launch_form(all_runs: list[RunInfo]) -> None:
    section_header("Launch New Simulation")
    active_backends = {
        r.config.get("backend") for r in all_runs if r.status == "running"
    }

    # Dataset selection lives OUTSIDE the form so conditional widgets render
    # immediately when the user changes the source radio (st.form batches all
    # widget events until submit, which prevents conditional re-rendering).
    with st.expander("Dataset (optional)"):
        ds_mode = st.radio(
            "Source",
            ["Default", "Preset", "Custom path"],
            horizontal=True,
            key="launch_ds_mode",
            help="Default = runner auto-detects the best available dataset.",
        )
        ds_preset_key = None
        ds_custom_path = ""
        if ds_mode == "Preset":
            _presets = get_dataset_presets()
            if _presets:
                ds_preset_key = st.selectbox(
                    "Dataset",
                    options=list(_presets.keys()),
                    format_func=lambda k: _presets[k]["label"],
                    key="launch_ds_preset",
                )
                st.caption(_presets[ds_preset_key]["description"])
            else:
                st.warning("No preset datasets found under `data/processed/`. Use Custom path.")
        elif ds_mode == "Custom path":
            ds_custom_path = st.text_input(
                "Portfolio path",
                value=st.session_state.get("_last_ds_custom_path", ""),
                placeholder="/absolute/path/to/dataset.parquet",
                key="launch_ds_custom",
            )

    # ── PD/LGD model bundle selector ────────────────────────────────────────
    # Resolve the dataset path (preview-only — same logic as submit handler)
    _preview_portfolio_path = None
    if ds_mode == "Preset" and ds_preset_key:
        _prev_presets = get_dataset_presets()
        _preview_portfolio_path = _prev_presets.get(ds_preset_key, {}).get("path")
    elif ds_mode == "Custom path" and ds_custom_path.strip():
        _preview_portfolio_path = ds_custom_path.strip()

    _ds_fp = _dataset_fingerprint(_preview_portfolio_path) if _preview_portfolio_path else None
    _mc_ready_bundle = _latest_mc_ready_bundle(_ds_fp) if _ds_fp else None
    _compatible = _find_compatible_bundles(_ds_fp) if _ds_fp else []
    _mc_ready_compatible = [b for b in _compatible if b["is_mc_ready"]]

    with st.expander("PD/LGD Model Bundle", expanded=True):
        # Build options list
        _bundle_options = []  # (display_label, pd_bundle_dir_str, lgd_bundle_dir_str)

        for b in _mc_ready_compatible:
            ts = b["timestamp"][:16] if b["timestamp"] else ""
            lbl = f"{b['bundle_id']} · {ts}" if ts else b["bundle_id"]
            _bundle_options.append((
                f"Bundle: {lbl}",
                str(b["bundle_dir"]),
                str(b["bundle_dir"]),
            ))

        if _legacy_is_mc_ready():
            from model_bundle import MODEL_DIR as _MD
            _bundle_options.append((
                "Legacy (global models/ files)",
                str(_MD),
                str(_MD),
            ))

        if not _bundle_options:
            if _ds_fp:
                st.warning(
                    "No compatible PD+LGD bundle found for the selected dataset. "
                    "Train PD and LGD models on this dataset first (PD Model / LGD Model pages), "
                    "then return here to run Monte Carlo.",
                    icon="⚠️",
                )
                _selected_pd_bundle_dir = None
                _selected_lgd_bundle_dir = None
            else:
                # Default dataset — check legacy
                if _legacy_is_mc_ready():
                    st.success("Using legacy global model artifacts from `models/`.")
                    from model_bundle import MODEL_DIR as _MD
                    _selected_pd_bundle_dir = str(_MD)
                    _selected_lgd_bundle_dir = str(_MD)
                else:
                    st.warning("No PD/LGD models found. Run PD and LGD training first.")
                    _selected_pd_bundle_dir = None
                    _selected_lgd_bundle_dir = None
        else:
            _bundle_labels = [o[0] for o in _bundle_options]
            _bi = st.selectbox(
                "Model bundle",
                options=range(len(_bundle_options)),
                format_func=lambda i: _bundle_labels[i],
                key="launch_model_bundle",
                help="Select which PD+LGD model artifacts to use for this simulation.",
            )
            _selected_pd_bundle_dir = _bundle_options[_bi][1]
            _selected_lgd_bundle_dir = _bundle_options[_bi][2]
            _sel = _bundle_options[_bi]
            if "Legacy" not in _sel[0]:
                st.caption("PD and LGD artifacts loaded from the selected bundle.")
            else:
                st.caption("Using global model files. Train a dataset-specific bundle for explicit lineage.")

    with st.form("launch_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            backend = st.selectbox("Backend", ["cpu", "mps", "cuda"], index=0)
        with col2:
            n_sims = st.number_input("Simulations", min_value=100, max_value=1_000_000,
                                     value=10_000, step=1_000)
        with col3:
            dtype = st.selectbox("dtype", ["float32", "float16", "float64"], index=0)

        with st.expander("Parallelism (optional)"):
            pc1, pc2, pc3 = st.columns(3)
            cpu_workers = pc1.number_input("CPU workers", min_value=0, value=0,
                                           help="0 = auto")
            torch_threads = pc2.number_input("Torch threads/worker", min_value=0, value=0,
                                             help="0 = auto")
            pyarrow_threads = pc3.number_input("PyArrow threads", min_value=1, value=1)

        with st.expander("Sample Efficiency (optional)", expanded=True):
            se1, se2 = st.columns(2)
            antithetic_variates = se1.checkbox(
                "Antithetic variates",
                value=True,
                help=(
                    "Draw paired +z / −z shocks to reduce estimator variance "
                    "without extra model evaluations. Recommended — nearly free."
                ),
            )
            adaptive_stopping = se2.checkbox(
                "Adaptive stopping",
                value=False,
                help=(
                    "Stop early when key risk metrics have converged within the "
                    "chosen tolerance across consecutive simulation batches."
                ),
            )

            if adaptive_stopping:
                as1, as2, as3 = st.columns(3)
                sim_batch_size = as1.number_input(
                    "Sim batch size",
                    min_value=100,
                    max_value=500_000,
                    value=10_000,
                    step=1_000,
                    help="Simulations per convergence-check batch.",
                )
                conv_tolerance = as2.number_input(
                    "Convergence tolerance",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.01,
                    step=0.001,
                    format="%.3f",
                    help="Stop when all tracked metrics change less than this fraction (1% = 0.01).",
                )
                max_sims_override = as3.number_input(
                    "Max simulations",
                    min_value=0,
                    max_value=1_000_000,
                    value=0,
                    step=10_000,
                    help="Hard cap on simulations (0 = use Simulations field above).",
                )
                conv_metrics = st.multiselect(
                    "Convergence metrics",
                    options=["expected_loss", "var_99", "var_999", "es_99"],
                    default=["expected_loss", "var_99"],
                    help="All selected metrics must converge before early stopping fires.",
                )
            else:
                sim_batch_size = 10_000
                conv_tolerance = 0.01
                max_sims_override = 0
                conv_metrics = ["expected_loss", "var_99"]

        submitted = st.form_submit_button("▶ Run Simulation", type="primary")

    if submitted:
        if backend in active_backends:
            st.warning(f"A {backend} run is already active. Wait for it to finish before starting another.")
            return

        # ── Block if no compatible PD/LGD bundle ─────────────────────────
        if _selected_pd_bundle_dir is None or _selected_lgd_bundle_dir is None:
            st.error(
                "Cannot launch Monte Carlo: no compatible PD/LGD model bundle available. "
                "Train PD and LGD models on the selected dataset first."
            )
            return

        # ── Resolve dataset selection ────────────────────────────────────
        portfolio_path_arg: str | None = None
        portfolio_label = "Default (auto-detected)"
        portfolio_source = "default"

        if ds_mode == "Preset" and ds_preset_key:
            _presets = get_dataset_presets()
            p_info = _presets.get(ds_preset_key, {})
            portfolio_path_arg = p_info.get("path")
            portfolio_label = p_info.get("label", ds_preset_key)
            portfolio_source = "preset"
        elif ds_mode == "Custom path" and ds_custom_path.strip():
            portfolio_path_arg = ds_custom_path.strip()
            portfolio_label = Path(portfolio_path_arg).name
            portfolio_source = "custom"

        if portfolio_path_arg is not None:
            meta = _get_portfolio_meta(portfolio_path_arg)
            if not meta["exists"]:
                st.error(f"Dataset path not found: `{portfolio_path_arg}`")
                return
            if portfolio_source == "custom":
                st.session_state["_last_ds_custom_path"] = portfolio_path_arg
            row_info = (
                f"{meta['row_count']:,} rows · {meta['file_count']} file(s)"
                if meta.get("row_count") else f"{meta['file_count']} file(s)"
            )
            st.info(f"Dataset: **{portfolio_label}** — {row_info}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"mc_{backend}_{ts}"
        log_path = MODEL_DIR / f"{prefix}.log"

        cmd = [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--backend", backend,
            "--output-prefix", prefix,
            "--n-simulations", str(int(n_sims)),
            "--dtype", dtype,
            "--pyarrow-threads", str(int(pyarrow_threads)),
        ]
        if cpu_workers > 0:
            cmd += ["--cpu-workers", str(int(cpu_workers))]
        if torch_threads > 0:
            cmd += ["--torch-threads-per-worker", str(int(torch_threads))]
        if portfolio_path_arg is not None:
            cmd += ["--portfolio-path", portfolio_path_arg]
        # Explicit PD/LGD bundle dirs for unambiguous model lineage
        cmd += ["--pd-bundle-dir", _selected_pd_bundle_dir]
        cmd += ["--lgd-bundle-dir", _selected_lgd_bundle_dir]
        if antithetic_variates:
            cmd += ["--antithetic-variates"]
        if adaptive_stopping:
            cmd += [
                "--adaptive-stopping",
                "--simulation-batch-size", str(int(sim_batch_size)),
                "--convergence-tolerance", str(conv_tolerance),
            ]
            if max_sims_override > 0:
                cmd += ["--max-simulations", str(int(max_sims_override))]
            if conv_metrics:
                cmd += ["--convergence-metrics"] + conv_metrics

        try:
            log_file = open(log_path, "w")
        except OSError as exc:
            st.error(f"Cannot open log file: {exc}")
            return
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            )
        except Exception as exc:
            log_file.close()
            st.error(f"Failed to launch runner: {exc}")
            return
        finally:
            log_file.close()  # parent closes its copy; child retains its own fd

        try:
            import psutil as _psutil
            pid_ct = _psutil.Process(proc.pid).create_time()
        except Exception:
            pid_ct = 0.0

        config = {
            "backend": backend,
            "n_simulations": int(n_sims),
            "dtype": dtype,
            "cpu_workers": int(cpu_workers) if cpu_workers > 0 else None,
            "torch_threads_per_worker": int(torch_threads) if torch_threads > 0 else None,
            "pyarrow_threads": int(pyarrow_threads),
            "antithetic_variates": antithetic_variates,
            "adaptive_stopping": adaptive_stopping,
            "simulation_batch_size": int(sim_batch_size) if adaptive_stopping else None,
            "convergence_tolerance": conv_tolerance if adaptive_stopping else None,
            "max_simulations_override": int(max_sims_override) if adaptive_stopping and max_sims_override > 0 else None,
            "convergence_metrics": conv_metrics if adaptive_stopping else None,
            "portfolio_path": portfolio_path_arg,
            "portfolio_label": portfolio_label,
            "portfolio_source": portfolio_source,
            "pd_bundle_dir": _selected_pd_bundle_dir,
            "lgd_bundle_dir": _selected_lgd_bundle_dir,
        }
        write_state(
            prefix=prefix,
            pid=proc.pid,
            pid_create_time=pid_ct,
            launch_ts=datetime.now().isoformat(),
            log_path=str(log_path),
            config=config,
        )

        st.session_state.setdefault("procs", {})[prefix] = proc
        st.session_state.setdefault("active_prefixes", []).append(prefix)
        st.session_state["_just_submitted"] = prefix
        st.rerun()


# ---------------------------------------------------------------------------
# Step metrics (U5)
# ---------------------------------------------------------------------------

def _render_step_metrics(run: RunInfo) -> None:
    summary_path = str(MODEL_DIR / f"{run.prefix}_runtime_summary.csv")
    risk_path = str(MODEL_DIR / f"{run.prefix}_risk_metrics.csv")

    summary_df = load_run_summary(summary_path)
    risk_df = load_run_risk(risk_path)
    log_metrics = parse_log_metrics(run.log_path or "")

    left, right = st.columns(2)

    with left:
        st.markdown("**Step Timings**")
        if summary_df is not None:
            rows = []
            for col, label in STEP_NAME_MAP.items():
                if col in summary_df.columns:
                    val = summary_df.iloc[0][col]
                    rows.append({"Step": label, "Seconds": f"{val:.1f}s"})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info("Step timing not available — run did not complete.")

    with right:
        st.markdown("**Final Risk Metrics**")
        if risk_df is not None:
            r = risk_df.iloc[0]
            tb = r.get("total_balance", 1)
            m1, m2 = st.columns(2)
            m1.metric("Expected Loss", f"${r['expected_loss']/1e6:,.0f}M",
                      f"{r['expected_loss']/tb*100:.2f}%")
            m2.metric("VaR 99%", f"${r['var_99']/1e6:,.0f}M",
                      f"{r['var_99']/tb*100:.2f}%", delta_color="inverse")
            m3, m4 = st.columns(2)
            m3.metric("VaR 99.9%", f"${r['var_999']/1e6:,.0f}M",
                      f"{r['var_999']/tb*100:.2f}%", delta_color="inverse")
            m4.metric("ES 99%", f"${r['es_99']/1e6:,.0f}M",
                      f"{r['es_99']/tb*100:.2f}%", delta_color="inverse")
        else:
            st.info("Risk metrics not available.")

    # Portfolio/baseline summary from log
    st.markdown("**Portfolio Baseline**")
    if log_metrics:
        cols = st.columns(5)
        cols[0].metric("Loans", f"{int(log_metrics['n_loans']):,}" if "n_loans" in log_metrics else "—")
        cols[1].metric("Balance", f"${log_metrics.get('total_balance', 0):.1f}B" if "total_balance" in log_metrics else "—")
        cols[2].metric("PD", f"{log_metrics.get('baseline_pd', 0):.2f}%" if "baseline_pd" in log_metrics else "—")
        cols[3].metric("LGD", f"{log_metrics.get('baseline_lgd', 0):.2f}%" if "baseline_lgd" in log_metrics else "—")
        cols[4].metric("Annual EL", f"${log_metrics.get('baseline_el_m', 0):,.0f}M" if "baseline_el_m" in log_metrics else "—")
    else:
        st.info("Portfolio summary not available — no log file captured.")


# ---------------------------------------------------------------------------
# Per-run charts (U6)
# ---------------------------------------------------------------------------

def _render_run_charts(run: RunInfo) -> None:
    if run.status == "running":
        return  # files may be incomplete

    dist_path = str(MODEL_DIR / f"{run.prefix}_loss_distribution.csv")
    sens_path = str(MODEL_DIR / f"{run.prefix}_sensitivity.csv")
    scen_path = str(MODEL_DIR / f"{run.prefix}_scenarios.csv")
    risk_path = str(MODEL_DIR / f"{run.prefix}_risk_metrics.csv")

    dist_df = load_run_dist(dist_path)
    sens_df = load_run_sensitivity(sens_path)
    scen_df = load_run_scenarios(scen_path)
    risk_df = load_run_risk(risk_path)

    tab_dist, tab_torn, tab_scat = st.tabs(["Loss Distribution", "Sensitivity", "Scenarios"])

    with tab_dist:
        if dist_df is not None and risk_df is not None:
            r = risk_df.iloc[0]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=dist_df["portfolio_loss"] / 1e6, nbinsx=80,
                                       name="Losses", marker_color=COLORS["primary"], opacity=0.7))
            fig.add_vline(x=r["expected_loss"] / 1e6, line_color=COLORS["success"], line_width=2,
                          annotation_text=f"EL: ${r['expected_loss']/1e6:,.0f}M",
                          annotation_font_color=COLORS["success"])
            fig.add_vline(x=r["var_99"] / 1e6, line_dash="dash", line_color=COLORS["warning"], line_width=2,
                          annotation_text=f"VaR99: ${r['var_99']/1e6:,.0f}M",
                          annotation_font_color=COLORS["warning"])
            fig.add_vline(x=r["var_999"] / 1e6, line_dash="dash", line_color=COLORS["danger"], line_width=2,
                          annotation_text=f"VaR99.9: ${r['var_999']/1e6:,.0f}M",
                          annotation_font_color=COLORS["danger"])
            fig.update_layout(title="Loss Distribution", xaxis_title="Portfolio Loss ($M)",
                              yaxis_title="Frequency", showlegend=False)
            style_chart(fig, 400)
            st.plotly_chart(fig, use_container_width=True, key=f"{run.prefix}-loss-distribution")
        else:
            st.info("Chart not available — loss_distribution.csv or risk_metrics.csv not yet written.")

    with tab_torn:
        if sens_df is not None:
            tornado = []
            for var in sens_df["variable"].unique():
                vd = sens_df[sens_df["variable"] == var]
                loss_series = _get_loss_series(vd)
                if loss_series is None:
                    continue
                tornado.append({
                    "variable": var,
                    "min": loss_series.min() / 1e6,
                    "max": loss_series.max() / 1e6,
                    "range": (loss_series.max() - loss_series.min()) / 1e6,
                })
            if tornado:
                tdf = pd.DataFrame(tornado).sort_values("range", ascending=True)
                fig = go.Figure()
                fig.add_trace(go.Bar(y=tdf["variable"], x=tdf["min"], orientation="h",
                                      name="Best", marker_color=COLORS["success"], opacity=0.7))
                fig.add_trace(go.Bar(y=tdf["variable"], x=tdf["max"] - tdf["min"], orientation="h",
                                      name="Worst Increment", marker_color=COLORS["danger"],
                                      opacity=0.7, base=tdf["min"]))
                fig.update_layout(title="Sensitivity: Loss Range by Macro Variable",
                                   xaxis_title="Loss ($M)", barmode="overlay")
                style_chart(fig, 400)
                st.plotly_chart(fig, use_container_width=True, key=f"{run.prefix}-sensitivity")
            else:
                st.info("Chart not available — sensitivity.csv is missing a loss column.")
        else:
            st.info("Chart not available — sensitivity.csv not yet written.")

    with tab_scat:
        if scen_df is not None and "unemployment_rate" in scen_df.columns and "portfolio_loss" in scen_df.columns:
            color_col = "hpi_change_annual" if "hpi_change_annual" in scen_df.columns else None
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=scen_df["unemployment_rate"],
                y=scen_df["portfolio_loss"] / 1e6,
                mode="markers",
                marker=dict(
                    size=3,
                    color=scen_df[color_col] if color_col else COLORS["primary"],
                    colorscale="RdYlGn" if color_col else None,
                    colorbar=dict(title="HPI Chg %") if color_col else None,
                    opacity=0.5,
                ),
            ))
            fig.update_layout(title="Scenarios: Unemployment vs Loss",
                               xaxis_title="Unemployment (%)", yaxis_title="Loss ($M)")
            style_chart(fig, 400)
            st.plotly_chart(fig, use_container_width=True, key=f"{run.prefix}-scenarios")
        else:
            st.info("Chart not available — scenarios.csv not yet written.")


# ---------------------------------------------------------------------------
# Config table (U7)
# ---------------------------------------------------------------------------

def _render_config_table(run: RunInfo) -> None:
    cfg = run.config or {}

    # Dataset badge — shown above the table for prominence
    p_path = cfg.get("portfolio_path")
    p_label = cfg.get("portfolio_label")
    display_label = _portfolio_display_label(p_path, p_label)
    st.markdown(
        f'<div style="background:#f0f9ff;border-left:3px solid #0ea5e9;border-radius:0 6px 6px 0;'
        f'padding:6px 12px;margin-bottom:8px;font-size:0.85rem;">'
        f'<strong>Dataset:</strong> {display_label}'
        + (f'<br><span style="font-size:0.75rem;color:#64748b;font-family:monospace;">{p_path}</span>'
           if p_path else "")
        + '</div>',
        unsafe_allow_html=True,
    )

    display_fields = [
        ("backend", "Backend"),
        ("n_simulations", "Simulations"),
        ("dtype", "dtype"),
        ("execution_mode", "Execution mode"),
        ("cpu_workers", "CPU workers"),
        ("torch_threads_per_worker", "Torch threads/worker"),
        ("pyarrow_threads", "PyArrow threads"),
        ("scenario_batch_size", "Scenario batch size"),
        ("loan_chunk_size", "Loan chunk size"),
        ("antithetic_variates", "Antithetic variates"),
        ("adaptive_stopping", "Adaptive stopping"),
        ("portfolio_label", "Dataset"),
        ("portfolio_source", "Dataset source"),
    ]
    rows = [
        {"Parameter": label, "Value": str(cfg.get(key, "—") or "—")}
        for key, label in display_fields
        if key != "_legacy" and cfg.get(key) not in (None, "")
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


_MAX_LOG_DISPLAY_BYTES = 512 * 1024  # 512 KB

def _render_raw_log(run: RunInfo) -> None:
    with st.expander("Raw log output"):
        if run.log_path and Path(run.log_path).exists():
            p = Path(run.log_path)
            size = p.stat().st_size
            if size > _MAX_LOG_DISPLAY_BYTES:
                with open(p, "rb") as f:
                    f.seek(-_MAX_LOG_DISPLAY_BYTES, 2)
                    content = f.read().decode("utf-8", errors="replace")
                st.caption(f"Showing last 512 KB of {size / 1e6:.1f} MB log.")
            else:
                content = p.read_text(errors="replace")
            st.code(content, language=None)
        else:
            st.info("No log captured (CLI run or log file missing).")


# ---------------------------------------------------------------------------
# Run card (U7)
# ---------------------------------------------------------------------------

def _status_badge(status: str) -> None:
    if status == "completed":
        st.success("✓ Completed")
    elif status == "running":
        st.info("⏳ Running")
    else:
        st.error("✗ Failed")


def _render_run_card(run: RunInfo) -> None:
    legacy = run.config.get("_legacy", False)
    label_suffix = " · legacy prefix" if legacy else ""
    source_badge = f"[{run.source.upper()}]"

    with st.expander(
        f"{run.prefix}  {source_badge}{label_suffix}  —  {run.status.upper()}",
        expanded=(run.status == "running"),
    ):
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            st.markdown(f"**Prefix:** `{run.prefix}`")
            if legacy:
                st.caption("⚠ Legacy prefix — this file may have been overwritten by a subsequent CLI run.")
        with hcol2:
            _status_badge(run.status)

        st.markdown("**Launch Config**")
        _render_config_table(run)

        if run.status == "running":
            st.markdown("**Live Log (last 50 lines)**")
            tail = read_log_tail(run.log_path or "", n=50)
            st.code(tail or "(log initializing…)", language=None)
            st.caption("Run in progress — page refreshes every ~3 seconds.")

        elif run.status == "completed":
            tab_metrics, tab_charts = st.tabs(["Step Metrics", "Charts"])
            with tab_metrics:
                _render_step_metrics(run)
            with tab_charts:
                _render_run_charts(run)
            _render_raw_log(run)

        else:  # failed
            if run.exit_code == -1:
                st.error("Run terminated with unknown exit code (process gone before capture or server restart).")
            else:
                st.error(f"Run exited with code {run.exit_code}.")
            _render_step_metrics(run)
            _render_raw_log(run)


# ---------------------------------------------------------------------------
# Polling loop (U4)
# ---------------------------------------------------------------------------

def _poll_active_runs(all_runs: list[RunInfo]) -> None:
    procs: dict = st.session_state.get("procs", {})
    completed_prefixes = []

    for prefix, proc in list(procs.items()):
        rc = proc.poll()
        if rc is not None:
            # Non-zero exit: the runner may have returned an error code even
            # after writing all outputs (e.g. warnings-as-errors). Prefer
            # artifact evidence over the exit code to avoid false failures.
            if rc == 0:
                status, exit_code = "completed", 0
            elif is_run_complete(prefix):
                status, exit_code = "completed", rc  # outputs present → completed
            else:
                status, exit_code = "failed", rc
            update_state(prefix, status, exit_code)
            completed_prefixes.append(prefix)

    for prefix in completed_prefixes:
        procs.pop(prefix, None)

    # Post-refresh recovery: sidecars that say running but have no Popen
    # (e.g. after a Streamlit server restart mid-run). Use artifact evidence
    # rather than defaulting to failed — the process may have finished cleanly.
    for run in all_runs:
        if run.status == "running" and run.prefix not in procs:
            if not is_alive(run.pid, run.pid_create_time or 0.0):
                terminal = "completed" if is_run_complete(run.prefix) else "failed"
                update_state(run.prefix, terminal, 0 if terminal == "completed" else -1)

    still_active = [p for p in procs.values() if p.poll() is None]
    if still_active:
        time.sleep(3)
        st.rerun()


# ---------------------------------------------------------------------------
# Main render (U7)
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Simulation Runs")
    info_box(
        "Launch <strong>run_monte_carlo_custom_backend.py</strong> with configurable options. "
        "Each run produces an isolated file set in <code>models/</code>. "
        "Previously run CLI simulations are auto-discovered."
    )

    st.session_state.setdefault("procs", {})
    st.session_state.setdefault("active_prefixes", [])

    all_runs = discover_runs()

    if "_just_submitted" in st.session_state:
        submitted_prefix = st.session_state.pop("_just_submitted")
        st.success(f"✓ Simulation submitted: **{submitted_prefix}**")

    _render_launch_form(all_runs)

    st.markdown("---")
    section_header("Simulation Runs")

    if not all_runs and not st.session_state["active_prefixes"]:
        st.info(
            "No simulation runs recorded yet. Use the launch form above to start your first run, "
            "or run `src/run_monte_carlo_custom_backend.py` from the CLI — results will appear here automatically."
        )
        return

    analytics_df = build_run_analytics_df(all_runs)
    if analytics_df is not None:
        _render_cross_run_analytics(analytics_df)
        st.markdown("---")

    section_header("Run Log")
    for run in all_runs:
        _render_run_card(run)

    _poll_active_runs(all_runs)
