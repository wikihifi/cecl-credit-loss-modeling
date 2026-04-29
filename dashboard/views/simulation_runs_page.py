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
from run_state import MODEL_DIR, RunInfo, discover_runs, is_alive, read_state, update_state, write_state

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

        submitted = st.form_submit_button("▶ Run Simulation", type="primary")

    if submitted:
        if backend in active_backends:
            st.warning(f"A {backend} run is already active. Wait for it to finish before starting another.")
            return

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
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart not available — loss_distribution.csv or risk_metrics.csv not yet written.")

    with tab_torn:
        if sens_df is not None:
            tornado = []
            for var in sens_df["variable"].unique():
                vd = sens_df[sens_df["variable"] == var]
                tornado.append({"variable": var,
                                 "min": vd["loss"].min() / 1e6,
                                 "max": vd["loss"].max() / 1e6,
                                 "range": (vd["loss"].max() - vd["loss"].min()) / 1e6})
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
            st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chart not available — scenarios.csv not yet written.")


# ---------------------------------------------------------------------------
# Config table (U7)
# ---------------------------------------------------------------------------

def _render_config_table(run: RunInfo) -> None:
    cfg = run.config or {}
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
    ]
    rows = [
        {"Parameter": label, "Value": str(cfg.get(key, "—") or "—")}
        for key, label in display_fields
        if key != "_legacy"
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
            status = "completed" if rc == 0 else "failed"
            update_state(prefix, status, rc)
            completed_prefixes.append(prefix)

    for prefix in completed_prefixes:
        procs.pop(prefix, None)

    # Post-refresh recovery: sidecars that say running but have no Popen
    for run in all_runs:
        if run.status == "running" and run.prefix not in procs:
            if not is_alive(run.pid, run.pid_create_time or 0.0):
                update_state(run.prefix, "failed", -1)

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

    _render_launch_form(all_runs)

    st.markdown("---")
    section_header("Simulation Runs")

    if not all_runs and not st.session_state["active_prefixes"]:
        st.info(
            "No simulation runs recorded yet. Use the launch form above to start your first run, "
            "or run `src/run_monte_carlo_custom_backend.py` from the CLI — results will appear here automatically."
        )
        return

    for run in all_runs:
        _render_run_card(run)

    _poll_active_runs(all_runs)
