"""
PD Model Page — Probability of Default
Supports dataset-aware training launches and bundle-aware chart display.
"""
from __future__ import annotations

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
from dataset_helpers import (
    assess_training_compatibility,
    get_dataset_presets,
    get_portfolio_meta,
    portfolio_display_label,
)
from portfolio_cache import dataset_fingerprint
from model_bundle import (
    BUNDLES_ROOT,
    MODEL_DIR,
    generate_bundle_id,
    get_bundle_dir,
    legacy_pd_exists,
    list_bundles,
    find_compatible_bundles,
    read_bundle_metadata,
)
from training_run_state import (
    TrainingRunInfo,
    active_training_run,
    discover_training_runs,
    is_alive,
    training_is_complete,
    update_training_state,
    write_training_state,
)

REPO_ROOT = Path(__file__).parent.parent.parent
RUNNER_SCRIPT = REPO_ROOT / "src" / "run_pd_model.py"


# ---------------------------------------------------------------------------
# Bundle-keyed data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=0)
def _load_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        if p.name == "validation_summary.csv":
            return pd.read_csv(p, index_col=0)
        return pd.read_csv(p)
    except Exception:
        return None


def _bundle_path(bundle_dir: Path, filename: str) -> str:
    return str(bundle_dir / filename)


# ---------------------------------------------------------------------------
# Bundle selector
# ---------------------------------------------------------------------------

def _render_bundle_selector() -> tuple:
    """
    Render a bundle/dataset-source selector.
    Returns (bundle_dir: Path | None, bundle_label: str, dataset_fingerprint_str: str | None)
    bundle_dir=None → use legacy global files from models/
    """
    all_pd_bundles = list_bundles("pd")

    options = []
    if legacy_pd_exists():
        options.append(("legacy", "Legacy (global models/ files)", MODEL_DIR, ""))
    for b in all_pd_bundles:
        ts = b["timestamp"][:16] if b["timestamp"] else ""
        ds_label = b["dataset_label"] or b["dataset_path"] or ""
        label = f"{b['bundle_id']}"
        if ds_label:
            label = f"{ds_label} · {ts}" if ts else ds_label
        elif ts:
            label = f"{b['bundle_id']} · {ts}"
        options.append((b["bundle_id"], label, b["bundle_dir"], b["dataset_fingerprint"]))

    if not options:
        st.info("No PD model found. Train one using the section below.")
        return None, "None", None

    labels = [o[1] for o in options]
    idx = st.selectbox(
        "Display bundle",
        options=range(len(options)),
        format_func=lambda i: labels[i],
        key="pd_bundle_select",
    )
    selected = options[idx]
    return selected[2], selected[1], selected[3] or None


# ---------------------------------------------------------------------------
# Dataset selector for training (outside form so conditional widgets render)
# ---------------------------------------------------------------------------

def _render_dataset_selector_train() -> tuple:
    """Returns (parquet_path, label, fingerprint, is_trainable)."""
    ds_mode = st.radio(
        "Training dataset",
        ["Default", "Preset", "Custom path"],
        horizontal=True,
        key="pd_train_ds_mode",
        help="Dataset to train the PD model against.",
    )
    if ds_mode == "Default":
        default_p = REPO_ROOT / "data" / "processed" / "loan_level_combined.parquet"
        if default_p.exists():
            fp = dataset_fingerprint(str(default_p))
            return str(default_p), "Standard · loan_level_combined.parquet", fp, True
        return None, "Default (not found)", None, False

    if ds_mode == "Preset":
        presets = {
            key: info
            for key, info in get_dataset_presets().items()
            if assess_training_compatibility(info["path"]).get("pd_trainable")
        }
        if not presets:
            st.warning(
                "No PD-trainable preset datasets found. "
                "Fullbook presets are typically scoring/simulation datasets, not labeled PD training datasets. "
                "Use a labeled custom path if needed."
            )
            return None, "Default", None, False
        key = st.selectbox(
            "Dataset", list(presets.keys()),
            format_func=lambda k: presets[k]["label"],
            key="pd_train_ds_preset",
        )
        info = presets[key]
        st.caption(info["description"])
        fp = dataset_fingerprint(info["path"])
        return info["path"], info["label"], fp, True

    custom = st.text_input(
        "Parquet path",
        value=st.session_state.get("_pd_train_custom", ""),
        placeholder="/absolute/path/to/dataset.parquet",
        key="pd_train_ds_custom",
    )
    if custom.strip():
        st.session_state["_pd_train_custom"] = custom.strip()
        fp = dataset_fingerprint(custom.strip())
        label = portfolio_display_label(custom.strip())
        compat = assess_training_compatibility(custom.strip())
        if compat.get("error"):
            st.warning(f"Compatibility check unavailable: {compat['error']}")
            return custom.strip(), label, fp, False
        if not compat.get("pd_trainable"):
            st.error(
                "This dataset is not compatible with PD training. "
                f"Positive defaults found: {compat.get('positive_defaults', 0):,}. "
                "PD training requires labeled default events in train/validation/test splits. "
                "This fullbook dataset is suitable for Monte Carlo scoring/simulation, not PD training."
            )
            return custom.strip(), label, fp, False
        st.success(
            f"PD-training compatible dataset detected — "
            f"{compat.get('positive_defaults', 0):,} defaulted loans available."
        )
        return custom.strip(), label, fp, True
    return None, "Default", None, False


# ---------------------------------------------------------------------------
# Launch section
# ---------------------------------------------------------------------------

def _render_launch_section() -> None:
    section_header("Train PD Model")

    active = active_training_run("pd")
    if active:
        _render_active_run(active)
        return

    # Dataset selector (outside form for live interactivity)
    with st.expander("Training Dataset", expanded=True):
        parquet_path, ds_label, fp, is_trainable = _render_dataset_selector_train()

    with st.form("pd_train_form"):
        st.markdown("Click **Train** to launch PD model training against the selected dataset.")
        st.caption(
            "Training includes: WoE/IV computation, Logistic Regression + XGBoost, "
            "full validation, and artifact persistence."
        )
        submitted = st.form_submit_button("▶ Train PD Model", type="primary")

    if submitted:
        if parquet_path is None:
            st.error("No valid dataset path. Select a dataset above.")
            return
        if not is_trainable:
            st.error(
                "Selected dataset is not PD-trainable. "
                "Choose the labeled standard training dataset or another labeled custom parquet."
            )
            return
        if not Path(parquet_path).exists():
            st.error(f"Dataset path not found: `{parquet_path}`")
            return

        bundle_id = generate_bundle_id("pd", fp or "unknown")
        bundle_dir = get_bundle_dir(bundle_id)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        log_path = bundle_dir / "training.log"
        cmd = [
            sys.executable, str(RUNNER_SCRIPT),
            "--portfolio-path", parquet_path,
            "--bundle-dir", str(bundle_dir),
            "--dataset-label", ds_label,
        ]
        if fp:
            cmd += ["--dataset-fingerprint", fp]

        try:
            log_file = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                    cwd=str(REPO_ROOT))
        except Exception as exc:
            st.error(f"Failed to launch PD training: {exc}")
            return
        finally:
            log_file.close()

        try:
            import psutil as _psutil
            pid_ct = _psutil.Process(proc.pid).create_time()
        except Exception:
            pid_ct = 0.0

        write_training_state(
            bundle_id=bundle_id,
            model_type="pd",
            pid=proc.pid,
            pid_create_time=pid_ct,
            launch_ts=datetime.now().isoformat(),
            log_path=str(log_path),
            bundle_dir=str(bundle_dir),
            dataset_path=parquet_path,
            dataset_label=ds_label,
            dataset_fingerprint=fp or "",
        )
        st.session_state.setdefault("pd_procs", {})[bundle_id] = proc
        st.success(f"PD training launched — bundle: `{bundle_id}`")
        st.rerun()


def _render_active_run(run: TrainingRunInfo) -> None:
    st.info(f"**PD training in progress** — bundle `{run.bundle_id}`")
    if run.dataset_label:
        st.caption(f"Dataset: {run.dataset_label}")

    if run.log_path and Path(run.log_path).exists():
        with st.expander("Live log (last 30 lines)", expanded=True):
            try:
                lines = Path(run.log_path).read_text().splitlines()
                st.code("\n".join(lines[-30:]), language=None)
            except Exception:
                st.caption("Log not readable yet.")

    col_refresh, col_cancel = st.columns([1, 1])
    with col_refresh:
        if st.button("↻ Refresh", key="pd_refresh"):
            st.rerun()


# ---------------------------------------------------------------------------
# Recent training runs
# ---------------------------------------------------------------------------

def _render_training_history() -> None:
    runs = discover_training_runs("pd")
    if not runs:
        return
    with st.expander("Training history", expanded=False):
        for run in runs[:8]:
            status_icon = {"completed": "✅", "failed": "❌", "running": "⏳"}.get(run.status, "?")
            ts = (run.launch_ts or "")[:16]
            ds = run.dataset_label or "unknown dataset"
            st.markdown(
                f"{status_icon} `{run.bundle_id}` · {ds} · {ts}"
            )


# ---------------------------------------------------------------------------
# Charts (bundle-aware)
# ---------------------------------------------------------------------------

def _render_charts(bundle_dir: Path) -> None:
    vs_path = str(bundle_dir / "validation_summary.csv")
    cal_path = str(bundle_dir / "lr_calibration_validation.csv")
    iv_path = str(bundle_dir / "iv_summary.csv")
    coef_path = str(bundle_dir / "lr_coefficients.csv")

    vs = _load_csv(vs_path)
    meta = read_bundle_metadata(bundle_dir)
    section_header("Model Performance Summary")

    k1, k2, k3, k4 = st.columns(4)
    if vs is not None:
        lr = vs["logistic_regression"]
        xgb = vs["xgboost"] if "xgboost" in vs.columns else pd.Series(dtype=float)
        xgb_metrics_available = pd.notna(xgb.get("val_auc")) and meta.get("xgboost_available", True) is not False
        xgb_error = meta.get("xgboost_error") or ""
        k1.metric("LR Validation AUC", f"{lr['val_auc']:.4f}")
        if xgb_metrics_available:
            k2.metric("XGBoost Val AUC", f"{xgb['val_auc']:.4f}",
                      f"+{xgb['val_auc']-lr['val_auc']:.4f}")
        else:
            k2.metric("XGBoost Val AUC", "Unavailable")
        k3.metric("LR Validation KS", f"{lr['val_ks']:.4f}")
        k4.metric("PSI (Train vs Val)", f"{lr['psi_val']:.4f}",
                  "Needs monitoring" if lr["psi_val"] > 0.25 else "Stable",
                  delta_color="inverse" if lr["psi_val"] > 0.25 else "normal")
        if not xgb_metrics_available and xgb_error:
            st.warning(
                "XGBoost challenger is unavailable for this bundle. "
                f"Runner recorded: `{xgb_error.splitlines()[0]}`"
            )
    else:
        k1.metric("LR Validation AUC", "—")
        k2.metric("XGBoost Val AUC", "—")
        k3.metric("LR Validation KS", "—")
        k4.metric("PSI (Train vs Val)", "—")
        st.info(f"Validation summary not found in `{bundle_dir.name}`. Run PD training first.")
        return

    tab_cmp, tab_cal, tab_iv, tab_coef, tab_why = st.tabs(
        ["Model Comparison", "Calibration", "Information Value", "Coefficients", "Why Logistic Regression?"]
    )

    with tab_cmp:
        cc, ce = st.columns([2, 1])
        with cc:
            lr_v = [lr["train_auc"], lr["val_auc"], lr["test_auc"],
                    lr["train_ks"], lr["val_ks"], lr["test_ks"]]
            labels = ["Train AUC", "Val AUC", "Test AUC", "Train KS", "Val KS", "Test KS"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=lr_v, name="Logistic Regression",
                                 marker_color=COLORS["primary"],
                                 text=[f"{v:.4f}" for v in lr_v], textposition="outside"))
            if xgb_metrics_available:
                xg_v = [xgb["train_auc"], xgb["val_auc"], xgb["test_auc"],
                        xgb["train_ks"], xgb["val_ks"], xgb["test_ks"]]
                fig.add_trace(go.Bar(x=labels, y=xg_v, name="XGBoost Challenger",
                                     marker_color=COLORS["success"],
                                     text=[f"{v:.4f}" for v in xg_v], textposition="outside"))
            fig.add_hline(y=0.75, line_dash="dash", line_color=COLORS["warning"],
                          annotation_text="AUC Target: 0.75")
            fig.update_layout(title="Primary vs Challenger Performance" if xgb_metrics_available else "Primary Model Performance",
                              barmode="group", bargap=0.2)
            style_chart(fig, 480)
            st.plotly_chart(fig, use_container_width=True)
        with ce:
            if xgb_metrics_available:
                info_box("<strong>AUC-ROC</strong> measures rank-ordering ability (>0.75 is good). "
                         "<strong>KS</strong> measures max separation (>0.30 is good). "
                         "XGBoost gains +7.5 AUC points over LR, the cost of interpretability.")
            else:
                info_box("<strong>AUC-ROC</strong> measures rank-ordering ability (>0.75 is good). "
                         "<strong>KS</strong> measures max separation (>0.30 is good). "
                         "This bundle was validated with Logistic Regression only.")
                warning_box("XGBoost challenger metrics are unavailable for this bundle. "
                            "On macOS this is commonly caused by a missing OpenMP runtime (`libomp`).")
            warning_box("<strong>PSI > 0.25:</strong> Score distribution shifted between training "
                        "and validation. Reflects genuine deterioration in origination quality.")

    with tab_cal:
        cc, ce = st.columns([2, 1])
        with cc:
            cal = _load_csv(cal_path)
            if cal is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(range(1, len(cal)+1)),
                                     y=cal["avg_predicted_pd"]*100,
                                     name="Predicted PD (%)",
                                     marker_color=COLORS["primary"], opacity=0.7))
                fig.add_trace(go.Scatter(x=list(range(1, len(cal)+1)),
                                         y=cal["actual_default_rate"]*100,
                                         name="Actual Default Rate (%)",
                                         mode="lines+markers",
                                         marker=dict(color=COLORS["danger"], size=10),
                                         line=dict(width=2.5)))
                fig.update_layout(title="Calibration: Predicted vs Actual by Decile",
                                  xaxis_title="Risk Decile (1=Lowest Risk)",
                                  yaxis_title="Rate (%)", bargap=0.3)
                style_chart(fig, 480)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Calibration data not available in this bundle.")
        with ce:
            info_box("<strong>Why calibration matters more than AUC:</strong> "
                     "If the model says 5% PD but actual is 8%, the bank is under-reserved by 3%. "
                     "Across $694B, that is $21B missing. "
                     "LR was trained without class reweighting to preserve calibration.")

    with tab_iv:
        cc, ce = st.columns([2, 1])
        with cc:
            iv = _load_csv(iv_path)
            if iv is not None:
                iv = iv.head(15)
                colors = [
                    COLORS["danger"] if v > 0.5 else COLORS["success"] if v > 0.3
                    else COLORS["primary"] if v > 0.1 else COLORS["warning"] if v > 0.02
                    else COLORS.get("gray", "#94a3b8")
                    for v in iv["iv"]
                ]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=iv["iv"], y=iv["feature"], orientation="h",
                                     marker_color=colors,
                                     text=[f"{v:.4f}" for v in iv["iv"]],
                                     textposition="outside"))
                fig.add_vline(x=0.02, line_dash="dash", line_color=COLORS.get("gray", "#94a3b8"))
                fig.add_vline(x=0.30, line_dash="dash", line_color=COLORS["success"])
                fig.add_vline(x=0.50, line_dash="dash", line_color=COLORS["danger"])
                fig.update_layout(title="Information Value: Feature Predictive Power",
                                  xaxis_title="IV", yaxis=dict(autorange="reversed"))
                style_chart(fig, 500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("IV summary not available in this bundle.")
        with ce:
            info_box("<strong>IV thresholds:</strong> Gray <0.02 (drop), Amber 0.02-0.10 (weak), "
                     "Blue 0.10-0.30 (medium), Green 0.30-0.50 (strong), Red >0.50 (suspicious).")

    with tab_coef:
        cc, ce = st.columns([2, 1])
        with cc:
            coef = _load_csv(coef_path)
            if coef is not None:
                coef["feature_clean"] = coef["feature"].str.replace("_woe", "")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=coef["coefficient"], y=coef["feature_clean"], orientation="h",
                    marker_color=[COLORS["primary"] if c > 0 else COLORS["danger"]
                                  for c in coef["coefficient"]],
                    text=[f"{v:+.4f}" for v in coef["coefficient"]], textposition="outside",
                ))
                fig.add_vline(x=0, line_color=COLORS.get("gray", "#94a3b8"))
                fig.update_layout(title="Logistic Regression Coefficients",
                                  xaxis_title="Coefficient",
                                  yaxis=dict(autorange="reversed"), showlegend=False)
                style_chart(fig, 450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coefficient data not available in this bundle.")
        with ce:
            info_box("Positive coefficients mean higher WoE (lower risk) reduces default probability. "
                     "All should be positive since WoE ensures monotonic relationships.")

    with tab_why:
        r1, r2 = st.columns(2)
        with r1:
            info_box("<strong>1. Interpretability:</strong> Coefficients directly answer 'why is this loan risky?'")
            info_box("<strong>2. Monotonicity:</strong> WoE guarantees higher FICO = lower risk, always.")
        with r2:
            info_box("<strong>3. Stability:</strong> Coefficients stable across time and resampling.")
            info_box("<strong>4. Regulatory acceptance:</strong> OCC/Fed/FDIC have decades of comfort with LR.")
        if xgb_metrics_available:
            warning_box("<strong>Tradeoff:</strong> LR sacrifices ~7.5 AUC points vs XGBoost. "
                        "This is the explicit cost of interpretability.")
        else:
            info_box("XGBoost comparison is unavailable for this bundle, so the page is showing the "
                     "interpretable Logistic Regression baseline only.")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("PD Model: Probability of Default")
    info_box(
        "<strong>What does the PD model do?</strong> It estimates the probability that a borrower "
        "will default within 12 months. First component of <strong>ECL = PD × LGD × EAD</strong>."
    )

    # ── Bundle / display selector ─────────────────────────────────────────
    section_header("Model Bundle")
    bundle_dir, bundle_label, bundle_fp = _render_bundle_selector()

    if bundle_dir is not None:
        ds_info_parts = [f"**Bundle:** `{bundle_dir.name}`"]
        if bundle_fp:
            ds_info_parts.append(f"Dataset fingerprint: `{bundle_fp}`")
        st.caption("  ·  ".join(ds_info_parts))

    _render_training_history()

    st.markdown("---")

    # ── Launch section ────────────────────────────────────────────────────
    _render_launch_section()

    st.markdown("---")

    # ── Charts from selected bundle ───────────────────────────────────────
    if bundle_dir is not None:
        _render_charts(bundle_dir)
    else:
        st.info("No PD model bundle available. Use the training section above to train one.")
