"""
LGD Model Page — Loss Given Default
Supports dataset-aware training launches and bundle-aware chart display.
"""
from __future__ import annotations

import subprocess
import sys
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
    portfolio_display_label,
)
from portfolio_cache import dataset_fingerprint
from model_bundle import (
    MODEL_DIR,
    generate_bundle_id,
    get_bundle_dir,
    legacy_lgd_exists,
    list_bundles,
    read_bundle_metadata,
)
from training_run_state import (
    TrainingRunInfo,
    active_training_run,
    discover_training_runs,
    update_training_state,
    write_training_state,
)

REPO_ROOT = Path(__file__).parent.parent.parent
RUNNER_SCRIPT = REPO_ROOT / "src" / "run_lgd_model.py"


# ---------------------------------------------------------------------------
# Bundle-keyed data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=0)
def _load_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, index_col=0) if "summary" in p.name else pd.read_csv(p)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bundle selector
# ---------------------------------------------------------------------------

def _render_bundle_selector() -> tuple:
    """Returns (bundle_dir: Path | None, bundle_label: str, fingerprint: str | None)."""
    all_lgd_bundles = list_bundles("lgd")

    options = []
    if legacy_lgd_exists():
        options.append(("legacy", "Legacy (global models/ files)", MODEL_DIR, ""))
    for b in all_lgd_bundles:
        ts = b["timestamp"][:16] if b["timestamp"] else ""
        ds_label = b["dataset_label"] or b["dataset_path"] or ""
        label = f"{b['bundle_id']}"
        if ds_label:
            label = f"{ds_label} · {ts}" if ts else ds_label
        elif ts:
            label = f"{b['bundle_id']} · {ts}"
        options.append((b["bundle_id"], label, b["bundle_dir"], b["dataset_fingerprint"]))

    if not options:
        st.info("No LGD model found. Train one using the section below.")
        return None, "None", None

    labels = [o[1] for o in options]
    idx = st.selectbox(
        "Display bundle",
        options=range(len(options)),
        format_func=lambda i: labels[i],
        key="lgd_bundle_select",
    )
    selected = options[idx]
    return selected[2], selected[1], selected[3] or None


# ---------------------------------------------------------------------------
# Dataset selector
# ---------------------------------------------------------------------------

def _render_dataset_selector_train() -> tuple:
    """Returns (parquet_path, label, fingerprint, is_trainable)."""
    ds_mode = st.radio(
        "Training dataset",
        ["Default", "Preset", "Custom path"],
        horizontal=True,
        key="lgd_train_ds_mode",
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
            if assess_training_compatibility(info["path"]).get("lgd_trainable")
        }
        if not presets:
            st.warning(
                "No LGD-trainable preset datasets found. "
                "Fullbook presets are typically scoring/simulation datasets, not labeled LGD training datasets. "
                "Use a labeled custom path if needed."
            )
            return None, "Default", None, False
        key = st.selectbox(
            "Dataset", list(presets.keys()),
            format_func=lambda k: presets[k]["label"],
            key="lgd_train_ds_preset",
        )
        info = presets[key]
        st.caption(info["description"])
        fp = dataset_fingerprint(info["path"])
        return info["path"], info["label"], fp, True

    custom = st.text_input(
        "Parquet path",
        value=st.session_state.get("_lgd_train_custom", ""),
        placeholder="/absolute/path/to/dataset.parquet",
        key="lgd_train_ds_custom",
    )
    if custom.strip():
        st.session_state["_lgd_train_custom"] = custom.strip()
        fp = dataset_fingerprint(custom.strip())
        label = portfolio_display_label(custom.strip())
        compat = assess_training_compatibility(custom.strip())
        if compat.get("error"):
            st.warning(f"Compatibility check unavailable: {compat['error']}")
            return custom.strip(), label, fp, False
        if not compat.get("lgd_trainable"):
            st.error(
                "This dataset is not compatible with LGD training. "
                f"Defaulted loans with valid LGD found: {compat.get('valid_lgd_rows', 0):,}. "
                "LGD training requires labeled default/loss observations. "
                "This fullbook dataset is suitable for Monte Carlo scoring/simulation, not LGD training."
            )
            return custom.strip(), label, fp, False
        st.success(
            f"LGD-training compatible dataset detected — "
            f"{compat.get('valid_lgd_rows', 0):,} defaulted loans with valid LGD available."
        )
        return custom.strip(), label, fp, True
    return None, "Default", None, False


# ---------------------------------------------------------------------------
# Launch section
# ---------------------------------------------------------------------------

def _render_launch_section() -> None:
    section_header("Train LGD Model")

    active = active_training_run("lgd")
    if active:
        _render_active_run(active)
        return

    with st.expander("Training Dataset", expanded=True):
        parquet_path, ds_label, fp, is_trainable = _render_dataset_selector_train()

    with st.form("lgd_train_form"):
        st.markdown("Click **Train** to launch LGD model training against the selected dataset.")
        st.caption(
            "Training includes: OLS + XGBoost, full validation, segment analysis, "
            "macro sensitivity, and artifact persistence."
        )
        submitted = st.form_submit_button("▶ Train LGD Model", type="primary")

    if submitted:
        if parquet_path is None:
            st.error("No valid dataset path. Select a dataset above.")
            return
        if not is_trainable:
            st.error(
                "Selected dataset is not LGD-trainable. "
                "Choose the labeled standard training dataset or another labeled custom parquet."
            )
            return
        if not Path(parquet_path).exists():
            st.error(f"Dataset path not found: `{parquet_path}`")
            return

        bundle_id = generate_bundle_id("lgd", fp or "unknown")
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
            st.error(f"Failed to launch LGD training: {exc}")
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
            model_type="lgd",
            pid=proc.pid,
            pid_create_time=pid_ct,
            launch_ts=datetime.now().isoformat(),
            log_path=str(log_path),
            bundle_dir=str(bundle_dir),
            dataset_path=parquet_path,
            dataset_label=ds_label,
            dataset_fingerprint=fp or "",
        )
        st.session_state.setdefault("lgd_procs", {})[bundle_id] = proc
        st.success(f"LGD training launched — bundle: `{bundle_id}`")
        st.rerun()


def _render_active_run(run: TrainingRunInfo) -> None:
    st.info(f"**LGD training in progress** — bundle `{run.bundle_id}`")
    if run.dataset_label:
        st.caption(f"Dataset: {run.dataset_label}")

    if run.log_path and Path(run.log_path).exists():
        with st.expander("Live log (last 30 lines)", expanded=True):
            try:
                lines = Path(run.log_path).read_text().splitlines()
                st.code("\n".join(lines[-30:]), language=None)
            except Exception:
                st.caption("Log not readable yet.")

    if st.button("↻ Refresh", key="lgd_refresh"):
        st.rerun()


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def _render_training_history() -> None:
    runs = discover_training_runs("lgd")
    if not runs:
        return
    with st.expander("Training history", expanded=False):
        for run in runs[:8]:
            icon = {"completed": "✅", "failed": "❌", "running": "⏳"}.get(run.status, "?")
            ts = (run.launch_ts or "")[:16]
            ds = run.dataset_label or "unknown dataset"
            st.markdown(f"{icon} `{run.bundle_id}` · {ds} · {ts}")


# ---------------------------------------------------------------------------
# Charts (bundle-aware)
# ---------------------------------------------------------------------------

def _render_charts(bundle_dir: Path) -> None:
    vd_path = str(bundle_dir / "lgd_validation_summary.csv")
    coef_path = str(bundle_dir / "lgd_ols_coefficients.csv")

    vd = _load_csv(vd_path)
    meta = read_bundle_metadata(bundle_dir)

    section_header("LGD Model Performance")
    k1, k2, k3, k4 = st.columns(4)
    if vd is not None:
        xgb_metrics_available = pd.notna(vd.loc["val_r2", "xgboost"]) and meta.get("xgboost_available", True) is not False
        xgb_error = meta.get("xgboost_error") or ""
        k1.metric("OLS Val R-sq", f"{vd.loc['val_r2','ols']:.4f}")
        if xgb_metrics_available:
            k2.metric("XGB Val R-sq", f"{vd.loc['val_r2','xgboost']:.4f}",
                      f"+{vd.loc['val_r2','xgboost']-vd.loc['val_r2','ols']:.4f}")
        else:
            k2.metric("XGB Val R-sq", "Unavailable")
        k3.metric("OLS Val RMSE", f"{vd.loc['val_rmse','ols']:.4f}")
        if not xgb_metrics_available and xgb_error:
            st.warning(
                "XGBoost challenger is unavailable for this bundle. "
                f"Runner recorded: `{xgb_error.splitlines()[0]}`"
            )
    else:
        k1.metric("OLS Val R-sq", "—")
        k2.metric("XGB Val R-sq", "—")
        k3.metric("OLS Val RMSE", "—")
        st.info(f"Validation summary not found in `{bundle_dir.name}`. Run LGD training first.")
        return

    k4.metric("Training Population", "Defaults only")

    info_box(
        "<strong>Is R-sq of 0.15 acceptable?</strong> Yes. LGD has high inherent variance from "
        "property-specific factors, foreclosure timelines, and market timing. "
        "Literature reports 0.10-0.25. What matters more: OLS calibration ratio ≈ 1.05."
    )

    tab_cmp, tab_coef, tab_meth = st.tabs(["Model Comparison", "OLS Coefficients", "Methodology"])

    with tab_cmp:
        cc, ce = st.columns([2, 1])
        with cc:
            metrics = ["train_r2", "val_r2", "test_r2", "train_rmse", "val_rmse", "test_rmse"]
            labels = ["Train R2", "Val R2", "Test R2", "Train RMSE", "Val RMSE", "Test RMSE"]
            ols_vals = [vd.loc[m, "ols"] for m in metrics]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=ols_vals, name="OLS",
                                 marker_color=COLORS["primary"],
                                 text=[f"{v:.4f}" for v in ols_vals], textposition="outside"))
            if xgb_metrics_available:
                xgb_vals = [vd.loc[m, "xgboost"] for m in metrics]
                fig.add_trace(go.Bar(x=labels, y=xgb_vals, name="XGBoost",
                                     marker_color=COLORS["success"],
                                     text=[f"{v:.4f}" for v in xgb_vals], textposition="outside"))
            fig.update_layout(
                title="LGD: OLS vs XGBoost" if xgb_metrics_available else "LGD: OLS Performance",
                barmode="group",
                bargap=0.2,
            )
            style_chart(fig, 440)
            st.plotly_chart(fig, use_container_width=True)
        with ce:
            if xgb_metrics_available:
                info_box("XGBoost captures ~10 more R-sq points through non-linear splits. "
                         "RMSE of 0.28 = predictions off ~28pp per loan, but portfolio mean is well-calibrated.")
            else:
                info_box("This bundle was validated with the OLS primary model only.")
                warning_box("XGBoost challenger metrics are unavailable for this bundle. "
                            "On macOS this is commonly caused by a missing OpenMP runtime (`libomp`).")

    with tab_coef:
        cc, ce = st.columns([2, 1])
        with cc:
            coef = _load_csv(coef_path)
            if coef is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=coef["coefficient"], y=coef["feature"], orientation="h",
                    marker_color=[COLORS["danger"] if c > 0 else COLORS["success"]
                                  for c in coef["coefficient"]],
                    text=[f"{v:+.6f}" for v in coef["coefficient"]], textposition="outside",
                ))
                fig.add_vline(x=0, line_color=COLORS.get("gray", "#94a3b8"))
                fig.update_layout(title="OLS LGD Coefficients", xaxis_title="Coefficient",
                                  yaxis=dict(autorange="reversed"), showlegend=False)
                style_chart(fig, 480)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coefficient data not available in this bundle.")
        with ce:
            info_box("<strong>Red (positive):</strong> increases LGD (more loss). "
                     "Higher LTV = less equity buffer.<br>"
                     "<strong>Green (negative):</strong> decreases LGD. "
                     "Mortgage insurance reduces loss by ~15pp.")

    with tab_meth:
        m1, m2 = st.columns(2)
        with m1:
            info_box("<strong>LGD Formula:</strong><br>Total Loss = EAD - Recovery + Costs<br>"
                     "LGD = Total Loss / EAD<br><br>"
                     "EAD from second-to-last observation.")
        with m2:
            info_box("<strong>Why OLS over Beta regression?</strong> Observed LGD can exceed 1.0 "
                     "(costs exceeding balance). Beta regression requires (0,1). "
                     "OLS with clipping is the pragmatic industry approach.")
        warning_box("<strong>Data note:</strong> net_sale_proceeds='C' (confidential) treated as NaN. "
                    "Treating as zero would artificially inflate LGD.")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("LGD Model: Loss Given Default")
    info_box(
        "<strong>LGD</strong> measures the fraction of exposure lost when a borrower defaults. "
        "An LGD of 0.40 = bank loses 40 cents per dollar. "
        "Second component of <strong>ECL = PD × LGD × EAD</strong>."
    )

    section_header("Model Bundle")
    bundle_dir, bundle_label, bundle_fp = _render_bundle_selector()

    if bundle_dir is not None:
        ds_info_parts = [f"**Bundle:** `{bundle_dir.name}`"]
        if bundle_fp:
            ds_info_parts.append(f"Dataset fingerprint: `{bundle_fp}`")
        st.caption("  ·  ".join(ds_info_parts))

    _render_training_history()

    st.markdown("---")
    _render_launch_section()
    st.markdown("---")

    if bundle_dir is not None:
        _render_charts(bundle_dir)
    else:
        st.info("No LGD model bundle available. Use the training section above to train one.")
