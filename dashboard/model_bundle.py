"""
Model Bundle Management

A model bundle is a directory under models/model_bundles/ that contains
PD and/or LGD model artifacts trained against a specific dataset.

Bundle ID format:
  pd_<8-hex-dataset-fingerprint>_<YYYYMMDD_HHMMSS>   → PD artifacts
  lgd_<8-hex-dataset-fingerprint>_<YYYYMMDD_HHMMSS>  → LGD artifacts

models/model_bundles/<bundle_id>/
  metadata.json          — bundle_id, model_type, dataset_path, fingerprint, timestamp
  [PD artifacts]
    pd_logistic_regression.pkl
    woe_results.pkl
    selected_features.txt
    validation_summary.csv
    lr_coefficients.csv
    iv_summary.csv
    lr_calibration_validation.csv
  [LGD artifacts]
    lgd_ols.pkl
    lgd_features.txt
    lgd_validation_summary.csv
    lgd_ols_coefficients.csv

Backward-compatible "legacy" mode:
  If no bundle exists, the functions return MODEL_DIR itself as a
  fallback so existing global artifact paths keep working.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
MODEL_DIR = REPO_ROOT / "models"
BUNDLES_ROOT = MODEL_DIR / "model_bundles"

# Required file checks
_PD_REQUIRED = {"pd_logistic_regression.pkl", "woe_results.pkl", "selected_features.txt"}
_LGD_REQUIRED = {"lgd_ols.pkl", "lgd_features.txt"}

_PD_DISPLAY_FILES = [
    "validation_summary.csv",
    "lr_coefficients.csv",
    "iv_summary.csv",
    "lr_calibration_validation.csv",
]
_LGD_DISPLAY_FILES = [
    "lgd_validation_summary.csv",
    "lgd_ols_coefficients.csv",
]


# ---------------------------------------------------------------------------
# Bundle ID generation
# ---------------------------------------------------------------------------

def generate_bundle_id(model_type: str, dataset_fingerprint: str) -> str:
    """Return a timestamped bundle ID, e.g. pd_abc12345_20260430_143022."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{dataset_fingerprint[:8]}_{ts}"


# ---------------------------------------------------------------------------
# Bundle paths
# ---------------------------------------------------------------------------

def get_bundle_dir(bundle_id: str) -> Path:
    return BUNDLES_ROOT / bundle_id


# ---------------------------------------------------------------------------
# Bundle completeness checks
# ---------------------------------------------------------------------------

def pd_bundle_is_complete(bundle_dir: Path) -> bool:
    return all((bundle_dir / f).exists() for f in _PD_REQUIRED)


def lgd_bundle_is_complete(bundle_dir: Path) -> bool:
    return all((bundle_dir / f).exists() for f in _LGD_REQUIRED)


def bundle_has_pd(bundle_dir: Path) -> bool:
    return pd_bundle_is_complete(bundle_dir)


def bundle_has_lgd(bundle_dir: Path) -> bool:
    return lgd_bundle_is_complete(bundle_dir)


def bundle_is_mc_ready(bundle_dir: Path) -> bool:
    """Both PD and LGD must be present for Monte Carlo."""
    return pd_bundle_is_complete(bundle_dir) and lgd_bundle_is_complete(bundle_dir)


# ---------------------------------------------------------------------------
# Bundle metadata
# ---------------------------------------------------------------------------

def read_bundle_metadata(bundle_dir: Path) -> dict:
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def write_bundle_metadata(bundle_dir: Path, meta: dict) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Bundle discovery
# ---------------------------------------------------------------------------

def list_bundles(model_type: Optional[str] = None) -> list:
    """
    Return all bundles sorted newest-first.

    Each item is a dict: bundle_id, bundle_dir, metadata, has_pd, has_lgd,
    is_mc_ready, model_type (inferred from bundle_id prefix).
    """
    if not BUNDLES_ROOT.exists():
        return []
    results = []
    for d in sorted(BUNDLES_ROOT.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta = read_bundle_metadata(d)
        inferred_type = meta.get("model_type") or d.name.split("_")[0]
        if model_type and inferred_type != model_type:
            continue
        results.append({
            "bundle_id": d.name,
            "bundle_dir": d,
            "metadata": meta,
            "has_pd": bundle_has_pd(d),
            "has_lgd": bundle_has_lgd(d),
            "is_mc_ready": bundle_is_mc_ready(d),
            "model_type": inferred_type,
            "timestamp": meta.get("training_timestamp", ""),
            "dataset_fingerprint": meta.get("dataset_fingerprint", ""),
            "dataset_path": meta.get("dataset_path", ""),
            "dataset_label": meta.get("dataset_label", ""),
        })
    return results


def find_compatible_bundles(dataset_fingerprint: str, model_type: Optional[str] = None) -> list:
    """Return bundles whose dataset_fingerprint matches, newest-first."""
    return [
        b for b in list_bundles(model_type)
        if b["dataset_fingerprint"] == dataset_fingerprint
    ]


def latest_compatible_bundle(dataset_fingerprint: str, model_type: Optional[str] = None) -> Optional[dict]:
    """Return the newest bundle matching the fingerprint, or None."""
    bundles = find_compatible_bundles(dataset_fingerprint, model_type)
    return bundles[0] if bundles else None


def latest_mc_ready_bundle(dataset_fingerprint: str) -> Optional[dict]:
    """Return newest bundle that has both PD and LGD for the given fingerprint."""
    for b in find_compatible_bundles(dataset_fingerprint):
        if b["is_mc_ready"]:
            return b
    return None


# ---------------------------------------------------------------------------
# Legacy fallback
# ---------------------------------------------------------------------------

def legacy_pd_exists() -> bool:
    return (MODEL_DIR / "pd_logistic_regression.pkl").exists()


def legacy_lgd_exists() -> bool:
    return (MODEL_DIR / "lgd_ols.pkl").exists()


def legacy_is_mc_ready() -> bool:
    return legacy_pd_exists() and legacy_lgd_exists()


def legacy_bundle_info() -> dict:
    """Describe the legacy global model files as a pseudo-bundle."""
    return {
        "bundle_id": "legacy",
        "bundle_dir": MODEL_DIR,
        "metadata": {"source": "global_model_files", "model_type": "pd+lgd"},
        "has_pd": legacy_pd_exists(),
        "has_lgd": legacy_lgd_exists(),
        "is_mc_ready": legacy_is_mc_ready(),
        "model_type": "pd+lgd",
        "timestamp": "",
        "dataset_fingerprint": "",
        "dataset_path": "",
        "dataset_label": "Legacy (global model files)",
    }
