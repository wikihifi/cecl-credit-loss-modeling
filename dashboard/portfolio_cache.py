"""
Per-dataset analysis cache for the Portfolio Overview page.

Cache structure:
  models/portfolio_overview_cache/<16-hex-fingerprint>/
    metadata.json         — dataset identity + analysis provenance
    portfolio_totals.csv  — one row: total_loans, total_balance, default_rate, mean_fico
    fico_summary.csv      — by fico_bucket: count, defaults, balance, default_rate
    ltv_summary.csv       — by ltv_bucket: count, defaults, balance, default_rate
    vintage_summary.csv   — by origination_year: count, defaults, balance, default_rate

Fingerprint:
  SHA-256( resolved_path + mtime + size ) → first 16 hex chars.
  A change in file mtime or size produces a new key, forcing re-analysis.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
CACHE_ROOT = REPO_ROOT / "models" / "portfolio_overview_cache"

_REQUIRED_FILES = {
    "metadata.json",
    "portfolio_totals.csv",
    "fico_summary.csv",
    "ltv_summary.csv",
    "vintage_summary.csv",
}

# Ordered candidate columns for each dimension
_BALANCE_CANDIDATES = ["current_balance", "original_upb", "ead"]
_FICO_BUCKET_CANDIDATES = ["fico_bucket"]
_FICO_RAW_CANDIDATES = ["borrower_credit_score", "credit_score", "fico_score"]
_LTV_BUCKET_CANDIDATES = ["ltv_bucket"]
_LTV_RAW_CANDIDATES = ["original_ltv", "ltv", "current_ltv"]
_DEFAULT_CANDIDATES = ["default_flag", "is_default", "defaulted"]
_VINTAGE_CANDIDATES = ["origination_year"]

_FICO_BINS = [0, 620, 660, 700, 740, 851]
_FICO_LABELS = ["<620", "620-659", "660-699", "700-739", "740+"]

_LTV_BINS = [0, 60, 80, 90, 200]
_LTV_LABELS = ["<60", "60-80", "80-90", "90+"]


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def dataset_fingerprint(path_str: str) -> str:
    """
    Return a 16-char hex fingerprint for a dataset path.

    For files: uses resolved path + mtime + size.
    For directories: uses resolved path + sorted file names + their mtimes/sizes
    (capped at 100 files to stay fast on huge sharded datasets).
    """
    p = Path(path_str).resolve()
    parts: dict = {"path": str(p)}
    try:
        if p.is_file():
            s = p.stat()
            parts["mtime"] = round(s.st_mtime, 3)
            parts["size"] = s.st_size
        elif p.is_dir():
            pq_files = sorted(p.rglob("*.parquet"))[:100]
            files_meta = []
            for f in pq_files:
                try:
                    s = f.stat()
                    files_meta.append((f.name, s.st_size, round(s.st_mtime, 3)))
                except Exception:
                    files_meta.append((f.name, 0, 0.0))
            parts["files"] = files_meta
    except Exception:
        pass
    raw = json.dumps(parts, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache directory helpers
# ---------------------------------------------------------------------------

def get_cache_dir(fingerprint: str) -> Path:
    return CACHE_ROOT / fingerprint


def cache_is_complete(cache_dir: Path) -> bool:
    return all((cache_dir / f).exists() for f in _REQUIRED_FILES)


# ---------------------------------------------------------------------------
# Dataset schema introspection (metadata-only, no row reads)
# ---------------------------------------------------------------------------

def _schema_names(path_str: str) -> list:
    """Return column names from parquet metadata; empty list on failure."""
    try:
        import pyarrow.parquet as pq
        p = Path(path_str)
        if p.is_file():
            return list(pq.read_metadata(path_str).schema.names)
        if p.is_dir():
            pq_files = list(p.rglob("*.parquet"))
            if pq_files:
                return list(pq.read_metadata(str(pq_files[0])).schema.names)
    except Exception:
        pass
    return []


def _pick(candidates: list, available: set) -> Optional[str]:
    for c in candidates:
        if c in available:
            return c
    return None


# ---------------------------------------------------------------------------
# Dataset analysis — reads parquet, computes summaries, writes cache
# ---------------------------------------------------------------------------

def analyze_dataset(parquet_path: str, cache_dir: Path) -> tuple:
    """
    Read the dataset, compute summary artifacts, and persist to cache_dir.

    Returns:
        (success: bool, message: str)
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return False, "pyarrow is required for dataset analysis"

    p = Path(parquet_path)
    if not p.exists():
        return False, f"Path not found: {parquet_path}"

    avail = set(_schema_names(parquet_path))
    if not avail:
        return False, "Could not read parquet schema"

    # --- Column selection
    balance_col = _pick(_BALANCE_CANDIDATES, avail)
    fico_bucket_col = _pick(_FICO_BUCKET_CANDIDATES, avail)
    fico_raw_col = _pick(_FICO_RAW_CANDIDATES, avail)
    ltv_bucket_col = _pick(_LTV_BUCKET_CANDIDATES, avail)
    ltv_raw_col = _pick(_LTV_RAW_CANDIDATES, avail)
    default_col = _pick(_DEFAULT_CANDIDATES, avail)
    vintage_col = _pick(_VINTAGE_CANDIDATES, avail)

    needed: set = set()
    if default_col:
        needed.add(default_col)
    if balance_col:
        needed.add(balance_col)
    if fico_bucket_col:
        needed.add(fico_bucket_col)
    if fico_raw_col:
        needed.add(fico_raw_col)
    if ltv_bucket_col:
        needed.add(ltv_bucket_col)
    elif ltv_raw_col:
        needed.add(ltv_raw_col)
    if vintage_col:
        needed.add(vintage_col)

    if not needed:
        return False, "No usable columns found in dataset"

    needed_list = sorted(needed)

    # --- Read (column-pruned for speed)
    try:
        table = pq.read_table(parquet_path, columns=needed_list)
        df = table.to_pandas()
        del table
    except Exception as exc:
        return False, f"Failed to read dataset: {exc}"

    # --- Resolve default column
    def_series = None
    if default_col and default_col in df.columns:
        def_series = pd.to_numeric(df[default_col], errors="coerce").fillna(0)

    # --- Resolve balance
    bal_series = None
    if balance_col and balance_col in df.columns:
        bal_series = pd.to_numeric(df[balance_col], errors="coerce").fillna(0)

    # --- Resolve FICO bucket
    if fico_bucket_col and fico_bucket_col in df.columns:
        fico_bucket = df[fico_bucket_col].astype(str)
    elif fico_raw_col and fico_raw_col in df.columns:
        fico_bucket = pd.cut(
            pd.to_numeric(df[fico_raw_col], errors="coerce"),
            bins=_FICO_BINS, labels=_FICO_LABELS, right=False,
        ).astype(str)
    else:
        fico_bucket = None

    # --- Resolve LTV bucket
    if ltv_bucket_col and ltv_bucket_col in df.columns:
        ltv_bucket = df[ltv_bucket_col].astype(str)
    elif ltv_raw_col and ltv_raw_col in df.columns:
        ltv_bucket = pd.cut(
            pd.to_numeric(df[ltv_raw_col], errors="coerce"),
            bins=_LTV_BINS, labels=_LTV_LABELS, right=False,
        ).astype(str)
    else:
        ltv_bucket = None

    # --- Portfolio totals
    total_loans = len(df)
    total_balance = float(bal_series.sum()) if bal_series is not None else 0.0
    default_rate = float(def_series.mean()) if def_series is not None else float("nan")
    mean_fico = float("nan")
    if fico_raw_col and fico_raw_col in df.columns:
        raw_fico = pd.to_numeric(df[fico_raw_col], errors="coerce")
        mean_fico = float(raw_fico.mean()) if raw_fico.notna().any() else float("nan")

    totals_df = pd.DataFrame([{
        "total_loans": total_loans,
        "total_balance": total_balance,
        "default_rate": default_rate,
        "mean_fico": mean_fico,
    }])

    # --- FICO summary
    def _bucket_summary(bucket_series, label_col: str) -> Optional[pd.DataFrame]:
        if bucket_series is None:
            return None
        parts: dict = {label_col: bucket_series}
        if def_series is not None:
            parts["default_flag"] = def_series.values
        if bal_series is not None:
            parts["balance"] = bal_series.values
        tmp = pd.DataFrame(parts)
        agg: dict = {"count": (label_col, "count")}
        if "default_flag" in tmp.columns:
            agg["defaults"] = ("default_flag", "sum")
        if "balance" in tmp.columns:
            agg["balance"] = ("balance", "sum")
        grp = tmp.groupby(label_col, observed=True).agg(**agg).reset_index()
        if "defaults" in grp.columns and "count" in grp.columns:
            grp["default_rate"] = grp["defaults"] / grp["count"].replace(0, float("nan"))
        return grp

    fico_df = _bucket_summary(fico_bucket, "fico_bucket")
    ltv_df = _bucket_summary(ltv_bucket, "ltv_bucket")

    # --- Vintage summary
    vintage_df = None
    if vintage_col and vintage_col in df.columns:
        vint_parts: dict = {"origination_year": pd.to_numeric(df[vintage_col], errors="coerce")}
        if def_series is not None:
            vint_parts["default_flag"] = def_series.values
        if bal_series is not None:
            vint_parts["balance"] = bal_series.values
        vtmp = pd.DataFrame(vint_parts).dropna(subset=["origination_year"])
        vtmp["origination_year"] = vtmp["origination_year"].astype(int)
        agg: dict = {"count": ("origination_year", "count")}
        if "default_flag" in vtmp.columns:
            agg["defaults"] = ("default_flag", "sum")
        if "balance" in vtmp.columns:
            agg["balance"] = ("balance", "sum")
        vintage_df = vtmp.groupby("origination_year").agg(**agg).reset_index()
        if "defaults" in vintage_df.columns:
            vintage_df["default_rate"] = (
                vintage_df["defaults"] / vintage_df["count"].replace(0, float("nan"))
            )

    # --- Persist
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "parquet_path": str(p.resolve()),
        "fingerprint": cache_dir.name,
        "total_loans": total_loans,
        "total_balance": total_balance,
        "columns_used": {
            "balance": balance_col,
            "fico_bucket": fico_bucket_col or f"derived:{fico_raw_col}",
            "ltv_bucket": ltv_bucket_col or f"derived:{ltv_raw_col}",
            "default": default_col,
            "vintage": vintage_col,
        },
    }
    (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    totals_df.to_csv(cache_dir / "portfolio_totals.csv", index=False)

    placeholder_df = pd.DataFrame(columns=["fico_bucket", "count", "defaults", "balance", "default_rate"])
    fico_out = fico_df if fico_df is not None else placeholder_df
    fico_out.to_csv(cache_dir / "fico_summary.csv", index=False)

    placeholder_ltv = pd.DataFrame(columns=["ltv_bucket", "count", "defaults", "balance", "default_rate"])
    ltv_out = ltv_df if ltv_df is not None else placeholder_ltv
    ltv_out.to_csv(cache_dir / "ltv_summary.csv", index=False)

    placeholder_vint = pd.DataFrame(columns=["origination_year", "count", "defaults", "balance", "default_rate"])
    vint_out = vintage_df if vintage_df is not None else placeholder_vint
    vint_out.to_csv(cache_dir / "vintage_summary.csv", index=False)

    return True, "Analysis complete"


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(cache_dir: Path) -> dict:
    """
    Load all cached artifacts for a dataset.

    Returns:
        dict with keys: metadata, totals, fico, ltv, vintage.
        Values are None if the artifact is missing or unreadable.
    """
    def _read(name: str) -> Optional[pd.DataFrame]:
        p = cache_dir / name
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p)
            return df if not df.empty else None
        except Exception:
            return None

    meta = None
    meta_p = cache_dir / "metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            pass

    return {
        "metadata": meta,
        "totals": _read("portfolio_totals.csv"),
        "fico": _read("fico_summary.csv"),
        "ltv": _read("ltv_summary.csv"),
        "vintage": _read("vintage_summary.csv"),
    }


# ---------------------------------------------------------------------------
# Standard (default) dataset path
# ---------------------------------------------------------------------------

def default_dataset_path() -> Optional[str]:
    """Return the default dataset path if it exists, else None."""
    p = REPO_ROOT / "data" / "processed" / "loan_level_combined.parquet"
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Seed cache from existing dashboard_*.csv (one-time migration for default ds)
# ---------------------------------------------------------------------------

def seed_cache_from_legacy(cache_dir: Path) -> bool:
    """
    If models/dashboard_*.csv exist, use them to populate a cache entry for
    the default dataset without running a full re-analysis.

    Returns True if seeding succeeded.
    """
    model_dir = REPO_ROOT / "models"
    src_map = {
        "portfolio_totals.csv": model_dir / "dashboard_portfolio_totals.csv",
        "fico_summary.csv": model_dir / "dashboard_fico_summary.csv",
        "ltv_summary.csv": model_dir / "dashboard_ltv_summary.csv",
        "vintage_summary.csv": model_dir / "dashboard_vintage_summary.csv",
    }
    if not all(v.exists() for v in src_map.values()):
        return False

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import shutil
        for dst_name, src_path in src_map.items():
            shutil.copy2(src_path, cache_dir / dst_name)
        meta = {"source": "seeded_from_legacy_dashboard_csvs"}
        (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return True
    except Exception:
        return False
