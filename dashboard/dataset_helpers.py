"""
Shared dataset helpers used by Portfolio Overview and Simulation Runs pages.

Provides preset discovery, parquet metadata inspection, and label resolution
without importing any Streamlit state.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def get_dataset_presets() -> dict:
    """
    Discover portfolio dataset presets from the repo's data/processed tree.

    Discovery order (no duplicates by resolved path):
    1. data/processed/loan_level_combined.parquet
    2. data/processed/fullbook/loan_level_combined.parquet
    3. Other *.parquet directly under data/processed/fullbook/ (not subdirs)
    4. Other *.parquet directly under data/processed/

    Subdirectories of fullbook/ (e.g. loan_level_state/) are intentionally
    skipped — they contain per-state shards, not standalone scoreable datasets.

    Returns:
        Ordered dict keyed by stable slug; each value has:
          "label"       — display string for the UI
          "path"        — absolute path string
          "description" — one-line description for the caption
    """
    data_dir = REPO_ROOT / "data" / "processed"
    presets: dict = {}
    seen_paths: set = set()

    def _rows_str(path: str) -> str:
        try:
            import pyarrow.parquet as pq
            m = pq.read_metadata(path)
            return f"{m.num_rows:,} rows"
        except Exception:
            return ""

    def _add(slug: str, label: str, path: Path, description: str) -> None:
        abs_path = str(path.resolve())
        if abs_path in seen_paths:
            return
        seen_paths.add(abs_path)
        presets[slug] = {"label": label, "path": abs_path, "description": description}

    std = data_dir / "loan_level_combined.parquet"
    if std.exists():
        rows = _rows_str(str(std))
        _add("standard", f"Standard · {std.name}", std,
             f"Labeled processed training dataset{' · ' + rows if rows else ''}")

    fb_dir = data_dir / "fullbook"
    fb_combined = fb_dir / "loan_level_combined.parquet"
    if fb_combined.exists():
        rows = _rows_str(str(fb_combined))
        _add("fullbook", f"Fullbook · {fb_combined.name}", fb_combined,
             f"Full-book scoring/simulation dataset{' · ' + rows if rows else ''}")

    if fb_dir.is_dir():
        for pf in sorted(fb_dir.glob("*.parquet")):
            if pf == fb_combined:
                continue
            rows = _rows_str(str(pf))
            slug = f"fullbook_{pf.stem}"
            _add(slug, f"Fullbook · {pf.stem}", pf,
                 f"Fullbook scoring/simulation variant{' · ' + rows if rows else ''}")

    for pf in sorted(data_dir.glob("*.parquet")):
        if pf == std:
            continue
        rows = _rows_str(str(pf))
        slug = f"processed_{pf.stem}"
        _add(slug, f"Processed · {pf.stem}", pf,
             f"Processed dataset{' · ' + rows if rows else ''}")

    return presets


def assess_training_compatibility(path: str) -> dict:
    """
    Return lightweight PD/LGD training compatibility information for a dataset.

    Uses pyarrow.dataset row-count filters so we can inspect very large parquet
    files/directories without loading the full dataset into pandas.
    """
    p = Path(path)
    result = {
        "path": str(p),
        "exists": p.exists(),
        "pd_trainable": False,
        "lgd_trainable": False,
        "positive_defaults": 0,
        "valid_lgd_rows": 0,
        "split_counts": {},
        "split_positive_defaults": {},
        "split_valid_lgd_rows": {},
        "error": None,
    }
    if not p.exists():
        result["error"] = "Path not found"
        return result

    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        result["error"] = f"pyarrow.dataset unavailable: {exc}"
        return result

    try:
        dataset = ds.dataset(str(p), format="parquet", partitioning="hive")
        schema_names = set(dataset.schema.names)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    if "default_flag" not in schema_names:
        result["error"] = "Missing required column `default_flag`"
        return result

    split_field = "data_split" if "data_split" in schema_names else None
    lgd_field = "lgd" if "lgd" in schema_names else None

    try:
        result["positive_defaults"] = int(
            dataset.scanner(filter=ds.field("default_flag") == 1).count_rows()
        )

        if lgd_field is not None:
            result["valid_lgd_rows"] = int(
                dataset.scanner(
                    filter=(ds.field("default_flag") == 1) & ds.field("lgd").is_valid()
                ).count_rows()
            )

        if split_field is not None:
            for split_name in ("train", "validation", "test"):
                total = int(
                    dataset.scanner(filter=ds.field("data_split") == split_name).count_rows()
                )
                positive = int(
                    dataset.scanner(
                        filter=(ds.field("data_split") == split_name)
                        & (ds.field("default_flag") == 1)
                    ).count_rows()
                )
                result["split_counts"][split_name] = total
                result["split_positive_defaults"][split_name] = positive
                if lgd_field is not None:
                    valid_lgd = int(
                        dataset.scanner(
                            filter=(ds.field("data_split") == split_name)
                            & (ds.field("default_flag") == 1)
                            & ds.field("lgd").is_valid()
                        ).count_rows()
                    )
                    result["split_valid_lgd_rows"][split_name] = valid_lgd

            result["pd_trainable"] = all(
                result["split_counts"].get(split, 0) > 0
                and 0 < result["split_positive_defaults"].get(split, 0) < result["split_counts"].get(split, 0)
                for split in ("train", "validation", "test")
            )
            if lgd_field is not None:
                result["lgd_trainable"] = all(
                    result["split_valid_lgd_rows"].get(split, 0) > 0
                    for split in ("train", "validation", "test")
                )
        else:
            result["pd_trainable"] = result["positive_defaults"] > 0
            result["lgd_trainable"] = result["valid_lgd_rows"] > 0
    except Exception as exc:
        result["error"] = str(exc)

    return result


def get_portfolio_meta(path: str) -> dict:
    """Return lightweight parquet metadata (no row reads) via pyarrow."""
    p = Path(path)
    result: dict = {
        "path": str(p), "exists": p.exists(),
        "is_file": False, "is_dir": False,
        "file_count": 0, "row_count": None, "row_group_count": None,
        "error": None,
    }
    if not p.exists():
        result["error"] = "Path not found"
        return result
    try:
        import pyarrow.parquet as pq
        if p.is_file():
            result["is_file"] = True
            result["file_count"] = 1
            m = pq.read_metadata(str(p))
            result["row_count"] = m.num_rows
            result["row_group_count"] = m.num_row_groups
        elif p.is_dir():
            result["is_dir"] = True
            pq_files = list(p.rglob("*.parquet"))
            result["file_count"] = len(pq_files)
            total_rows, total_rg = 0, 0
            for pf in pq_files[:30]:
                try:
                    fm = pq.read_metadata(str(pf))
                    total_rows += fm.num_rows
                    total_rg += fm.num_row_groups
                except Exception:
                    pass
            result["row_count"] = total_rows
            result["row_group_count"] = total_rg
    except Exception as exc:
        result["error"] = str(exc)
    return result


def portfolio_display_label(portfolio_path, portfolio_label=None) -> str:
    """Human-readable one-liner for a dataset path."""
    if portfolio_label:
        return portfolio_label
    if not portfolio_path:
        return "Default (auto-detected)"
    try:
        resolved = str(Path(portfolio_path).resolve())
        for info in get_dataset_presets().values():
            if info["path"] == resolved or info["path"] == portfolio_path:
                return info["label"]
    except Exception:
        pass
    return Path(portfolio_path).name
