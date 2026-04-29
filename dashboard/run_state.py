"""
Run state manager for the Simulation Runs screen.
Handles state file I/O, PID liveness, and run discovery.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

_SAFE_PREFIX_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

try:
    import psutil
except ImportError:
    psutil = None  # degrades to os.kill fallback; no create_time guard

MODEL_DIR = Path(__file__).parent.parent / "models"

# Prefixes produced by the old single-run CLI invocations (no timestamp component).
# These may exist in models/ but are excluded from auto-discovery because they
# may have been silently overwritten by subsequent CLI runs.
_LEGACY_PREFIXES = {"mc_cpu", "mc_mps"}


@dataclass
class RunInfo:
    prefix: str
    status: str                          # "running" | "completed" | "failed"
    source: str                          # "ui" | "cli"
    pid: Optional[int] = None
    pid_create_time: Optional[float] = None
    exit_code: Optional[int] = None
    launch_ts: Optional[str] = None      # ISO timestamp string
    log_path: Optional[str] = None
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State file I/O
# ---------------------------------------------------------------------------

def _sidecar_path(prefix: str) -> Path:
    return MODEL_DIR / f"{prefix}.run.json"


def write_state(
    prefix: str,
    pid: int,
    pid_create_time: float,
    launch_ts: str,
    log_path: str,
    config: dict,
) -> None:
    data = {
        "prefix": prefix,
        "pid": pid,
        "pid_create_time": pid_create_time,
        "status": "running",
        "exit_code": None,
        "launch_ts": launch_ts,
        "log_path": log_path,
        "config": config,
    }
    _atomic_write(_sidecar_path(prefix), data)


def update_state(prefix: str, status: str, exit_code: Optional[int]) -> None:
    path = _sidecar_path(prefix)
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"prefix": prefix}
    data["status"] = status
    data["exit_code"] = exit_code
    _atomic_write(path, data)


def read_state(prefix: str) -> Optional[RunInfo]:
    path = _sidecar_path(prefix)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None

    # Validate prefix from sidecar to prevent path traversal via crafted JSON
    raw_prefix = data.get("prefix", prefix)
    if not _SAFE_PREFIX_RE.match(raw_prefix):
        return None

    # Validate log_path is contained within MODEL_DIR to prevent path traversal
    raw_log_path = data.get("log_path")
    if raw_log_path is not None:
        try:
            resolved = Path(raw_log_path).resolve()
            if not resolved.is_relative_to(MODEL_DIR.resolve()):
                raw_log_path = None
        except (ValueError, OSError):
            raw_log_path = None

    return RunInfo(
        prefix=raw_prefix,
        status=data.get("status", "unknown"),
        source="ui",
        pid=data.get("pid"),
        pid_create_time=data.get("pid_create_time"),
        exit_code=data.get("exit_code"),
        launch_ts=data.get("launch_ts"),
        log_path=raw_log_path,
        config=data.get("config", {}),
    )


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.rename(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Artifact-based completion check
# ---------------------------------------------------------------------------

def is_run_complete(prefix: str) -> bool:
    """Return True if the run's expected output files are both present and non-empty."""
    summary = MODEL_DIR / f"{prefix}_runtime_summary.csv"
    risk = MODEL_DIR / f"{prefix}_risk_metrics.csv"
    return (
        summary.exists() and summary.stat().st_size > 0
        and risk.exists() and risk.stat().st_size > 0
    )


# ---------------------------------------------------------------------------
# PID liveness
# ---------------------------------------------------------------------------

def is_alive(pid: int, pid_create_time: float, tolerance: float = 5.0) -> bool:
    """Return True if pid is live and its create_time matches within tolerance."""
    if pid is None:
        return False
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            actual_ct = proc.create_time()
            return abs(actual_ct - pid_create_time) <= tolerance
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    else:
        # Fallback: existence check only, no create_time guard
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs() -> list[RunInfo]:
    """
    Glob models/ for all mc_*_runtime_summary.csv files, build RunInfos,
    sort newest-first.
    """
    runs: list[RunInfo] = []
    for csv_path in MODEL_DIR.glob("mc_*_runtime_summary.csv"):
        prefix = csv_path.name.replace("_runtime_summary.csv", "")
        if len(prefix) > 80 or not _SAFE_PREFIX_RE.match(prefix):
            continue
        if prefix in _LEGACY_PREFIXES:
            # Show legacy runs but mark them so the UI can badge them
            run = _build_cli_run(prefix, csv_path, legacy=True)
            if run:
                runs.append(run)
            continue

        sidecar = _sidecar_path(prefix)
        if sidecar.exists():
            run = read_state(prefix)
            if run is None:
                continue
            # Reconcile live status: if sidecar says running but process is gone,
            # prefer artifact evidence over PID state — the process may have
            # completed and exited before the poller captured its exit code.
            if run.status == "running" and not is_alive(run.pid, run.pid_create_time or 0.0):
                terminal = "completed" if is_run_complete(prefix) else "failed"
                exit_code = 0 if terminal == "completed" else -1
                update_state(prefix, terminal, exit_code)
                run.status = terminal
                run.exit_code = exit_code
        else:
            run = _build_cli_run(prefix, csv_path, legacy=False)
            if run is None:
                continue

        runs.append(run)

    # Sort: UI runs by launch_ts, CLI runs by file mtime, newest first
    def _sort_key(r: RunInfo):
        if r.launch_ts:
            return r.launch_ts
        csv_path = MODEL_DIR / f"{r.prefix}_runtime_summary.csv"
        try:
            return str(csv_path.stat().st_mtime)
        except OSError:
            return ""

    runs.sort(key=_sort_key, reverse=True)
    return runs


def _build_cli_run(prefix: str, csv_path: Path, *, legacy: bool) -> Optional[RunInfo]:
    """Build a RunInfo for a CLI-produced run (no state sidecar)."""
    risk_path = MODEL_DIR / f"{prefix}_risk_metrics.csv"
    config = {}
    if risk_path.exists():
        try:
            row = pd.read_csv(risk_path).iloc[0]
            for col in ("backend", "n_simulations", "dtype", "cpu_workers",
                        "torch_threads_per_worker", "pyarrow_threads",
                        "scenario_batch_size", "loan_chunk_size", "execution_mode"):
                if col in row.index:
                    val = row[col]
                    config[col] = None if pd.isna(val) else val
        except Exception:
            pass

    # Derive backend from prefix if not in CSV
    if "backend" not in config or not config["backend"]:
        parts = prefix.split("_")
        if len(parts) >= 2:
            config["backend"] = parts[1]

    status = "completed"  # csv_path always exists — it was the glob match
    if legacy:
        config["_legacy"] = True

    return RunInfo(
        prefix=prefix,
        status=status,
        source="cli",
        config=config,
    )
