"""
Training run state manager for PD and LGD model training jobs.

State sidecars are written to:
  models/model_bundles/<bundle_id>.train.json

This is intentionally separate from run_state.py (which covers MC runs)
so training runs don't appear in the Simulation Runs history.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

from model_bundle import (
    BUNDLES_ROOT,
    MODEL_DIR,
    bundle_has_lgd,
    bundle_has_pd,
    bundle_is_mc_ready,
)

_SAFE_RE = __import__("re").compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class TrainingRunInfo:
    bundle_id: str
    model_type: str                   # "pd" | "lgd"
    status: str                       # "running" | "completed" | "failed"
    pid: Optional[int] = None
    pid_create_time: Optional[float] = None
    exit_code: Optional[int] = None
    launch_ts: Optional[str] = None
    log_path: Optional[str] = None
    bundle_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_label: Optional[str] = None
    dataset_fingerprint: Optional[str] = None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _sidecar_path(bundle_id: str) -> Path:
    BUNDLES_ROOT.mkdir(parents=True, exist_ok=True)
    return BUNDLES_ROOT / f"{bundle_id}.train.json"


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def write_training_state(
    bundle_id: str,
    model_type: str,
    pid: int,
    pid_create_time: float,
    launch_ts: str,
    log_path: str,
    bundle_dir: str,
    dataset_path: str,
    dataset_label: str,
    dataset_fingerprint: str,
) -> None:
    data = {
        "bundle_id": bundle_id,
        "model_type": model_type,
        "pid": pid,
        "pid_create_time": pid_create_time,
        "status": "running",
        "exit_code": None,
        "launch_ts": launch_ts,
        "log_path": log_path,
        "bundle_dir": bundle_dir,
        "dataset_path": dataset_path,
        "dataset_label": dataset_label,
        "dataset_fingerprint": dataset_fingerprint,
    }
    _atomic_write(_sidecar_path(bundle_id), data)


def update_training_state(bundle_id: str, status: str, exit_code: Optional[int]) -> None:
    path = _sidecar_path(bundle_id)
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"bundle_id": bundle_id}
    data["status"] = status
    data["exit_code"] = exit_code
    _atomic_write(path, data)


def read_training_state(bundle_id: str) -> Optional[TrainingRunInfo]:
    path = _sidecar_path(bundle_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    raw_id = data.get("bundle_id", bundle_id)
    if not _SAFE_RE.match(raw_id):
        return None
    return TrainingRunInfo(
        bundle_id=raw_id,
        model_type=data.get("model_type", "unknown"),
        status=data.get("status", "unknown"),
        pid=data.get("pid"),
        pid_create_time=data.get("pid_create_time"),
        exit_code=data.get("exit_code"),
        launch_ts=data.get("launch_ts"),
        log_path=data.get("log_path"),
        bundle_dir=data.get("bundle_dir"),
        dataset_path=data.get("dataset_path"),
        dataset_label=data.get("dataset_label"),
        dataset_fingerprint=data.get("dataset_fingerprint"),
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
# Liveness
# ---------------------------------------------------------------------------

def is_alive(pid: Optional[int], pid_create_time: Optional[float], tolerance: float = 5.0) -> bool:
    if pid is None:
        return False
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            return abs(proc.create_time() - (pid_create_time or 0.0)) <= tolerance
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Completion check (by bundle artifacts)
# ---------------------------------------------------------------------------

def training_is_complete(bundle_dir_str: str, model_type: str) -> bool:
    bd = Path(bundle_dir_str)
    if model_type == "pd":
        return bundle_has_pd(bd)
    if model_type == "lgd":
        return bundle_has_lgd(bd)
    return bundle_is_mc_ready(bd)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_training_runs(model_type: Optional[str] = None) -> list:
    """
    Return all known training runs, newest-first, with status reconciled.
    model_type: "pd", "lgd", or None (all).
    """
    if not BUNDLES_ROOT.exists():
        return []

    runs: list[TrainingRunInfo] = []
    for sidecar in sorted(BUNDLES_ROOT.glob("*.train.json"), reverse=True):
        bundle_id = sidecar.name[: -len(".train.json")]
        if not _SAFE_RE.match(bundle_id):
            continue
        run = read_training_state(bundle_id)
        if run is None:
            continue
        if model_type and run.model_type != model_type:
            continue

        # Reconcile running state
        if run.status == "running":
            if not is_alive(run.pid, run.pid_create_time):
                if run.bundle_dir and training_is_complete(run.bundle_dir, run.model_type):
                    update_training_state(bundle_id, "completed", 0)
                    run.status = "completed"
                else:
                    update_training_state(bundle_id, "failed", -1)
                    run.status = "failed"

        runs.append(run)
    return runs


def active_training_run(model_type: str) -> Optional[TrainingRunInfo]:
    """Return the currently running training job of the given type, or None."""
    for run in discover_training_runs(model_type):
        if run.status == "running":
            return run
    return None
