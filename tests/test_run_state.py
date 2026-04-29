"""
Tests for dashboard/run_state.py
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure dashboard/ is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))

import run_state
from run_state import (
    RunInfo,
    _LEGACY_PREFIXES,
    _sidecar_path,
    discover_runs,
    is_alive,
    read_state,
    update_state,
    write_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_model_dir(tmp_path, monkeypatch):
    """Replace MODEL_DIR with a temp directory for isolation."""
    monkeypatch.setattr(run_state, "MODEL_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# write_state / read_state round-trip
# ---------------------------------------------------------------------------

def test_write_read_roundtrip(tmp_model_dir):
    write_state(
        prefix="mc_cpu_20260428_120000",
        pid=12345,
        pid_create_time=1714300000.0,
        launch_ts="2026-04-28T12:00:00",
        log_path=str(tmp_model_dir / "mc_cpu_20260428_120000.log"),
        config={"backend": "cpu", "n_simulations": 10000, "dtype": "float32"},
    )
    run = read_state("mc_cpu_20260428_120000")
    assert run is not None
    assert run.prefix == "mc_cpu_20260428_120000"
    assert run.pid == 12345
    assert run.pid_create_time == 1714300000.0
    assert run.status == "running"
    assert run.exit_code is None
    assert run.launch_ts == "2026-04-28T12:00:00"
    assert run.config["backend"] == "cpu"
    assert run.source == "ui"


def test_read_state_returns_none_when_missing(tmp_model_dir):
    result = read_state("mc_cpu_20260428_999999")
    assert result is None


def test_update_state_changes_status_and_exit_code(tmp_model_dir):
    write_state(
        prefix="mc_cpu_20260428_120001",
        pid=9999,
        pid_create_time=0.0,
        launch_ts="2026-04-28T12:00:01",
        log_path="",
        config={},
    )
    update_state("mc_cpu_20260428_120001", "completed", 0)
    run = read_state("mc_cpu_20260428_120001")
    assert run.status == "completed"
    assert run.exit_code == 0


# ---------------------------------------------------------------------------
# PID liveness
# ---------------------------------------------------------------------------

def test_is_alive_current_process():
    import psutil as _psutil
    pid = os.getpid()
    ct = _psutil.Process(pid).create_time()
    assert is_alive(pid, ct) is True


def test_is_alive_nonexistent_pid():
    # PID 99999999 very unlikely to exist
    assert is_alive(99999999, 0.0) is False


def test_is_alive_create_time_mismatch():
    # Use current PID but a wildly wrong create_time
    assert is_alive(os.getpid(), 0.0, tolerance=1.0) is False


def test_is_alive_none_pid():
    assert is_alive(None, 0.0) is False


# ---------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------

def test_discover_runs_returns_ui_and_cli(tmp_model_dir):
    # UI run: write sidecar + runtime_summary
    prefix_ui = "mc_cpu_20260428_130000"
    write_state(
        prefix=prefix_ui,
        pid=os.getpid(),
        pid_create_time=0.0,
        launch_ts="2026-04-28T13:00:00",
        log_path="",
        config={"backend": "cpu"},
    )
    (tmp_model_dir / f"{prefix_ui}_runtime_summary.csv").write_text(
        "backend,n_simulations\ncpu,100\n"
    )

    # CLI run: no sidecar, just runtime_summary
    prefix_cli = "mc_mps_20260428_120000"
    (tmp_model_dir / f"{prefix_cli}_runtime_summary.csv").write_text(
        "backend,n_simulations\nmps,100\n"
    )

    runs = discover_runs()
    prefixes = [r.prefix for r in runs]
    assert prefix_ui in prefixes
    assert prefix_cli in prefixes


def test_discover_runs_cli_source(tmp_model_dir):
    prefix = "mc_mps_20260428_110000"
    (tmp_model_dir / f"{prefix}_runtime_summary.csv").write_text(
        "backend,n_simulations\nmps,5000\n"
    )
    runs = discover_runs()
    cli_run = next(r for r in runs if r.prefix == prefix)
    assert cli_run.source == "cli"


def test_discover_runs_excludes_exact_legacy_prefixes_from_normal_list(tmp_model_dir):
    # Legacy bare prefixes should be included but flagged as legacy
    (tmp_model_dir / "mc_cpu_runtime_summary.csv").write_text("backend\ncpu\n")
    # Timestamped prefix must NOT be excluded
    (tmp_model_dir / "mc_cpu_20260428_100000_runtime_summary.csv").write_text(
        "backend\ncpu\n"
    )
    runs = discover_runs()
    prefixes = [r.prefix for r in runs]
    # Legacy prefix is still returned (shown with badge), not dropped
    assert "mc_cpu" in prefixes
    # Timestamped prefix is definitely returned
    assert "mc_cpu_20260428_100000" in prefixes


def test_discover_runs_timestamped_prefix_not_excluded(tmp_model_dir):
    """mc_cpu_20260428_100000 must NOT be excluded by the legacy check."""
    prefix = "mc_cpu_20260428_100000"
    (tmp_model_dir / f"{prefix}_runtime_summary.csv").write_text(
        "backend\ncpu\n"
    )
    runs = discover_runs()
    assert any(r.prefix == prefix for r in runs)


def test_discover_runs_sorted_newest_first(tmp_model_dir):
    for ts in ["20260428_090000", "20260428_110000", "20260428_100000"]:
        prefix = f"mc_cpu_{ts}"
        write_state(
            prefix=prefix, pid=1, pid_create_time=0.0,
            launch_ts=f"2026-04-28T{ts[:2]}:{ts[2:4]}:{ts[4:6]}",
            log_path="", config={},
        )
        (tmp_model_dir / f"{prefix}_runtime_summary.csv").write_text("x\n1\n")

    runs = discover_runs()
    timestamps = [r.launch_ts for r in runs if r.launch_ts]
    assert timestamps == sorted(timestamps, reverse=True)


def test_discover_runs_dead_process_marks_failed(tmp_model_dir):
    """If sidecar says running but PID is dead, discover_runs updates to failed."""
    prefix = "mc_cpu_20260428_140000"
    write_state(
        prefix=prefix, pid=99999999, pid_create_time=0.0,
        launch_ts="2026-04-28T14:00:00", log_path="", config={},
    )
    (tmp_model_dir / f"{prefix}_runtime_summary.csv").write_text("x\n1\n")

    runs = discover_runs()
    run = next(r for r in runs if r.prefix == prefix)
    assert run.status == "failed"
    assert run.exit_code == -1
