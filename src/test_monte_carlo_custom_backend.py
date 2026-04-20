"""
Test Monte Carlo Custom Backend
===============================

Validates deterministic CPU application-level parallelism for the
torch-backed Monte Carlo backend on small synthetic scored portfolios.
"""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from monte_carlo_custom_backend import (
    compute_risk_metrics,
    compute_scenario_losses,
    iter_scored_portfolio_chunks,
    run_monte_carlo,
)


passed = 0
failed = 0


def check(condition, test_name, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {test_name}")
    else:
        failed += 1
        print(f"  FAIL: {test_name}")
    if detail:
        print(f"        {detail}")


def build_macro_stats():
    return {
        "means": {
            "unemployment_rate": 4.3,
            "hpi_change_annual": 0.0,
            "gdp_growth_annual": 2.0,
        },
        "stds": {
            "unemployment_rate": 1.2,
            "hpi_change_annual": 5.0,
            "gdp_growth_annual": 2.0,
        },
        "correlation_matrix": np.array(
            [
                [1.0, -0.2, -0.4],
                [-0.2, 1.0, 0.3],
                [-0.4, 0.3, 1.0],
            ]
        ),
        "variable_names": [
            "unemployment_rate",
            "hpi_change_annual",
            "gdp_growth_annual",
        ],
    }


def make_scored_portfolio_df(n_rows=120):
    idx = np.arange(n_rows, dtype=np.float32)
    return pd.DataFrame(
        {
            "original_upb": 100_000.0 + idx * 150.0,
            "pd_baseline": np.clip(0.01 + idx * 0.0001, 0.01, 0.25),
            "lgd_baseline": np.clip(0.20 + (idx % 13) * 0.01, 0.2, 0.75),
            "pd_sensitivity": 0.9 + (idx % 7) * 0.05,
            "lgd_sensitivity": 0.85 + (idx % 5) * 0.04,
        }
    )


def write_scored_parquet(path, df, row_group_size=20):
    df.to_parquet(
        path,
        engine="pyarrow",
        index=False,
        row_group_size=row_group_size,
    )


def create_partitioned_dataset(root, df):
    part_a = root / "Property_State=AA" / "final"
    part_b = root / "Property_State=BB" / "final"
    part_a.mkdir(parents=True, exist_ok=True)
    part_b.mkdir(parents=True, exist_ok=True)
    midpoint = len(df) // 2
    write_scored_parquet(part_a / "data_0.parquet", df.iloc[:midpoint].reset_index(drop=True), row_group_size=15)
    write_scored_parquet(part_b / "data_0.parquet", df.iloc[midpoint:].reset_index(drop=True), row_group_size=15)


def test_parallel_matches_serial_run_monte_carlo(tmpdir):
    print("\n" + "=" * 70)
    print("TEST 1: Serial vs parallel run_monte_carlo")
    print("=" * 70)

    scored_path = tmpdir / "scored_portfolio.parquet"
    write_scored_parquet(scored_path, make_scored_portfolio_df(), row_group_size=20)
    macro_stats = build_macro_stats()

    serial_losses, _ = run_monte_carlo(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        macro_stats=macro_stats,
        n_simulations=64,
        random_seed=42,
        backend="cpu",
        dtype="float32",
        scenario_batch_size=16,
        loan_chunk_size=20,
        portfolio_chunks=iter_scored_portfolio_chunks(scored_path, batch_size=20),
        cpu_workers=1,
        torch_threads_per_worker=1,
        pyarrow_threads=1,
    )
    parallel_losses_2, _ = run_monte_carlo(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        macro_stats=macro_stats,
        n_simulations=64,
        random_seed=42,
        backend="cpu",
        dtype="float32",
        scenario_batch_size=16,
        loan_chunk_size=20,
        portfolio_path=scored_path,
        cpu_workers=2,
        torch_threads_per_worker=1,
        pyarrow_threads=1,
    )
    parallel_losses_3, _ = run_monte_carlo(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        macro_stats=macro_stats,
        n_simulations=64,
        random_seed=42,
        backend="cpu",
        dtype="float32",
        scenario_batch_size=16,
        loan_chunk_size=20,
        portfolio_path=scored_path,
        cpu_workers=3,
        torch_threads_per_worker=1,
        pyarrow_threads=1,
    )

    check(
        np.array_equal(serial_losses, parallel_losses_2),
        "Parallel CPU matches serial CPU with 2 workers",
        f"Max abs diff: {np.max(np.abs(serial_losses - parallel_losses_2)):.8f}",
    )
    check(
        np.array_equal(serial_losses, parallel_losses_3),
        "Parallel CPU matches serial CPU with 3 workers",
        f"Max abs diff: {np.max(np.abs(serial_losses - parallel_losses_3)):.8f}",
    )

    total_balance = float(make_scored_portfolio_df()["original_upb"].sum())
    serial_metrics = compute_risk_metrics(serial_losses, total_balance, backend="cpu", dtype="float32")
    parallel_metrics = compute_risk_metrics(parallel_losses_3, total_balance, backend="cpu", dtype="float32")
    check(
        serial_metrics == parallel_metrics,
        "Risk metrics match exactly between serial and parallel runs",
    )


def test_parallel_matches_serial_compute_scenario_losses(tmpdir):
    print("\n" + "=" * 70)
    print("TEST 2: Serial vs parallel compute_scenario_losses")
    print("=" * 70)

    scored_path = tmpdir / "scored_scenarios.parquet"
    write_scored_parquet(scored_path, make_scored_portfolio_df(n_rows=90), row_group_size=15)
    scenarios = pd.DataFrame(
        [
            {"unemployment_rate": 5.0, "hpi_change_annual": -5.0, "gdp_growth_annual": 1.0},
            {"unemployment_rate": 7.0, "hpi_change_annual": -10.0, "gdp_growth_annual": -2.0},
            {"unemployment_rate": 9.0, "hpi_change_annual": -20.0, "gdp_growth_annual": -4.0},
        ]
    )

    serial_losses, _ = compute_scenario_losses(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        scenarios=scenarios,
        backend="cpu",
        dtype="float32",
        scenario_batch_size=2,
        loan_chunk_size=15,
        portfolio_chunks=iter_scored_portfolio_chunks(scored_path, batch_size=15),
        cpu_workers=1,
        torch_threads_per_worker=1,
        pyarrow_threads=1,
    )
    parallel_losses, _ = compute_scenario_losses(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        scenarios=scenarios,
        backend="cpu",
        dtype="float32",
        scenario_batch_size=2,
        loan_chunk_size=15,
        portfolio_path=scored_path,
        cpu_workers=2,
        torch_threads_per_worker=1,
        pyarrow_threads=1,
    )

    check(
        np.array_equal(serial_losses, parallel_losses),
        "compute_scenario_losses matches between serial and parallel CPU",
        f"Max abs diff: {np.max(np.abs(serial_losses - parallel_losses)):.8f}",
    )


def test_nested_partitioned_dataset_parallel(tmpdir):
    print("\n" + "=" * 70)
    print("TEST 3: Nested partitioned dataset works in parallel mode")
    print("=" * 70)

    dataset_root = tmpdir / "parts"
    create_partitioned_dataset(dataset_root, make_scored_portfolio_df(n_rows=80))
    macro_stats = build_macro_stats()

    serial_losses, _ = run_monte_carlo(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        macro_stats=macro_stats,
        n_simulations=32,
        random_seed=7,
        backend="cpu",
        dtype="float32",
        scenario_batch_size=8,
        loan_chunk_size=15,
        portfolio_chunks=iter_scored_portfolio_chunks(dataset_root, batch_size=15),
        cpu_workers=1,
        torch_threads_per_worker=1,
        pyarrow_threads=1,
    )
    parallel_losses, _ = run_monte_carlo(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        macro_stats=macro_stats,
        n_simulations=32,
        random_seed=7,
        backend="cpu",
        dtype="float32",
        scenario_batch_size=8,
        loan_chunk_size=15,
        portfolio_path=dataset_root,
        cpu_workers=2,
        torch_threads_per_worker=1,
        pyarrow_threads=1,
    )

    check(
        np.array_equal(serial_losses, parallel_losses),
        "Nested partitioned parquet dataset matches in parallel CPU mode",
        f"Max abs diff: {np.max(np.abs(serial_losses - parallel_losses)):.8f}",
    )


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="mc_custom_backend_test_"))
    try:
        test_parallel_matches_serial_run_monte_carlo(tmpdir)
        test_parallel_matches_serial_compute_scenario_losses(tmpdir)
        test_nested_partitioned_dataset_parallel(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
