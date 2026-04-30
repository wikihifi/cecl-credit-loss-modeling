"""
Monte Carlo runner with selectable CPU, MPS, or CUDA backend.

This mirrors `src/run_monte_carlo.py` for data loading, baseline scoring, and
CSV outputs, but routes the Monte Carlo kernel through
`src/monte_carlo_custom_backend.py`.

Examples
--------
python src/run_monte_carlo_custom_backend.py --backend cpu
python src/run_monte_carlo_custom_backend.py --backend mps
python src/run_monte_carlo_custom_backend.py --backend cuda --n-simulations 50000

By default, outputs are written with a backend-specific prefix so multiple runs
can coexist, e.g.:
  - models/mc_cpu_loss_distribution.csv
  - models/mc_mps_loss_distribution.csv
  - models/mc_cuda_loss_distribution.csv

If you want the generated results to replace the dashboard's standard Monte
Carlo files, also pass `--write-standard-files`.
"""

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pd_model import apply_woe_transformation
from loan_specific_multipliers import build_loan_specific_sensitivities
from monte_carlo_custom_backend import (
    build_scored_portfolio_work_items,
    configure_pyarrow_threads,
    configure_torch_cpu_threads,
    compute_historical_macro_stats,
    compute_risk_metrics,
    compute_scenario_losses,
    get_execution_mode,
    iter_scored_portfolio_chunks,
    resolve_cpu_parallelism,
    run_monte_carlo,
    summarize_portfolio_path,
)


COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
PARTITIONED_DATASET_PATH = project_root / "data" / "parts"
MACRO_PATH = project_root / "data" / "processed" / "macro" / "fred_macro_monthly.csv"
MODEL_DIR = project_root / "models"
SENSITIVITY_REQUIRED_COLUMNS = {
    "original_upb",
    "data_split",
    "default_flag",
    "lgd",
    "fico_bucket",
    "borrower_credit_score",
    "ltv_bucket",
    "original_ltv",
    "fico_missing",
    "has_mortgage_insurance",
    "msa",
    "property_state",
    "is_cashout_refi",
    "loan_purpose",
    "is_refi_nocashout",
    "is_investment_property",
    "occupancy_status",
    "is_second_home",
    "is_condo",
    "property_type",
    "is_manufactured_housing",
    "is_multi_unit",
    "number_of_units",
    "amortization_type",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo with a selectable CPU, MPS, or CUDA backend."
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help="Execution backend for the Monte Carlo kernel.",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo scenarios to run.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
        help="Torch dtype for the backend-aware Monte Carlo kernel.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Prefix for output files written to models/. "
            "Default is mc_<backend>."
        ),
    )
    parser.add_argument(
        "--write-standard-files",
        action="store_true",
        help=(
            "Also write the standard dashboard Monte Carlo filenames "
            "(mc_loss_distribution.csv, mc_scenarios.csv, mc_risk_metrics.csv, "
            "mc_sensitivity.csv)."
        ),
    )
    parser.add_argument(
        "--scenario-batch-size",
        type=int,
        default=128,
        help="Scenarios per batch for loan-level aggregation.",
    )
    parser.add_argument(
        "--loan-chunk-size",
        type=int,
        default=100_000,
        help="Loans per chunk for loan-level aggregation.",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="CPU worker processes for deterministic application-level parallelism.",
    )
    parser.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=None,
        help="Torch CPU threads to use inside each CPU worker process.",
    )
    parser.add_argument(
        "--pyarrow-threads",
        type=int,
        default=1,
        help="PyArrow CPU threads per process for parquet scanning and row-group reads.",
    )
    parser.add_argument(
        "--portfolio-path",
        default=None,
        help=(
            "Optional path to the engineered portfolio parquet or parquet dataset "
            "to score. Overrides the default search order."
        ),
    )
    parser.add_argument(
        "--antithetic-variates",
        action="store_true",
        help=(
            "Use antithetic variates for variance reduction. "
            "Draws n/2 random shocks and pairs each with its negative, "
            "reducing estimator variance without extra model evaluations."
        ),
    )
    parser.add_argument(
        "--adaptive-stopping",
        action="store_true",
        help=(
            "Stop early when key risk metrics have converged within --convergence-tolerance "
            "across two consecutive simulation batches. Requires --simulation-batch-size."
        ),
    )
    parser.add_argument(
        "--simulation-batch-size",
        type=int,
        default=10_000,
        help=(
            "Simulations per adaptive-stopping batch. Also used as the minimum "
            "granularity when --adaptive-stopping is active. Default: 10000."
        ),
    )
    parser.add_argument(
        "--convergence-tolerance",
        type=float,
        default=0.01,
        help=(
            "Relative change threshold for adaptive stopping. "
            "A batch is considered converged when all tracked metrics change by "
            "less than this fraction between consecutive batches. Default: 0.01 (1%%)."
        ),
    )
    parser.add_argument(
        "--min-simulations",
        type=int,
        default=0,
        help=(
            "Minimum simulations to run before adaptive stopping is allowed. "
            "Defaults to one simulation batch if 0."
        ),
    )
    parser.add_argument(
        "--max-simulations",
        type=int,
        default=0,
        help=(
            "Maximum simulations when adaptive stopping is active. "
            "Defaults to --n-simulations if 0."
        ),
    )
    parser.add_argument(
        "--convergence-metrics",
        nargs="+",
        choices=["expected_loss", "var_99", "var_999", "es_99"],
        default=["expected_loss", "var_99"],
        help=(
            "Risk metrics to monitor for adaptive stopping convergence. "
            "Default: expected_loss var_99."
        ),
    )
    parser.add_argument(
        "--pd-bundle-dir",
        default=None,
        help=(
            "Path to a model bundle directory containing PD artifacts "
            "(pd_logistic_regression.pkl, woe_results.pkl, selected_features.txt). "
            "If not provided, loads from the global models/ directory."
        ),
    )
    parser.add_argument(
        "--lgd-bundle-dir",
        default=None,
        help=(
            "Path to a model bundle directory containing LGD artifacts "
            "(lgd_ols.pkl, lgd_features.txt). "
            "If not provided, loads from the global models/ directory."
        ),
    )
    return parser.parse_args()


def resolve_macro_path():
    macro_path = MACRO_PATH
    if not Path(macro_path).exists():
        alt_path = project_root / "data" / "macro" / "fred_macro_monthly.csv"
        if alt_path.exists():
            macro_path = alt_path
        else:
            raise FileNotFoundError(
                f"Macro data not found at {MACRO_PATH} or {alt_path}"
            )
    return macro_path


def resolve_portfolio_path(portfolio_path_override=None):
    """Resolve the portfolio source, honoring an explicit override when provided."""
    if portfolio_path_override:
        portfolio_path = Path(portfolio_path_override)
        if portfolio_path.exists():
            return portfolio_path
        raise FileNotFoundError(f"Portfolio path not found: {portfolio_path}")

    if PARTITIONED_DATASET_PATH.exists():
        return PARTITIONED_DATASET_PATH
    if COMBINED_PATH.exists():
        return COMBINED_PATH
    raise FileNotFoundError(
        f"Portfolio data not found at {PARTITIONED_DATASET_PATH} or {COMBINED_PATH}"
    )


def read_portfolio_parquet(parquet_path, columns=None):
    """
    Read either a single parquet file or a partitioned parquet dataset.

    Uses a dataset scan for directory inputs so nested partition folders work
    regardless of depth.
    """
    parquet_path = Path(parquet_path)
    if parquet_path.is_file():
        return pd.read_parquet(parquet_path, columns=columns)

    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError(
            "Reading a partitioned parquet dataset requires pyarrow.dataset."
        ) from exc

    dataset = ds.dataset(parquet_path, format="parquet", partitioning="hive")
    table = dataset.to_table(columns=columns)
    return table.to_pandas()


def list_portfolio_files(parquet_path):
    """List parquet files for lightweight dataset logging."""
    parquet_path = Path(parquet_path)
    if parquet_path.is_file():
        return [parquet_path]
    return sorted(path for path in parquet_path.rglob("*.parquet") if path.is_file())


def get_portfolio_schema_columns(parquet_path):
    """Read parquet schema column names without loading the dataset into pandas."""
    parquet_path = Path(parquet_path)
    if parquet_path.is_file():
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading parquet schema requires pyarrow.") from exc
        return pq.ParquetFile(parquet_path).schema.names

    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError("Reading parquet schema requires pyarrow.dataset.") from exc

    dataset = ds.dataset(parquet_path, format="parquet", partitioning="hive")
    return dataset.schema.names


def build_step1_required_columns(pd_features, lgd_features):
    """Build the minimum column set needed for Step 1 scoring and sensitivity calibration."""
    return sorted(set(pd_features) | set(lgd_features) | SENSITIVITY_REQUIRED_COLUMNS)


def resolve_compatible_portfolio_path(required_columns, portfolio_path_override=None):
    """
    Choose a portfolio source that actually contains the engineered columns
    needed by the Monte Carlo runner.
    """
    candidate_paths = []
    if portfolio_path_override:
        candidate_paths.append(resolve_portfolio_path(portfolio_path_override))
    else:
        if PARTITIONED_DATASET_PATH.exists():
            candidate_paths.append(PARTITIONED_DATASET_PATH)
        if COMBINED_PATH.exists():
            candidate_paths.append(COMBINED_PATH)

    compatibility_rows = []
    for candidate in candidate_paths:
        available_columns = set(get_portfolio_schema_columns(candidate))
        matched = sorted(set(required_columns) & available_columns)
        missing = sorted(set(required_columns) - available_columns)
        compatibility_rows.append(
            {
                "path": candidate,
                "matched_columns": matched,
                "missing_columns": missing,
            }
        )

    if not compatibility_rows:
        if portfolio_path_override:
            raise FileNotFoundError(f"Portfolio path not found: {portfolio_path_override}")
        raise FileNotFoundError(
            f"Portfolio data not found at {PARTITIONED_DATASET_PATH} or {COMBINED_PATH}"
        )

    best_match = max(compatibility_rows, key=lambda row: len(row["matched_columns"]))
    if "original_upb" in best_match["matched_columns"]:
        return best_match["path"], best_match["matched_columns"], best_match["missing_columns"]

    missing_preview = ", ".join(best_match["missing_columns"][:12])
    suffix = " ..." if len(best_match["missing_columns"]) > 12 else ""
    raise ValueError(
        "No compatible engineered portfolio dataset was found for Monte Carlo scoring. "
        f"Best candidate was {best_match['path']} but it is missing required columns such as: "
        f"{missing_preview}{suffix}. "
        "The runner expects the engineered loan-level dataset, not the raw partitioned source."
    )


def save_outputs(output_prefix, losses, scenarios, metrics, sensitivity_df, loan_sensitivity_summary):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    loss_dist_path = MODEL_DIR / f"{output_prefix}_loss_distribution.csv"
    scenarios_path = MODEL_DIR / f"{output_prefix}_scenarios.csv"
    metrics_path = MODEL_DIR / f"{output_prefix}_risk_metrics.csv"
    sensitivity_path = MODEL_DIR / f"{output_prefix}_sensitivity.csv"
    loan_sens_path = MODEL_DIR / f"{output_prefix}_loan_sensitivity_summary.csv"

    loss_dist = pd.DataFrame({
        "simulation_id": range(len(losses)),
        "portfolio_loss": losses,
        "loss_rate": scenarios["loss_rate"].values,
    })
    loss_dist.to_csv(loss_dist_path, index=False)
    scenarios.to_csv(scenarios_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    sensitivity_df.to_csv(sensitivity_path, index=False)
    loan_sensitivity_summary.to_csv(loan_sens_path, index=False)

    return {
        "loss_distribution": loss_dist_path,
        "scenarios": scenarios_path,
        "risk_metrics": metrics_path,
        "sensitivity": sensitivity_path,
        "loan_sensitivity_summary": loan_sens_path,
    }


def write_standard_files(losses, scenarios, metrics, sensitivity_df):
    loss_dist = pd.DataFrame({
        "simulation_id": range(len(losses)),
        "portfolio_loss": losses,
        "loss_rate": scenarios["loss_rate"].values,
    })
    loss_dist.to_csv(MODEL_DIR / "mc_loss_distribution.csv", index=False)
    scenarios.to_csv(MODEL_DIR / "mc_scenarios.csv", index=False)
    pd.DataFrame([metrics]).to_csv(MODEL_DIR / "mc_risk_metrics.csv", index=False)
    sensitivity_df.to_csv(MODEL_DIR / "mc_sensitivity.csv", index=False)


def write_scored_portfolio_parquet(
    output_path,
    portfolio_upb,
    pd_baseline,
    lgd_baseline,
    pd_sensitivity,
    lgd_sensitivity,
    row_group_size,
):
    """Persist the compact Monte Carlo inputs so later steps can stream them."""
    scored_portfolio = pd.DataFrame({
        "original_upb": np.asarray(portfolio_upb, dtype=np.float32),
        "pd_baseline": np.asarray(pd_baseline, dtype=np.float32),
        "lgd_baseline": np.asarray(lgd_baseline, dtype=np.float32),
        "pd_sensitivity": np.asarray(pd_sensitivity, dtype=np.float32),
        "lgd_sensitivity": np.asarray(lgd_sensitivity, dtype=np.float32),
    })
    scored_portfolio.to_parquet(
        output_path,
        engine="pyarrow",
        index=False,
        row_group_size=row_group_size,
    )
    return output_path


def log_series_quality(name, values):
    """Print null/non-finite diagnostics for a 1D array-like input."""
    arr = np.asarray(values)
    null_mask = pd.isna(arr)
    null_count = int(null_mask.sum())

    finite_mask = np.ones(arr.shape, dtype=bool)
    try:
        numeric_arr = arr.astype(float)
        finite_mask = np.isfinite(numeric_arr)
    except (TypeError, ValueError):
        pass

    non_finite_count = int((~finite_mask & ~null_mask).sum())
    print(
        f"    {name}: nulls={null_count:,}, non_finite={non_finite_count:,}, "
        f"rows={len(arr):,}"
    )


def build_valid_scoring_mask(
    portfolio_upb,
    pd_baseline,
    lgd_baseline,
    pd_sensitivity,
    lgd_sensitivity,
):
    """Return the rows that are safe to pass into Monte Carlo aggregation."""
    masks = []
    for values in [
        portfolio_upb,
        pd_baseline,
        lgd_baseline,
        pd_sensitivity,
        lgd_sensitivity,
    ]:
        arr = np.asarray(values, dtype=float)
        masks.append(np.isfinite(arr))
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask & mask
    return combined_mask


def sensitivity_analysis_custom(
    scored_portfolio_path,
    total_balance,
    baseline_loss,
    macro_stats,
    backend,
    dtype,
    scenario_batch_size,
    loan_chunk_size,
    cpu_workers,
    torch_threads_per_worker,
    pyarrow_threads,
):
    """One-at-a-time sensitivity analysis using loan-specific multipliers."""
    variables_to_shock = {
        "unemployment_rate": [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0],
        "hpi_change_annual": [10.0, 5.0, 0.0, -5.0, -10.0, -20.0, -30.0],
        "gdp_growth_annual": [4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -8.0],
    }

    print("\n  Sensitivity Analysis (one variable at a time):")
    results = []

    for var_name, shock_values in variables_to_shock.items():
        scenario_rows = []
        for value in shock_values:
            row = {
                v: macro_stats["means"][v]
                for v in macro_stats["variable_names"]
                if v in macro_stats["means"]
            }
            row[var_name] = value
            scenario_rows.append(row)

        scenarios_df = pd.DataFrame(scenario_rows)
        losses, scored = compute_scenario_losses(
            portfolio_upb=None,
            pd_baseline=None,
            lgd_baseline=None,
            scenarios=scenarios_df,
            portfolio_path=scored_portfolio_path,
            portfolio_chunks=iter_scored_portfolio_chunks(
                scored_portfolio_path,
                batch_size=loan_chunk_size,
            ),
            backend=backend,
            dtype=dtype,
            scenario_batch_size=scenario_batch_size,
            loan_chunk_size=loan_chunk_size,
            cpu_workers=cpu_workers,
            torch_threads_per_worker=torch_threads_per_worker,
            pyarrow_threads=pyarrow_threads,
        )

        print(f"\n  {var_name}:")
        print(f"  {'Value':>10s} {'PD Mult':>8s} {'LGD Mult':>9s} "
              f"{'Loss ($M)':>12s} {'Loss Rate':>10s} {'vs Base':>10s}")
        print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*12} {'-'*10} {'-'*10}")

        for _, row in scored.iterrows():
            vs_base = (
                row["portfolio_loss"] / baseline_loss
                if baseline_loss > 0 and np.isfinite(baseline_loss)
                else np.nan
            )
            print(
                f"  {row[var_name]:>10.1f} {row['pd_multiplier']:>8.2f}x {row['lgd_multiplier']:>9.2f}x "
                f"${row['portfolio_loss']/1e6:>11,.0f} {row['loss_rate']*100:>9.2f}% {vs_base:>10.2f}x"
            )
            results.append({
                "variable": var_name,
                "value": row[var_name],
                "pd_multiplier": row["pd_multiplier"],
                "lgd_multiplier": row["lgd_multiplier"],
                "portfolio_loss": row["portfolio_loss"],
                "loss_rate": row["loss_rate"],
                "vs_base": vs_base,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Adaptive stopping helpers
# ---------------------------------------------------------------------------

def _quick_risk_metrics(losses: np.ndarray) -> dict:
    """Compute EL, VaR99, VaR99.9, ES99 from a numpy loss array (no GPU)."""
    el = float(np.mean(losses))
    var99 = float(np.quantile(losses, 0.99))
    var999 = float(np.quantile(losses, 0.999))
    tail_mask = losses >= var99
    es99 = float(np.mean(losses[tail_mask])) if tail_mask.any() else var99
    return {"expected_loss": el, "var_99": var99, "var_999": var999, "es_99": es99}


def _is_converged(
    prev: dict,
    curr: dict,
    metric_names: list,
    tolerance: float,
) -> bool:
    """Return True when every tracked metric's relative change is below tolerance."""
    for name in metric_names:
        denom = abs(prev[name])
        if denom < 1.0:
            denom = 1.0
        if abs(curr[name] - prev[name]) / denom > tolerance:
            return False
    return True


def main():
    args = parse_args()
    output_prefix = args.output_prefix or f"mc_{args.backend}"
    cpu_parallelism = resolve_cpu_parallelism(
        backend=args.backend,
        cpu_workers=args.cpu_workers,
        torch_threads_per_worker=args.torch_threads_per_worker,
        pyarrow_threads=args.pyarrow_threads,
    )
    execution_mode = get_execution_mode(
        backend=args.backend,
        cpu_workers=cpu_parallelism["cpu_workers"],
        portfolio_path="streamed_scored_portfolio",
    )

    if args.backend == "cpu":
        parent_torch_threads = (
            1 if cpu_parallelism["cpu_workers"] > 1
            else cpu_parallelism["torch_threads_per_worker"]
        )
        configure_torch_cpu_threads(parent_torch_threads)
        configure_pyarrow_threads(cpu_parallelism["pyarrow_threads"])

    t_total_start = time.time()
    timings = {}

    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - MONTE CARLO CUSTOM BACKEND")
    print("=" * 70)
    print(f"Backend: {args.backend}")
    print(f"Simulations: {args.n_simulations:,}")
    print(f"Dtype: {args.dtype}")
    print(f"Random seed: {args.random_seed}")
    print(f"Scenario batch size: {args.scenario_batch_size:,}")
    print(f"Loan chunk size: {args.loan_chunk_size:,}")
    print(f"Execution mode: {execution_mode}")
    print(f"Antithetic variates: {args.antithetic_variates}")
    if args.adaptive_stopping:
        max_sims = args.max_simulations or args.n_simulations
        min_sims = args.min_simulations or args.simulation_batch_size
        print(
            f"Adaptive stopping: enabled "
            f"(batch={args.simulation_batch_size:,}, tol={args.convergence_tolerance}, "
            f"min={min_sims:,}, max={max_sims:,}, metrics={args.convergence_metrics})"
        )
    if args.backend == "cpu":
        print(f"CPU workers: {cpu_parallelism['cpu_workers']}")
        print(f"Torch threads/worker: {cpu_parallelism['torch_threads_per_worker']}")
        print(f"PyArrow threads/process: {cpu_parallelism['pyarrow_threads']}")
    print(f"Output prefix: {output_prefix}")

    # ------------------------------------------------------------------
    # Step 1: Load portfolio and score baseline PD/LGD
    # ------------------------------------------------------------------
    step_start = time.time()
    print("\nStep 1: Loading portfolio and scoring baseline...")

    pd_dir = Path(args.pd_bundle_dir) if args.pd_bundle_dir else MODEL_DIR
    lgd_dir = Path(args.lgd_bundle_dir) if args.lgd_bundle_dir else MODEL_DIR
    print(f"  PD artifacts: {pd_dir}")
    print(f"  LGD artifacts: {lgd_dir}")

    pd_model = joblib.load(pd_dir / "pd_logistic_regression.pkl")
    woe_results = joblib.load(pd_dir / "woe_results.pkl")
    with open(pd_dir / "selected_features.txt") as f:
        pd_features = [line.strip() for line in f if line.strip()]

    lgd_model = joblib.load(lgd_dir / "lgd_ols.pkl")
    with open(lgd_dir / "lgd_features.txt") as f:
        lgd_features = [line.strip() for line in f if line.strip()]

    required_columns = build_step1_required_columns(pd_features, lgd_features)
    portfolio_path, selected_columns, missing_columns = resolve_compatible_portfolio_path(
        required_columns,
        portfolio_path_override=args.portfolio_path,
    )
    print(f"  Portfolio source: {portfolio_path}")
    portfolio_files = list_portfolio_files(portfolio_path)

    print(f"  Portfolio parquet files: {len(portfolio_files):,}")
    print(f"  Step 1 requested columns: {len(required_columns):,}")
    print(f"  Step 1 selected columns:  {len(selected_columns):,}")
    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        suffix = " ..." if len(missing_columns) > 10 else ""
        print(f"  Missing optional columns: {preview}{suffix}")

    load_start = time.time()
    print("  Loading reduced portfolio frame into pandas...")
    df = read_portfolio_parquet(portfolio_path, columns=selected_columns)
    load_elapsed = time.time() - load_start
    df_memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(
        f"  Loaded {len(df):,} rows x {len(df.columns):,} columns "
        f"in {load_elapsed:.1f}s ({df_memory_mb:,.0f} MB in memory)"
    )
    if "data_split" in df.columns:
        df.loc[df["data_split"] == "unknown", "data_split"] = "train"
    else:
        print("  Column 'data_split' not found; defaulting all rows to 'train'.")
        df["data_split"] = "train"
    n_loans = len(df)
    total_balance = df["original_upb"].sum()
    portfolio_upb = df["original_upb"].values.astype(float)
    print(f"  Portfolio: {n_loans:,} loans, ${total_balance/1e9:.1f}B")

    print(f"  Applying WoE transform for {len(pd_features):,} PD features...")
    X_woe = apply_woe_transformation(df, woe_results, pd_features)
    print(f"  Scoring PD model on matrix shape {X_woe.shape}...")
    pd_baseline = pd_model.predict_proba(X_woe)[:, 1]
    del X_woe
    gc.collect()

    print(f"  Preparing LGD features for {len(lgd_features):,} columns...")
    X_lgd = df[lgd_features].copy()
    lgd_fill = {"loan_age_at_default": 48.0, "was_modified": 0.0}
    for col in lgd_features:
        if col in lgd_fill:
            X_lgd[col] = X_lgd[col].fillna(lgd_fill[col])
        else:
            X_lgd[col] = X_lgd[col].fillna(X_lgd[col].median())
    print(f"  Scoring LGD model on matrix shape {X_lgd.shape}...")
    lgd_baseline = lgd_model.predict(X_lgd)
    lgd_baseline = np.clip(lgd_baseline, 0.0, 1.0)
    del X_lgd
    gc.collect()

    baseline_el = (pd_baseline * lgd_baseline * portfolio_upb).sum()
    print(f"  Baseline PD: {pd_baseline.mean()*100:.2f}%")
    print(f"  Baseline LGD: {lgd_baseline.mean()*100:.2f}%")
    print(f"  Baseline Annual EL: ${baseline_el/1e6:,.0f}M "
          f"({baseline_el/total_balance*100:.2f}%)")
    timings["baseline_scoring_seconds"] = time.time() - step_start

    # ------------------------------------------------------------------
    # Step 2: Build loan-specific sensitivities
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print("Step 2: Loan-specific stress sensitivities")
    print(f"{'='*70}")

    pd_sensitivity, lgd_sensitivity, loan_sensitivity_summary = build_loan_specific_sensitivities(
        df=df,
        portfolio_upb=portfolio_upb,
        pd_baseline=pd_baseline,
        lgd_baseline=lgd_baseline,
        geography_primary="msa",
        geography_fallback="property_state",
    )
    for _, row in loan_sensitivity_summary.iterrows():
        print(
            f"  {row['metric']}: weighted_mean={row['weighted_mean']:.3f}, "
            f"p05={row['p05']:.3f}, p50={row['p50']:.3f}, p95={row['p95']:.3f}, "
            f"max={row['max']:.3f}"
        )
    print(
        "  Geography coverage: "
        f"MSA={loan_sensitivity_summary['msa_share'].iloc[0]*100:.1f}%, "
        f"State fallback={loan_sensitivity_summary['state_share'].iloc[0]*100:.1f}%, "
        f"Neutral={loan_sensitivity_summary['neutral_share'].iloc[0]*100:.1f}%"
    )

    print("  Monte Carlo scoring input quality:")
    log_series_quality("original_upb", portfolio_upb)
    log_series_quality("pd_baseline", pd_baseline)
    log_series_quality("lgd_baseline", lgd_baseline)
    log_series_quality("pd_sensitivity", pd_sensitivity)
    log_series_quality("lgd_sensitivity", lgd_sensitivity)

    valid_scoring_mask = build_valid_scoring_mask(
        portfolio_upb=portfolio_upb,
        pd_baseline=pd_baseline,
        lgd_baseline=lgd_baseline,
        pd_sensitivity=pd_sensitivity,
        lgd_sensitivity=lgd_sensitivity,
    )
    invalid_rows = int((~valid_scoring_mask).sum())
    if invalid_rows > 0:
        print(
            f"  Dropping {invalid_rows:,} row(s) with invalid Monte Carlo inputs "
            "before writing the scored portfolio."
        )
        df = df.loc[valid_scoring_mask].copy()
        portfolio_upb = portfolio_upb[valid_scoring_mask]
        pd_baseline = pd_baseline[valid_scoring_mask]
        lgd_baseline = lgd_baseline[valid_scoring_mask]
        pd_sensitivity = pd_sensitivity[valid_scoring_mask]
        lgd_sensitivity = lgd_sensitivity[valid_scoring_mask]
        total_balance = float(np.sum(portfolio_upb))
        baseline_el = float(np.sum(pd_baseline * lgd_baseline * portfolio_upb))
        print(
            f"  Clean scoring set: {len(portfolio_upb):,} loans, "
            f"${total_balance/1e9:.1f}B balance"
        )

    del df
    gc.collect()
    timings["loan_specific_sensitivity_seconds"] = time.time() - step_start

    # ------------------------------------------------------------------
    # Step 2b: Persist compact scored portfolio for streamed Monte Carlo
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print("Step 2b: Writing compact scored portfolio")
    print(f"{'='*70}")

    scored_portfolio_path = MODEL_DIR / f"{output_prefix}_scored_portfolio.parquet"
    write_scored_portfolio_parquet(
        output_path=scored_portfolio_path,
        portfolio_upb=portfolio_upb,
        pd_baseline=pd_baseline,
        lgd_baseline=lgd_baseline,
        pd_sensitivity=pd_sensitivity,
        lgd_sensitivity=lgd_sensitivity,
        row_group_size=args.loan_chunk_size,
    )
    print(f"  Scored portfolio parquet: {scored_portfolio_path}")

    del portfolio_upb, pd_baseline, lgd_baseline, pd_sensitivity, lgd_sensitivity
    gc.collect()
    timings["scored_portfolio_write_seconds"] = time.time() - step_start

    # ------------------------------------------------------------------
    # Step 3: Compute historical macro statistics
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print("Step 3: Historical macro statistics")
    print(f"{'='*70}")

    macro_path = resolve_macro_path()
    macro_stats = compute_historical_macro_stats(macro_path)
    timings["macro_stats_seconds"] = time.time() - step_start

    # ------------------------------------------------------------------
    # Step 4: Run Monte Carlo simulation
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print(f"Step 4: Running {args.n_simulations:,} Monte Carlo simulations")
    print(f"{'='*70}")

    scored_portfolio_summary = summarize_portfolio_path(scored_portfolio_path)
    print(f"  Scored portfolio path: {scored_portfolio_path}")
    print(
        f"  Scored portfolio summary: {scored_portfolio_summary['file_count']:,} file(s), "
        f"{scored_portfolio_summary['row_group_count']:,} row group(s), "
        f"{scored_portfolio_summary['row_count']:,} row(s)"
    )
    if args.backend == "cpu" and cpu_parallelism["cpu_workers"] > 1:
        work_item_count = len(build_scored_portfolio_work_items(scored_portfolio_path))
        print(
            f"  Parallel work plan: {work_item_count:,} work item(s) across "
            f"{cpu_parallelism['cpu_workers']} worker(s)"
        )

    _mc_kwargs = dict(
        portfolio_upb=None,
        pd_baseline=None,
        lgd_baseline=None,
        macro_stats=macro_stats,
        backend=args.backend,
        dtype=args.dtype,
        scenario_batch_size=args.scenario_batch_size,
        loan_chunk_size=args.loan_chunk_size,
        portfolio_path=scored_portfolio_path,
        cpu_workers=cpu_parallelism["cpu_workers"],
        torch_threads_per_worker=cpu_parallelism["torch_threads_per_worker"],
        pyarrow_threads=cpu_parallelism["pyarrow_threads"],
        antithetic=args.antithetic_variates,
    )

    if not args.adaptive_stopping:
        losses, scenarios = run_monte_carlo(
            **_mc_kwargs,
            n_simulations=args.n_simulations,
            random_seed=args.random_seed,
            portfolio_chunks=iter_scored_portfolio_chunks(
                scored_portfolio_path,
                batch_size=args.loan_chunk_size,
            ),
        )
        simulations_executed = args.n_simulations
        early_stopped = False
        batches_run = 1
    else:
        sim_batch = args.simulation_batch_size
        max_sims = args.max_simulations if args.max_simulations > 0 else args.n_simulations
        min_sims = args.min_simulations if args.min_simulations > 0 else sim_batch
        conv_metrics = args.convergence_metrics
        tol = args.convergence_tolerance

        all_losses_parts: list = []
        all_scenarios_parts: list = []
        prev_metrics_snap: dict | None = None
        simulations_executed = 0
        early_stopped = False
        batches_run = 0

        while simulations_executed < max_sims:
            remaining = max_sims - simulations_executed
            this_batch = min(sim_batch, remaining)
            batch_seed = args.random_seed + batches_run * 7919  # prime stride
            print(
                f"  [adaptive batch {batches_run + 1}] "
                f"sims={this_batch:,}, seed={batch_seed}, "
                f"total_so_far={simulations_executed:,}"
            )
            b_losses, b_scenarios = run_monte_carlo(
                **_mc_kwargs,
                n_simulations=this_batch,
                random_seed=batch_seed,
                portfolio_chunks=iter_scored_portfolio_chunks(
                    scored_portfolio_path,
                    batch_size=args.loan_chunk_size,
                ),
            )
            all_losses_parts.append(b_losses)
            all_scenarios_parts.append(b_scenarios)
            simulations_executed += this_batch
            batches_run += 1

            if simulations_executed >= min_sims:
                curr_snap = _quick_risk_metrics(np.concatenate(all_losses_parts))
                if prev_metrics_snap is not None and _is_converged(
                    prev_metrics_snap, curr_snap, conv_metrics, tol
                ):
                    print(
                        f"  [adaptive] converged after {simulations_executed:,} simulations "
                        f"({batches_run} batches). Metrics: "
                        + ", ".join(
                            f"{m}={curr_snap[m]/1e6:.2f}M" for m in conv_metrics
                        )
                    )
                    early_stopped = True
                    break
                prev_metrics_snap = curr_snap

        losses = np.concatenate(all_losses_parts)
        scenarios = pd.concat(all_scenarios_parts, ignore_index=True)
        scenarios["portfolio_loss"] = losses
        scenarios["loss_rate"] = losses / total_balance

    timings["monte_carlo_seconds"] = time.time() - step_start
    finite_losses = int(np.isfinite(losses).sum())
    print(
        f"  Step 4 finished with {finite_losses:,}/{len(losses):,} finite losses "
        f"in {timings['monte_carlo_seconds']:.1f}s"
        + (f" [early_stopped after {simulations_executed:,} sims]" if early_stopped else "")
    )

    # ------------------------------------------------------------------
    # Step 5: Compute risk metrics
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print("Step 5: Risk metrics from loss distribution")
    print(f"{'='*70}")

    metrics = compute_risk_metrics(
        losses=losses,
        total_balance=total_balance,
        backend=args.backend,
        dtype=args.dtype,
    )
    metrics["total_balance"] = total_balance
    metrics["backend"] = args.backend
    metrics["execution_mode"] = execution_mode
    metrics["n_simulations"] = args.n_simulations
    metrics["simulations_executed"] = simulations_executed
    metrics["random_seed"] = args.random_seed
    metrics["dtype"] = args.dtype
    metrics["scenario_batch_size"] = args.scenario_batch_size
    metrics["loan_chunk_size"] = args.loan_chunk_size
    metrics["cpu_workers"] = cpu_parallelism["cpu_workers"]
    metrics["torch_threads_per_worker"] = cpu_parallelism["torch_threads_per_worker"]
    metrics["pyarrow_threads"] = cpu_parallelism["pyarrow_threads"]
    metrics["antithetic_variates"] = args.antithetic_variates
    metrics["adaptive_stopping"] = args.adaptive_stopping
    metrics["early_stopped"] = early_stopped
    metrics["batches_run"] = batches_run
    metrics["simulation_batch_size"] = args.simulation_batch_size if args.adaptive_stopping else None
    metrics["convergence_tolerance"] = args.convergence_tolerance if args.adaptive_stopping else None
    metrics["convergence_metrics"] = (
        ",".join(args.convergence_metrics) if args.adaptive_stopping else None
    )
    timings["risk_metrics_seconds"] = time.time() - step_start

    # ------------------------------------------------------------------
    # Step 6: Sensitivity analysis
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print("Step 6: Sensitivity analysis (tornado chart data)")
    print(f"{'='*70}")

    sensitivity_df = sensitivity_analysis_custom(
        scored_portfolio_path=scored_portfolio_path,
        total_balance=total_balance,
        baseline_loss=baseline_el,
        macro_stats=macro_stats,
        backend=args.backend,
        dtype=args.dtype,
        scenario_batch_size=args.scenario_batch_size,
        loan_chunk_size=args.loan_chunk_size,
        cpu_workers=cpu_parallelism["cpu_workers"],
        torch_threads_per_worker=cpu_parallelism["torch_threads_per_worker"],
        pyarrow_threads=cpu_parallelism["pyarrow_threads"],
    )
    timings["sensitivity_seconds"] = time.time() - step_start

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print("Step 7: Saving results")
    print(f"{'='*70}")

    saved_paths = save_outputs(
        output_prefix=output_prefix,
        losses=losses,
        scenarios=scenarios,
        metrics=metrics,
        sensitivity_df=sensitivity_df,
        loan_sensitivity_summary=loan_sensitivity_summary,
    )

    if args.write_standard_files:
        write_standard_files(
            losses=losses,
            scenarios=scenarios,
            metrics=metrics,
            sensitivity_df=sensitivity_df,
        )
        print("  Also wrote standard dashboard Monte Carlo files.")

    timings["save_seconds"] = time.time() - step_start

    runtime_df = pd.DataFrame([{
        "backend": args.backend,
        "n_simulations": args.n_simulations,
        "simulations_executed": simulations_executed,
        "random_seed": args.random_seed,
        "dtype": args.dtype,
        "execution_mode": execution_mode,
        "cpu_workers": cpu_parallelism["cpu_workers"],
        "torch_threads_per_worker": cpu_parallelism["torch_threads_per_worker"],
        "pyarrow_threads": cpu_parallelism["pyarrow_threads"],
        "antithetic_variates": args.antithetic_variates,
        "adaptive_stopping": args.adaptive_stopping,
        "early_stopped": early_stopped,
        "batches_run": batches_run,
        **timings,
        "total_runtime_seconds": time.time() - t_total_start,
    }])
    runtime_df.to_csv(
        MODEL_DIR / f"{output_prefix}_runtime_summary.csv",
        index=False,
    )
    saved_paths["runtime_summary"] = MODEL_DIR / f"{output_prefix}_runtime_summary.csv"

    print(f"  Saved backend-specific outputs:")
    for label, path in saved_paths.items():
        print(f"    {label}: {path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_total_start

    print(f"\n{'='*70}")
    print("MONTE CARLO CUSTOM BACKEND RESULTS")
    print(f"{'='*70}")
    print(f"  Backend:               {args.backend}")
    print(f"  Execution mode:        {execution_mode}")
    print(f"  Portfolio:             {n_loans:,} loans, ${total_balance/1e9:.1f}B")
    print(f"  Simulations:           {args.n_simulations:,}")
    print(f"  Expected Loss (mean):  ${metrics['expected_loss']/1e6:>10,.0f}M "
          f"({metrics['expected_loss']/total_balance*100:.2f}%)")
    print(f"  VaR 99%:               ${metrics['var_99']/1e6:>10,.0f}M "
          f"({metrics['var_99']/total_balance*100:.2f}%)")
    print(f"  VaR 99.9%:             ${metrics['var_999']/1e6:>10,.0f}M "
          f"({metrics['var_999']/total_balance*100:.2f}%)")
    print(f"  Expected Shortfall:    ${metrics['es_99']/1e6:>10,.0f}M "
          f"({metrics['es_99']/total_balance*100:.2f}%)")
    print(f"\n  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
