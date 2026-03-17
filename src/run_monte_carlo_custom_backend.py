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
    compute_historical_macro_stats,
    compute_risk_metrics,
    compute_scenario_losses,
    run_monte_carlo,
)


COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
MACRO_PATH = project_root / "data" / "processed" / "macro" / "fred_macro_monthly.csv"
MODEL_DIR = project_root / "models"


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


def sensitivity_analysis_custom(
    portfolio_upb,
    pd_baseline,
    lgd_baseline,
    macro_stats,
    pd_sensitivity,
    lgd_sensitivity,
    backend,
    dtype,
    scenario_batch_size,
    loan_chunk_size,
):
    """One-at-a-time sensitivity analysis using loan-specific multipliers."""
    total_balance = portfolio_upb.sum()
    baseline_loss = float((portfolio_upb * pd_baseline * lgd_baseline).sum())

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
            portfolio_upb=portfolio_upb,
            pd_baseline=pd_baseline,
            lgd_baseline=lgd_baseline,
            scenarios=scenarios_df,
            pd_sensitivity=pd_sensitivity,
            lgd_sensitivity=lgd_sensitivity,
            backend=backend,
            dtype=dtype,
            scenario_batch_size=scenario_batch_size,
            loan_chunk_size=loan_chunk_size,
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


def main():
    args = parse_args()
    output_prefix = args.output_prefix or f"mc_{args.backend}"

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
    print(f"Output prefix: {output_prefix}")

    # ------------------------------------------------------------------
    # Step 1: Load portfolio and score baseline PD/LGD
    # ------------------------------------------------------------------
    step_start = time.time()
    print("\nStep 1: Loading portfolio and scoring baseline...")

    df = pd.read_parquet(COMBINED_PATH)
    df.loc[df["data_split"] == "unknown", "data_split"] = "train"
    total_balance = df["original_upb"].sum()
    portfolio_upb = df["original_upb"].values.astype(float)
    print(f"  Portfolio: {len(df):,} loans, ${total_balance/1e9:.1f}B")

    pd_model = joblib.load(MODEL_DIR / "pd_logistic_regression.pkl")
    woe_results = joblib.load(MODEL_DIR / "woe_results.pkl")
    with open(MODEL_DIR / "selected_features.txt") as f:
        pd_features = [line.strip() for line in f if line.strip()]

    X_woe = apply_woe_transformation(df, woe_results, pd_features)
    pd_baseline = pd_model.predict_proba(X_woe)[:, 1]
    del X_woe
    gc.collect()

    lgd_model = joblib.load(MODEL_DIR / "lgd_ols.pkl")
    with open(MODEL_DIR / "lgd_features.txt") as f:
        lgd_features = [line.strip() for line in f if line.strip()]

    X_lgd = df[lgd_features].copy()
    lgd_fill = {"loan_age_at_default": 48.0, "was_modified": 0.0}
    for col in lgd_features:
        if col in lgd_fill:
            X_lgd[col] = X_lgd[col].fillna(lgd_fill[col])
        else:
            X_lgd[col] = X_lgd[col].fillna(X_lgd[col].median())
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
    del df
    gc.collect()
    timings["loan_specific_sensitivity_seconds"] = time.time() - step_start

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

    losses, scenarios = run_monte_carlo(
        portfolio_upb=portfolio_upb,
        pd_baseline=pd_baseline,
        lgd_baseline=lgd_baseline,
        macro_stats=macro_stats,
        pd_sensitivity=pd_sensitivity,
        lgd_sensitivity=lgd_sensitivity,
        n_simulations=args.n_simulations,
        random_seed=args.random_seed,
        backend=args.backend,
        dtype=args.dtype,
        scenario_batch_size=args.scenario_batch_size,
        loan_chunk_size=args.loan_chunk_size,
    )
    timings["monte_carlo_seconds"] = time.time() - step_start

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
    metrics["n_simulations"] = args.n_simulations
    metrics["random_seed"] = args.random_seed
    metrics["dtype"] = args.dtype
    metrics["scenario_batch_size"] = args.scenario_batch_size
    metrics["loan_chunk_size"] = args.loan_chunk_size
    timings["risk_metrics_seconds"] = time.time() - step_start

    # ------------------------------------------------------------------
    # Step 6: Sensitivity analysis
    # ------------------------------------------------------------------
    step_start = time.time()
    print(f"\n{'='*70}")
    print("Step 6: Sensitivity analysis (tornado chart data)")
    print(f"{'='*70}")

    sensitivity_df = sensitivity_analysis_custom(
        portfolio_upb=portfolio_upb,
        pd_baseline=pd_baseline,
        lgd_baseline=lgd_baseline,
        macro_stats=macro_stats,
        pd_sensitivity=pd_sensitivity,
        lgd_sensitivity=lgd_sensitivity,
        backend=args.backend,
        dtype=args.dtype,
        scenario_batch_size=args.scenario_batch_size,
        loan_chunk_size=args.loan_chunk_size,
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
        "random_seed": args.random_seed,
        "dtype": args.dtype,
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
    print(f"  Portfolio:             {len(portfolio_upb):,} loans, ${total_balance/1e9:.1f}B")
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
