"""
Torch-backed Monte Carlo module with selectable CPU, MPS, or CUDA execution.

This module preserves the same high-level Monte Carlo workflow as
`src/monte_carlo.py`, but moves the dense numeric kernels onto PyTorch so they
can run on:

- `cpu`
- `mps` (Apple Silicon)
- `cuda` (NVIDIA GPU)

The public functions return NumPy arrays / pandas DataFrames by default so the
rest of the project can integrate with this module without rewriting its I/O
path. Internally, the heavy scenario generation, multiplier calculation, loss
vectorization, and risk metric calculations are executed with torch tensors.

Notes:
- PyTorch is an optional dependency for this module and must be installed
  separately.
- The default dtype is float32 because MPS does not reliably support float64
  across all operations.
- Exact bitwise reproducibility across CPU, MPS, and CUDA should not be
  assumed because backend kernels can differ slightly.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _import_torch():
    """Import torch lazily so the rest of the repo remains importable."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "monte_carlo_custom_backend.py requires PyTorch. "
            "Install a torch build that matches your target backend."
        ) from exc
    return torch


def resolve_backend_device(backend: str = "cpu"):
    """
    Resolve a backend name into a torch device after validating availability.

    Parameters
    ----------
    backend : str
        One of: 'cpu', 'mps', 'cuda'.
    """
    torch = _import_torch()

    backend = backend.lower().strip()
    if backend == "cpu":
        return torch.device("cpu")

    if backend == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError(
                "Backend 'mps' requested but PyTorch MPS is not available."
            )
        return torch.device("mps")

    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Backend 'cuda' requested but CUDA is not available."
            )
        return torch.device("cuda")

    raise ValueError("backend must be one of: 'cpu', 'mps', 'cuda'")


def _resolve_torch_dtype(dtype, device):
    """Map a dtype hint to a torch dtype, defaulting to float32."""
    torch = _import_torch()

    if dtype is None:
        resolved = torch.float32
    elif isinstance(dtype, str):
        lookup = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if dtype not in lookup:
            raise ValueError("dtype must be one of: float16, float32, float64")
        resolved = lookup[dtype]
    else:
        resolved = dtype

    # MPS support for float64 is incomplete; keep the default safe.
    if device.type == "mps" and resolved == torch.float64:
        warnings.warn(
            "float64 requested on MPS; falling back to float32 for compatibility."
        )
        resolved = torch.float32

    return resolved


def _tensor_to_numpy(tensor):
    """Detach a tensor and move it to CPU NumPy."""
    return tensor.detach().to("cpu").numpy()


def _scenario_tensor_to_dataframe(scenarios_tensor, variable_names: Sequence[str]):
    """Convert a 2D scenario tensor into a pandas DataFrame."""
    return pd.DataFrame(_tensor_to_numpy(scenarios_tensor), columns=list(variable_names))


def _prepare_portfolio_tensors(
    portfolio_upb,
    pd_baseline,
    lgd_baseline,
    pd_sensitivity,
    lgd_sensitivity,
    backend,
    dtype,
):
    """Convert portfolio arrays into tensors on the requested backend."""
    torch = _import_torch()
    device = resolve_backend_device(backend)
    torch_dtype = _resolve_torch_dtype(dtype, device)

    portfolio_upb_t = torch.as_tensor(portfolio_upb, device=device, dtype=torch_dtype)
    pd_baseline_t = torch.as_tensor(pd_baseline, device=device, dtype=torch_dtype)
    lgd_baseline_t = torch.as_tensor(lgd_baseline, device=device, dtype=torch_dtype)

    if pd_sensitivity is None:
        pd_sensitivity_t = torch.ones_like(pd_baseline_t)
    else:
        pd_sensitivity_t = torch.as_tensor(pd_sensitivity, device=device, dtype=torch_dtype)

    if lgd_sensitivity is None:
        lgd_sensitivity_t = torch.ones_like(lgd_baseline_t)
    else:
        lgd_sensitivity_t = torch.as_tensor(lgd_sensitivity, device=device, dtype=torch_dtype)

    return (
        portfolio_upb_t,
        pd_baseline_t,
        lgd_baseline_t,
        pd_sensitivity_t,
        lgd_sensitivity_t,
        device,
        torch_dtype,
    )


def _coerce_scenarios_to_tensor(
    scenarios,
    variable_names: Sequence[str] | None,
    backend: str,
    dtype,
):
    """
    Accept scenario input as either:
    - pandas DataFrame
    - NumPy array plus variable_names
    - torch tensor plus variable_names
    """
    torch = _import_torch()
    device = resolve_backend_device(backend)
    torch_dtype = _resolve_torch_dtype(dtype, device)

    if isinstance(scenarios, pd.DataFrame):
        names = list(scenarios.columns)
        tensor = torch.as_tensor(
            scenarios.to_numpy(copy=True),
            device=device,
            dtype=torch_dtype,
        )
        return tensor, names, device, torch_dtype

    if isinstance(scenarios, np.ndarray):
        if variable_names is None:
            raise ValueError("variable_names is required when scenarios is a NumPy array")
        tensor = torch.as_tensor(scenarios, device=device, dtype=torch_dtype)
        return tensor, list(variable_names), device, torch_dtype

    if "torch" in str(type(scenarios)).lower():
        if variable_names is None:
            raise ValueError("variable_names is required when scenarios is a torch tensor")
        tensor = scenarios.to(device=device, dtype=torch_dtype)
        return tensor, list(variable_names), device, torch_dtype

    raise TypeError("scenarios must be a pandas DataFrame, NumPy array, or torch tensor")


def compute_historical_macro_stats(macro_csv_path):
    """
    Compute historical macro means, standard deviations, and correlations.

    This function remains pandas-based because it is small and I/O-driven; the
    GPU/MPS acceleration begins after the historical statistics are loaded.
    """
    macro_df = pd.read_csv(macro_csv_path, index_col=0, parse_dates=True)

    if "hpi_national" in macro_df.columns:
        macro_df["hpi_change_annual"] = macro_df["hpi_national"].pct_change(12) * 100
    if "gdp" in macro_df.columns:
        macro_df["gdp_growth_annual"] = macro_df["gdp"].pct_change(4) * 100

    analysis_cols = []
    col_names = []

    if "unemployment_rate" in macro_df.columns:
        analysis_cols.append("unemployment_rate")
        col_names.append("unemployment_rate")
    if "hpi_change_annual" in macro_df.columns:
        analysis_cols.append("hpi_change_annual")
        col_names.append("hpi_change_annual")
    if "gdp_growth_annual" in macro_df.columns:
        analysis_cols.append("gdp_growth_annual")
        col_names.append("gdp_growth_annual")

    subset = macro_df[analysis_cols].dropna()

    means = {col: subset[col].mean() for col in analysis_cols}
    stds = {col: subset[col].std() for col in analysis_cols}
    corr_matrix = subset.corr().values

    print(f"  Historical macro statistics (from {len(subset)} monthly observations):")
    for col in analysis_cols:
        print(f"    {col}: mean={means[col]:.2f}, std={stds[col]:.2f}")

    print("\n  Correlation matrix:")
    print(f"  {'':>25s}", end="")
    for c in col_names:
        print(f" {c[:12]:>12s}", end="")
    print()
    for i, c1 in enumerate(col_names):
        print(f"  {c1:>25s}", end="")
        for j in range(len(col_names)):
            print(f" {corr_matrix[i, j]:>12.3f}", end="")
        print()

    return {
        "means": means,
        "stds": stds,
        "correlation_matrix": corr_matrix,
        "variable_names": col_names,
    }


def _generate_correlated_scenarios_tensor(
    n_simulations: int,
    macro_stats: Mapping[str, object],
    random_seed: int = 42,
    backend: str = "cpu",
    dtype: str | None = "float32",
):
    """Internal torch implementation of correlated macro scenario generation."""
    torch = _import_torch()
    device = resolve_backend_device(backend)
    torch_dtype = _resolve_torch_dtype(dtype, device)
    variable_names = list(macro_stats["variable_names"])
    n_vars = len(variable_names)

    # Set the torch RNG seed so the same backend/run configuration is repeatable.
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    corr = torch.as_tensor(
        np.array(macro_stats["correlation_matrix"], dtype=np.float64),
        device=device,
        dtype=torch_dtype,
    ).clone()

    # Ensure positive definiteness before Cholesky.
    try:
        eigvals = torch.linalg.eigvalsh(corr)
    except RuntimeError:
        eigvals = torch.linalg.eigvalsh(corr.to("cpu")).to(device)

    min_eig = torch.min(eigvals)
    if float(min_eig.detach().to("cpu")) < 0:
        corr = corr + torch.eye(n_vars, device=device, dtype=torch_dtype) * (
            torch.abs(min_eig) + 0.01
        )

    try:
        L = torch.linalg.cholesky(corr)
    except RuntimeError:
        L = torch.linalg.cholesky(corr.to("cpu")).to(device)

    Z = torch.randn((n_simulations, n_vars), device=device, dtype=torch_dtype)
    correlated_Z = Z @ L.T

    means = torch.tensor(
        [macro_stats["means"][col] for col in variable_names],
        device=device,
        dtype=torch_dtype,
    )
    stds = torch.tensor(
        [macro_stats["stds"][col] for col in variable_names],
        device=device,
        dtype=torch_dtype,
    )

    scenarios = means + correlated_Z * stds

    for idx, name in enumerate(variable_names):
        if name == "unemployment_rate":
            scenarios[:, idx] = torch.clamp(scenarios[:, idx], 2.0, 15.0)
        elif name == "hpi_change_annual":
            scenarios[:, idx] = torch.clamp(scenarios[:, idx], -40.0, 30.0)
        elif name == "gdp_growth_annual":
            scenarios[:, idx] = torch.clamp(scenarios[:, idx], -15.0, 15.0)

    return scenarios, variable_names, device, torch_dtype


def generate_correlated_scenarios(
    n_simulations,
    macro_stats,
    random_seed: int = 42,
    backend: str = "cpu",
    dtype: str | None = "float32",
    return_tensor: bool = False,
):
    """
    Generate correlated random macro scenarios on the selected backend.

    Parameters
    ----------
    n_simulations : int
        Number of scenarios to generate.
    macro_stats : dict
        Output from compute_historical_macro_stats.
    random_seed : int
        Random seed for reproducibility.
    backend : str
        One of: 'cpu', 'mps', 'cuda'.
    dtype : str or torch.dtype
        Torch dtype hint. Defaults to float32.
    return_tensor : bool
        If True, return `(tensor, variable_names)` instead of a DataFrame.
    """
    scenarios, variable_names, _, _ = _generate_correlated_scenarios_tensor(
        n_simulations=n_simulations,
        macro_stats=macro_stats,
        random_seed=random_seed,
        backend=backend,
        dtype=dtype,
    )

    if return_tensor:
        return scenarios, variable_names

    return _scenario_tensor_to_dataframe(scenarios, variable_names)


def _compute_scenario_multipliers_tensor(
    scenarios_tensor,
    variable_names: Sequence[str],
    baseline_ur: float = 4.3,
    baseline_hpi_change: float = 0.0,
):
    """Internal torch implementation of multiplier construction."""
    torch = _import_torch()

    idx = {name: i for i, name in enumerate(variable_names)}
    n_scenarios = scenarios_tensor.shape[0]
    device = scenarios_tensor.device
    dtype = scenarios_tensor.dtype

    pd_multiplier = torch.ones(n_scenarios, device=device, dtype=dtype)
    lgd_multiplier = torch.ones(n_scenarios, device=device, dtype=dtype)

    if "unemployment_rate" in idx:
        ur_delta = torch.clamp(scenarios_tensor[:, idx["unemployment_rate"]] - baseline_ur, min=0.0)
        pd_multiplier = pd_multiplier + 0.25 * ur_delta

    if "gdp_growth_annual" in idx:
        gdp_neg = torch.clamp(-scenarios_tensor[:, idx["gdp_growth_annual"]], min=0.0)
        pd_multiplier = pd_multiplier + 0.05 * gdp_neg

    if "hpi_change_annual" in idx:
        hpi_decline = torch.clamp(
            -scenarios_tensor[:, idx["hpi_change_annual"]] + baseline_hpi_change,
            min=0.0,
        )
        lgd_multiplier = lgd_multiplier + 0.015 * hpi_decline

    pd_multiplier = torch.clamp(pd_multiplier, 0.5, 5.0)
    lgd_multiplier = torch.clamp(lgd_multiplier, 0.5, 3.0)

    multipliers = torch.stack([pd_multiplier, lgd_multiplier], dim=1)
    multiplier_names = ["pd_multiplier", "lgd_multiplier"]
    return multipliers, multiplier_names


def _aggregate_portfolio_losses_tensor(
    portfolio_upb_t,
    pd_baseline_t,
    lgd_baseline_t,
    pd_sensitivity_t,
    lgd_sensitivity_t,
    multipliers_t,
    scenario_batch_size: int,
    loan_chunk_size: int,
):
    """Aggregate stressed portfolio losses given per-scenario multipliers."""
    torch = _import_torch()

    baseline_portfolio_el = torch.sum(portfolio_upb_t * pd_baseline_t * lgd_baseline_t)
    total_balance_t = torch.sum(portfolio_upb_t)

    uses_uniform_sensitivity = bool(
        torch.all(pd_sensitivity_t == 1.0) and torch.all(lgd_sensitivity_t == 1.0)
    )
    if uses_uniform_sensitivity:
        losses_t = multipliers_t[:, 0] * multipliers_t[:, 1] * baseline_portfolio_el
        return losses_t, total_balance_t

    n_loans = portfolio_upb_t.shape[0]
    n_scenarios = multipliers_t.shape[0]
    losses_t = torch.zeros(n_scenarios, device=portfolio_upb_t.device, dtype=portfolio_upb_t.dtype)

    for start_scenario in range(0, n_scenarios, scenario_batch_size):
        end_scenario = min(start_scenario + scenario_batch_size, n_scenarios)
        batch_multipliers = multipliers_t[start_scenario:end_scenario]
        pd_delta = batch_multipliers[:, 0] - 1.0
        lgd_delta = batch_multipliers[:, 1] - 1.0
        batch_losses = torch.zeros(
            end_scenario - start_scenario,
            device=portfolio_upb_t.device,
            dtype=portfolio_upb_t.dtype,
        )

        for start_loan in range(0, n_loans, loan_chunk_size):
            end_loan = min(start_loan + loan_chunk_size, n_loans)

            upb_chunk = portfolio_upb_t[start_loan:end_loan]
            pd_chunk = pd_baseline_t[start_loan:end_loan]
            lgd_chunk = lgd_baseline_t[start_loan:end_loan]
            pd_sens_chunk = pd_sensitivity_t[start_loan:end_loan]
            lgd_sens_chunk = lgd_sensitivity_t[start_loan:end_loan]

            stressed_pd_multiplier = 1.0 + pd_delta[:, None] * pd_sens_chunk[None, :]
            stressed_lgd_multiplier = 1.0 + lgd_delta[:, None] * lgd_sens_chunk[None, :]

            stressed_pd = torch.clamp(pd_chunk[None, :] * stressed_pd_multiplier, 0.0, 1.0)
            stressed_lgd = torch.clamp(lgd_chunk[None, :] * stressed_lgd_multiplier, 0.0, 1.5)

            chunk_losses = torch.sum(
                upb_chunk[None, :] * stressed_pd * stressed_lgd,
                dim=1,
            )
            batch_losses = batch_losses + chunk_losses

        losses_t[start_scenario:end_scenario] = batch_losses

    return losses_t, total_balance_t


def compute_scenario_multipliers(
    scenarios,
    variable_names: Sequence[str] | None = None,
    baseline_ur: float = 4.3,
    baseline_hpi_change: float = 0.0,
    backend: str = "cpu",
    dtype: str | None = "float32",
    return_tensor: bool = False,
):
    """
    Convert macro scenarios into PD and LGD multipliers on the selected backend.

    Parameters
    ----------
    scenarios : pd.DataFrame, np.ndarray, or torch.Tensor
        Scenario values by row.
    variable_names : sequence of str, optional
        Required when `scenarios` is not a DataFrame.
    baseline_ur : float
        Baseline unemployment rate.
    baseline_hpi_change : float
        Baseline annual HPI change.
    backend : str
        One of: 'cpu', 'mps', 'cuda'.
    dtype : str or torch.dtype
        Torch dtype hint. Defaults to float32.
    return_tensor : bool
        If True, return `(tensor, multiplier_names)` instead of a DataFrame.
    """
    scenarios_tensor, scenario_names, _, _ = _coerce_scenarios_to_tensor(
        scenarios=scenarios,
        variable_names=variable_names,
        backend=backend,
        dtype=dtype,
    )

    multipliers, multiplier_names = _compute_scenario_multipliers_tensor(
        scenarios_tensor=scenarios_tensor,
        variable_names=scenario_names,
        baseline_ur=baseline_ur,
        baseline_hpi_change=baseline_hpi_change,
    )

    if return_tensor:
        return multipliers, multiplier_names

    out_df = _scenario_tensor_to_dataframe(scenarios_tensor, scenario_names)
    mult_df = _scenario_tensor_to_dataframe(multipliers, multiplier_names)
    return pd.concat([out_df, mult_df], axis=1)


def compute_scenario_losses(
    portfolio_upb,
    pd_baseline,
    lgd_baseline,
    scenarios,
    variable_names: Sequence[str] | None = None,
    pd_sensitivity=None,
    lgd_sensitivity=None,
    baseline_ur: float = 4.3,
    baseline_hpi_change: float = 0.0,
    backend: str = "cpu",
    dtype: str | None = "float32",
    scenario_batch_size: int = 128,
    loan_chunk_size: int = 100_000,
    return_tensors: bool = False,
):
    """Score an explicit set of macro scenarios against the portfolio."""
    scenarios_tensor, scenario_names, _, _ = _coerce_scenarios_to_tensor(
        scenarios=scenarios,
        variable_names=variable_names,
        backend=backend,
        dtype=dtype,
    )
    (
        portfolio_upb_t,
        pd_baseline_t,
        lgd_baseline_t,
        pd_sensitivity_t,
        lgd_sensitivity_t,
        _,
        _,
    ) = _prepare_portfolio_tensors(
        portfolio_upb=portfolio_upb,
        pd_baseline=pd_baseline,
        lgd_baseline=lgd_baseline,
        pd_sensitivity=pd_sensitivity,
        lgd_sensitivity=lgd_sensitivity,
        backend=backend,
        dtype=dtype,
    )

    multipliers_t, multiplier_names = _compute_scenario_multipliers_tensor(
        scenarios_tensor=scenarios_tensor,
        variable_names=scenario_names,
        baseline_ur=baseline_ur,
        baseline_hpi_change=baseline_hpi_change,
    )
    losses_t, total_balance_t = _aggregate_portfolio_losses_tensor(
        portfolio_upb_t=portfolio_upb_t,
        pd_baseline_t=pd_baseline_t,
        lgd_baseline_t=lgd_baseline_t,
        pd_sensitivity_t=pd_sensitivity_t,
        lgd_sensitivity_t=lgd_sensitivity_t,
        multipliers_t=multipliers_t,
        scenario_batch_size=scenario_batch_size,
        loan_chunk_size=loan_chunk_size,
    )
    loss_rate_t = losses_t / total_balance_t

    if return_tensors:
        return losses_t, scenarios_tensor, multipliers_t

    scenarios_df = _scenario_tensor_to_dataframe(scenarios_tensor, scenario_names)
    mult_df = _scenario_tensor_to_dataframe(multipliers_t, multiplier_names)
    scenarios_df = pd.concat([scenarios_df, mult_df], axis=1)
    scenarios_df["portfolio_loss"] = _tensor_to_numpy(losses_t)
    scenarios_df["loss_rate"] = _tensor_to_numpy(loss_rate_t)
    return _tensor_to_numpy(losses_t), scenarios_df


def _torch_quantile(values, q):
    """
    Quantile helper with a CPU fallback for backends that do not support the
    requested op or dtype cleanly.
    """
    torch = _import_torch()
    try:
        return torch.quantile(values, q)
    except RuntimeError:
        return torch.quantile(values.to("cpu"), q).to(values.device)


def compute_risk_metrics(
    losses,
    total_balance,
    backend: str = "cpu",
    dtype: str | None = "float32",
):
    """
    Compute risk metrics using torch on the requested backend.

    Parameters
    ----------
    losses : np.ndarray or torch.Tensor
        Portfolio loss values.
    total_balance : float
        Total portfolio balance.
    backend : str
        One of: 'cpu', 'mps', 'cuda'.
    dtype : str or torch.dtype
        Torch dtype hint. Defaults to float32.
    """
    torch = _import_torch()
    device = resolve_backend_device(backend)
    torch_dtype = _resolve_torch_dtype(dtype, device)

    if isinstance(losses, np.ndarray):
        losses_t = torch.as_tensor(losses, device=device, dtype=torch_dtype)
    else:
        losses_t = losses.to(device=device, dtype=torch_dtype)

    sorted_losses, _ = torch.sort(losses_t)

    expected_loss = torch.mean(losses_t)
    std_loss = torch.std(losses_t, unbiased=False)
    var_99 = _torch_quantile(losses_t, 0.99)
    var_999 = _torch_quantile(losses_t, 0.999)
    es_99 = torch.mean(losses_t[losses_t >= var_99])

    p5 = _torch_quantile(losses_t, 0.05)
    p25 = _torch_quantile(losses_t, 0.25)
    p50 = _torch_quantile(losses_t, 0.50)
    p75 = _torch_quantile(losses_t, 0.75)
    p95 = _torch_quantile(losses_t, 0.95)

    metrics = {
        "expected_loss": float(expected_loss.detach().to("cpu")),
        "std_loss": float(std_loss.detach().to("cpu")),
        "var_99": float(var_99.detach().to("cpu")),
        "var_999": float(var_999.detach().to("cpu")),
        "es_99": float(es_99.detach().to("cpu")),
        "p5": float(p5.detach().to("cpu")),
        "p25": float(p25.detach().to("cpu")),
        "p50": float(p50.detach().to("cpu")),
        "p75": float(p75.detach().to("cpu")),
        "p95": float(p95.detach().to("cpu")),
        "min_loss": float(sorted_losses[0].detach().to("cpu")),
        "max_loss": float(sorted_losses[-1].detach().to("cpu")),
    }

    print("\n  Risk Metrics:")
    print(f"    Expected Loss (mean):   ${metrics['expected_loss']/1e6:>10,.0f}M "
          f"({metrics['expected_loss']/total_balance*100:.2f}%)")
    print(f"    Std Deviation:          ${metrics['std_loss']/1e6:>10,.0f}M")
    print(f"    VaR 99%:                ${metrics['var_99']/1e6:>10,.0f}M "
          f"({metrics['var_99']/total_balance*100:.2f}%)")
    print(f"    VaR 99.9%:              ${metrics['var_999']/1e6:>10,.0f}M "
          f"({metrics['var_999']/total_balance*100:.2f}%)")
    print(f"    Expected Shortfall 99%: ${metrics['es_99']/1e6:>10,.0f}M "
          f"({metrics['es_99']/total_balance*100:.2f}%)")
    print("\n  Loss Distribution:")
    print(f"    5th percentile:  ${metrics['p5']/1e6:>10,.0f}M ({metrics['p5']/total_balance*100:.2f}%)")
    print(f"    25th percentile: ${metrics['p25']/1e6:>10,.0f}M ({metrics['p25']/total_balance*100:.2f}%)")
    print(f"    Median:          ${metrics['p50']/1e6:>10,.0f}M ({metrics['p50']/total_balance*100:.2f}%)")
    print(f"    75th percentile: ${metrics['p75']/1e6:>10,.0f}M ({metrics['p75']/total_balance*100:.2f}%)")
    print(f"    95th percentile: ${metrics['p95']/1e6:>10,.0f}M ({metrics['p95']/total_balance*100:.2f}%)")
    print(f"    Maximum:         ${metrics['max_loss']/1e6:>10,.0f}M ({metrics['max_loss']/total_balance*100:.2f}%)")

    return metrics


def run_monte_carlo(
    portfolio_upb,
    pd_baseline,
    lgd_baseline,
    macro_stats,
    pd_sensitivity=None,
    lgd_sensitivity=None,
    n_simulations: int = 10_000,
    random_seed: int = 42,
    backend: str = "cpu",
    dtype: str | None = "float32",
    scenario_batch_size: int = 128,
    loan_chunk_size: int = 100_000,
    return_tensors: bool = False,
):
    """
    Run Monte Carlo simulation on the requested backend.

    Parameters
    ----------
    portfolio_upb : np.ndarray
        Original UPB for each loan.
    pd_baseline : np.ndarray
        Baseline PD for each loan.
    lgd_baseline : np.ndarray
        Baseline LGD for each loan.
    macro_stats : dict
        Historical macro statistics.
    pd_sensitivity : np.ndarray, optional
        Per-loan PD stress sensitivity. If omitted, all loans use 1.0.
    lgd_sensitivity : np.ndarray, optional
        Per-loan LGD stress sensitivity. If omitted, all loans use 1.0.
    n_simulations : int
        Number of scenarios.
    random_seed : int
        Random seed.
    backend : str
        One of: 'cpu', 'mps', 'cuda'.
    dtype : str or torch.dtype
        Torch dtype hint. Defaults to float32.
    scenario_batch_size : int
        Number of scenarios per batch in the per-loan aggregation loop.
    loan_chunk_size : int
        Number of loans per chunk in the per-loan aggregation loop.
    return_tensors : bool
        If True, return torch tensors instead of NumPy/DataFrame objects.
    """
    (
        portfolio_upb_t,
        pd_baseline_t,
        lgd_baseline_t,
        pd_sensitivity_t,
        lgd_sensitivity_t,
        device,
        torch_dtype,
    ) = _prepare_portfolio_tensors(
        portfolio_upb=portfolio_upb,
        pd_baseline=pd_baseline,
        lgd_baseline=lgd_baseline,
        pd_sensitivity=pd_sensitivity,
        lgd_sensitivity=lgd_sensitivity,
        backend=backend,
        dtype=dtype,
    )
    print(f"\n  Running {n_simulations:,} Monte Carlo simulations on {device.type}...")
    t0 = time.time()

    scenarios_t, variable_names, _, _ = _generate_correlated_scenarios_tensor(
        n_simulations=n_simulations,
        macro_stats=macro_stats,
        random_seed=random_seed,
        backend=backend,
        dtype=dtype,
    )

    multipliers_t, multiplier_names = _compute_scenario_multipliers_tensor(
        scenarios_tensor=scenarios_t,
        variable_names=variable_names,
    )

    losses_t, total_balance_t = _aggregate_portfolio_losses_tensor(
        portfolio_upb_t=portfolio_upb_t,
        pd_baseline_t=pd_baseline_t,
        lgd_baseline_t=lgd_baseline_t,
        pd_sensitivity_t=pd_sensitivity_t,
        lgd_sensitivity_t=lgd_sensitivity_t,
        multipliers_t=multipliers_t,
        scenario_batch_size=scenario_batch_size,
        loan_chunk_size=loan_chunk_size,
    )
    loss_rate_t = losses_t / total_balance_t

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f} seconds")

    if return_tensors:
        return losses_t, scenarios_t, multipliers_t

    scenarios_df = _scenario_tensor_to_dataframe(scenarios_t, variable_names)
    mult_df = _scenario_tensor_to_dataframe(multipliers_t, multiplier_names)
    scenarios_df = pd.concat([scenarios_df, mult_df], axis=1)
    scenarios_df["portfolio_loss"] = _tensor_to_numpy(losses_t)
    scenarios_df["loss_rate"] = _tensor_to_numpy(loss_rate_t)

    return _tensor_to_numpy(losses_t), scenarios_df


__all__ = [
    "compute_historical_macro_stats",
    "generate_correlated_scenarios",
    "compute_scenario_multipliers",
    "compute_scenario_losses",
    "compute_risk_metrics",
    "run_monte_carlo",
    "resolve_backend_device",
]
