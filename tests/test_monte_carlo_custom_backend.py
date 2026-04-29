"""
Tests for antithetic variates and adaptive stopping in the Monte Carlo engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import monte_carlo_custom_backend as mc
from monte_carlo_custom_backend import _generate_correlated_scenarios_tensor


# ---------------------------------------------------------------------------
# Minimal macro_stats fixture (3 macro variables, identity correlation)
# ---------------------------------------------------------------------------

@pytest.fixture()
def macro_stats():
    corr = np.eye(3).tolist()
    return {
        "variable_names": ["unemployment_rate", "hpi_change_annual", "gdp_growth_annual"],
        "means": {"unemployment_rate": 6.0, "hpi_change_annual": 2.0, "gdp_growth_annual": 2.0},
        "stds": {"unemployment_rate": 1.5, "hpi_change_annual": 5.0, "gdp_growth_annual": 1.5},
        "correlation_matrix": corr,
    }


# ---------------------------------------------------------------------------
# _generate_correlated_scenarios_tensor — antithetic variates
# ---------------------------------------------------------------------------

def test_antithetic_output_shape(macro_stats):
    n = 1000
    scenarios, var_names, _, _ = _generate_correlated_scenarios_tensor(
        n_simulations=n, macro_stats=macro_stats, antithetic=True
    )
    assert scenarios.shape == (n, 3)


def test_antithetic_odd_n_simulations(macro_stats):
    """Odd n_simulations must still produce exactly n rows."""
    n = 999
    scenarios, _, _, _ = _generate_correlated_scenarios_tensor(
        n_simulations=n, macro_stats=macro_stats, antithetic=True
    )
    assert scenarios.shape[0] == n


def test_antithetic_reduces_variance(macro_stats):
    """Antithetic variates should reduce estimator variance on EL."""
    n = 4000
    rng = np.random.default_rng(0)

    def _run(antithetic: bool) -> float:
        scen, _, _, _ = _generate_correlated_scenarios_tensor(
            n_simulations=n, macro_stats=macro_stats, random_seed=42, antithetic=antithetic
        )
        # Proxy EL: mean of unemployment_rate (centred around its mean)
        arr = scen[:, 0].to("cpu").numpy()
        return float(np.mean(arr))

    # Repeat with different seeds and compare variance of the EL proxy across runs
    results_standard = []
    results_antithetic = []
    for seed in range(20):
        def _run_seed(antithetic: bool, seed: int) -> float:
            scen, _, _, _ = _generate_correlated_scenarios_tensor(
                n_simulations=n, macro_stats=macro_stats, random_seed=seed,
                antithetic=antithetic,
            )
            return float(scen[:, 0].to("cpu").numpy().mean())

        results_standard.append(_run_seed(False, seed))
        results_antithetic.append(_run_seed(True, seed))

    var_std = np.var(results_standard)
    var_anti = np.var(results_antithetic)
    # Antithetic variance must be strictly smaller (theoretical: ~50% reduction)
    assert var_anti < var_std, (
        f"Antithetic variance ({var_anti:.6f}) not < standard ({var_std:.6f})"
    )


def test_antithetic_paired_structure(macro_stats):
    """With identity correlation, Z_half and -Z_half map to symmetric scenarios."""
    n = 200  # even, so exact pairs
    scen, _, _, _ = _generate_correlated_scenarios_tensor(
        n_simulations=n, macro_stats=macro_stats, random_seed=7, antithetic=True
    )
    arr = scen.to("cpu").numpy()
    n_half = n // 2
    first_half = arr[:n_half]
    second_half = arr[n_half:]
    # The two halves should NOT be identical (one is the antipodal transform)
    assert not np.allclose(first_half, second_half)


def test_antithetic_false_matches_standard_shape(macro_stats):
    n = 500
    scen_std, _, _, _ = _generate_correlated_scenarios_tensor(
        n_simulations=n, macro_stats=macro_stats, antithetic=False
    )
    scen_anti, _, _, _ = _generate_correlated_scenarios_tensor(
        n_simulations=n, macro_stats=macro_stats, antithetic=True
    )
    assert scen_std.shape == scen_anti.shape


# ---------------------------------------------------------------------------
# Adaptive stopping helpers
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from run_monte_carlo_custom_backend import _is_converged, _quick_risk_metrics


def test_quick_risk_metrics_basic():
    rng = np.random.default_rng(1)
    losses = rng.exponential(scale=1e6, size=5000)
    m = _quick_risk_metrics(losses)
    assert "expected_loss" in m
    assert "var_99" in m
    assert "var_999" in m
    assert "es_99" in m
    assert m["var_99"] <= m["var_999"]
    assert m["es_99"] >= m["var_99"]


def test_is_converged_returns_true_when_within_tolerance():
    prev = {"expected_loss": 1_000_000.0, "var_99": 2_000_000.0}
    curr = {"expected_loss": 1_005_000.0, "var_99": 2_010_000.0}  # 0.5% and 0.5% change
    assert _is_converged(prev, curr, ["expected_loss", "var_99"], tolerance=0.01)


def test_is_converged_returns_false_when_outside_tolerance():
    prev = {"expected_loss": 1_000_000.0, "var_99": 2_000_000.0}
    curr = {"expected_loss": 1_050_000.0, "var_99": 2_010_000.0}  # 5% change in EL
    assert not _is_converged(prev, curr, ["expected_loss", "var_99"], tolerance=0.01)


def test_is_converged_near_zero_denominator():
    """When prev metric is near zero, fallback denom=1.0 prevents division error."""
    prev = {"expected_loss": 0.0}
    curr = {"expected_loss": 0.5}
    # 0.5 / 1.0 = 0.5 > 0.01 → not converged
    assert not _is_converged(prev, curr, ["expected_loss"], tolerance=0.01)


def test_is_converged_all_metrics_must_pass():
    """Convergence requires ALL listed metrics to be within tolerance."""
    prev = {"expected_loss": 1e6, "var_99": 2e6}
    curr = {"expected_loss": 1.001e6, "var_99": 2.5e6}  # var_99 changed 25%
    assert not _is_converged(prev, curr, ["expected_loss", "var_99"], tolerance=0.01)


def test_quick_risk_metrics_single_value():
    losses = np.array([1e6])
    m = _quick_risk_metrics(losses)
    assert np.isfinite(m["expected_loss"])
    assert np.isfinite(m["var_99"])
