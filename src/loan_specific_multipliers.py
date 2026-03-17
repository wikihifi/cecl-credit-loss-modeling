"""
Loan-specific stress sensitivity calibration for Monte Carlo simulation.

This module builds per-loan PD and LGD sensitivity vectors so scenario shocks
can propagate differently by borrower quality, leverage, geography, and
product structure.

Design goals
------------
- Use fields already present in the combined loan-level dataset.
- Keep baseline behavior stable by normalizing portfolio-weighted sensitivity
  means back to 1.0.
- Use MSA as the primary geography key with state fallback.
- Blend simple published-prior style monotonicity with empirical lifts from
  the project dataset so the resulting sensitivities are explainable and
  data-aligned.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


FICO_BUCKET_LABELS = ["<620", "620-660", "660-700", "700-740", "740+"]
LTV_BUCKET_LABELS = ["<60", "60-70", "70-80", "80-90", "90+"]

FICO_PD_PRIORS = {
    "<620": 1.45,
    "620-660": 1.25,
    "660-700": 1.10,
    "700-740": 0.95,
    "740+": 0.80,
}
FICO_LGD_PRIORS = {
    "<620": 1.10,
    "620-660": 1.06,
    "660-700": 1.02,
    "700-740": 0.98,
    "740+": 0.95,
}
LTV_PD_PRIORS = {
    "<60": 0.75,
    "60-70": 0.85,
    "70-80": 0.95,
    "80-90": 1.10,
    "90+": 1.35,
}
LTV_LGD_PRIORS = {
    "<60": 0.70,
    "60-70": 0.82,
    "70-80": 0.95,
    "80-90": 1.15,
    "90+": 1.45,
}

PRODUCT_FLAG_PRIORS = {
    "is_cashout_refi": (1.18, 1.08),
    "is_refi_nocashout": (1.05, 1.02),
    "is_investment_property": (1.30, 1.12),
    "is_second_home": (1.10, 1.05),
    "is_condo": (1.06, 1.05),
    "is_manufactured_housing": (1.25, 1.15),
    "is_multi_unit": (1.12, 1.06),
    "is_arm": (1.10, 1.04),
}


def _clean_key(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})
    return cleaned


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    if weights.sum() <= 0:
        return float(np.nanmean(values))
    return float(np.average(values, weights=weights))


def _normalize_with_weights(
    values: np.ndarray,
    weights: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), lower, upper)
    mean_value = _weighted_mean(clipped, weights)
    if not np.isfinite(mean_value) or mean_value <= 0:
        return np.ones_like(clipped)
    normalized = clipped / mean_value
    return np.clip(normalized, lower, upper)


def _blend_lift(empirical: float, prior: float, count: int, shrinkage: int) -> float:
    if not np.isfinite(empirical) or empirical <= 0:
        return prior
    weight = count / (count + shrinkage) if count > 0 else 0.0
    return prior + weight * (empirical - prior)


def _build_bucket_series(df: pd.DataFrame, bucket_col: str, raw_col: str, bucket_type: str) -> pd.Series:
    if bucket_col in df.columns:
        return df[bucket_col].astype("string")

    if raw_col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    values = pd.to_numeric(df[raw_col], errors="coerce")
    if bucket_type == "fico":
        bins = [0, 620, 660, 700, 740, 850]
        labels = FICO_BUCKET_LABELS
    else:
        bins = [0, 60, 70, 80, 90, 200]
        labels = LTV_BUCKET_LABELS

    bucketed = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
    return bucketed.astype("string")


def _bucket_lift_map(
    bucket_series: pd.Series,
    metric: pd.Series,
    prior_map: Dict[str, float],
    shrinkage: int,
    clip_range: Tuple[float, float],
) -> Dict[str, float]:
    portfolio_mean = metric.mean()
    if not np.isfinite(portfolio_mean) or portfolio_mean <= 0:
        return dict(prior_map)

    tmp = pd.DataFrame({"bucket": bucket_series.astype("string"), "metric": metric})
    tmp = tmp.dropna(subset=["bucket", "metric"])
    if tmp.empty:
        return dict(prior_map)

    stats = tmp.groupby("bucket", observed=False)["metric"].agg(["count", "mean"])
    lift_map: Dict[str, float] = {}
    for bucket, prior in prior_map.items():
        if bucket in stats.index:
            count = int(stats.at[bucket, "count"])
            empirical = float(stats.at[bucket, "mean"] / portfolio_mean)
            blended = _blend_lift(empirical, prior, count, shrinkage)
        else:
            blended = prior
        lift_map[bucket] = float(np.clip(blended, clip_range[0], clip_range[1]))
    return lift_map


def _apply_bucket_map(
    bucket_series: pd.Series,
    lift_map: Dict[str, float],
    default_value: float = 1.0,
) -> np.ndarray:
    mapped = bucket_series.astype("string").map(lift_map)
    return mapped.fillna(default_value).astype(float).to_numpy()


def _binary_flag(df: pd.DataFrame, col: str, expected_value: str | None = None) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    series = df[col]
    if expected_value is None:
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).to_numpy() == 1
    return series.astype("string").str.strip().eq(expected_value).to_numpy()


def _product_flag_lift(
    flag_values: np.ndarray,
    metric: pd.Series,
    prior_lift: float,
    shrinkage: int,
    clip_range: Tuple[float, float],
) -> float:
    if flag_values.sum() == 0:
        return 1.0

    metric_values = pd.to_numeric(metric, errors="coerce")
    portfolio_mean = metric_values.mean()
    if not np.isfinite(portfolio_mean) or portfolio_mean <= 0:
        return prior_lift

    flagged_metric = metric_values[flag_values]
    flagged_metric = flagged_metric.dropna()
    if flagged_metric.empty:
        return prior_lift

    empirical = float(flagged_metric.mean() / portfolio_mean)
    blended = _blend_lift(empirical, prior_lift, int(flag_values.sum()), shrinkage)
    return float(np.clip(blended, clip_range[0], clip_range[1]))


def _build_geography_lifts(
    key_series: pd.Series,
    metric: pd.Series,
    min_count: int,
    shrinkage: int,
    clip_range: Tuple[float, float],
) -> Tuple[Dict[str, float], Dict[str, int]]:
    portfolio_mean = metric.mean()
    tmp = pd.DataFrame({"key": _clean_key(key_series), "metric": metric})
    tmp = tmp.dropna(subset=["key", "metric"])
    if tmp.empty or not np.isfinite(portfolio_mean) or portfolio_mean <= 0:
        return {}, {}

    stats = tmp.groupby("key", observed=False)["metric"].agg(["count", "mean"])
    lift_map: Dict[str, float] = {}
    count_map: Dict[str, int] = {}
    for key, row in stats.iterrows():
        count = int(row["count"])
        count_map[str(key)] = count
        if count < min_count:
            continue
        empirical = float(row["mean"] / portfolio_mean)
        lift = _blend_lift(empirical, 1.0, count, shrinkage)
        lift_map[str(key)] = float(np.clip(lift, clip_range[0], clip_range[1]))
    return lift_map, count_map


def _get_product_flags(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    flags: Dict[str, np.ndarray] = {
        "is_cashout_refi": (
            _binary_flag(df, "is_cashout_refi")
            if "is_cashout_refi" in df.columns
            else _binary_flag(df, "loan_purpose", "C")
        ),
        "is_refi_nocashout": (
            _binary_flag(df, "is_refi_nocashout")
            if "is_refi_nocashout" in df.columns
            else _binary_flag(df, "loan_purpose", "N")
        ),
        "is_investment_property": (
            _binary_flag(df, "is_investment_property")
            if "is_investment_property" in df.columns
            else _binary_flag(df, "occupancy_status", "I")
        ),
        "is_second_home": (
            _binary_flag(df, "is_second_home")
            if "is_second_home" in df.columns
            else _binary_flag(df, "occupancy_status", "S")
        ),
        "is_condo": (
            _binary_flag(df, "is_condo")
            if "is_condo" in df.columns
            else _binary_flag(df, "property_type", "CO")
        ),
        "is_manufactured_housing": (
            _binary_flag(df, "is_manufactured_housing")
            if "is_manufactured_housing" in df.columns
            else _binary_flag(df, "property_type", "MH")
        ),
        "is_multi_unit": (
            _binary_flag(df, "is_multi_unit")
            if "is_multi_unit" in df.columns
            else pd.to_numeric(df.get("number_of_units", pd.Series(1, index=df.index)), errors="coerce")
            .fillna(1)
            .gt(1)
            .to_numpy()
        ),
    }
    if "amortization_type" in df.columns:
        flags["is_arm"] = (
            df["amortization_type"].astype("string").str.strip().fillna("FRM") != "FRM"
        ).to_numpy()
    else:
        flags["is_arm"] = np.zeros(len(df), dtype=bool)
    return flags


def build_loan_specific_sensitivities(
    df: pd.DataFrame,
    portfolio_upb: np.ndarray,
    pd_baseline: np.ndarray,
    lgd_baseline: np.ndarray,
    geography_primary: str = "msa",
    geography_fallback: str = "property_state",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build normalized per-loan PD and LGD sensitivity vectors.

    Returns
    -------
    tuple
        `(pd_sensitivity, lgd_sensitivity, summary_df)`
    """
    n_loans = len(df)
    if len(portfolio_upb) != n_loans or len(pd_baseline) != n_loans or len(lgd_baseline) != n_loans:
        raise ValueError("Sensitivity inputs must have the same length as df")

    base_weights = (
        np.asarray(portfolio_upb, dtype=float)
        * np.clip(np.asarray(pd_baseline, dtype=float), 1e-6, None)
        * np.clip(np.asarray(lgd_baseline, dtype=float), 1e-6, None)
    )

    default_source = df["default_flag"] if "default_flag" in df.columns else pd.Series(0.0, index=df.index)
    lgd_source = df["lgd"] if "lgd" in df.columns else pd.Series(np.nan, index=df.index)
    default_metric = pd.to_numeric(default_source, errors="coerce").fillna(0.0)
    lgd_metric = pd.to_numeric(lgd_source, errors="coerce")
    lgd_default_metric = lgd_metric.where(default_metric > 0)

    fico_bucket = _build_bucket_series(df, "fico_bucket", "borrower_credit_score", "fico")
    ltv_bucket = _build_bucket_series(df, "ltv_bucket", "original_ltv", "ltv")

    fico_pd_map = _bucket_lift_map(fico_bucket, default_metric, FICO_PD_PRIORS, 40_000, (0.70, 1.60))
    fico_lgd_map = _bucket_lift_map(fico_bucket, lgd_default_metric, FICO_LGD_PRIORS, 15_000, (0.85, 1.20))
    ltv_pd_map = _bucket_lift_map(ltv_bucket, default_metric, LTV_PD_PRIORS, 40_000, (0.70, 1.60))
    ltv_lgd_map = _bucket_lift_map(ltv_bucket, lgd_default_metric, LTV_LGD_PRIORS, 15_000, (0.70, 1.70))

    fico_pd = _apply_bucket_map(fico_bucket, fico_pd_map)
    fico_lgd = _apply_bucket_map(fico_bucket, fico_lgd_map)
    ltv_pd = _apply_bucket_map(ltv_bucket, ltv_pd_map)
    ltv_lgd = _apply_bucket_map(ltv_bucket, ltv_lgd_map)

    if "fico_missing" in df.columns:
        fico_missing = _binary_flag(df, "fico_missing")
        fico_pd = np.where(fico_missing, fico_pd * 1.05, fico_pd)
        fico_lgd = np.where(fico_missing, fico_lgd * 1.02, fico_lgd)

    if "has_mortgage_insurance" in df.columns:
        has_mi = _binary_flag(df, "has_mortgage_insurance")
        ltv_lgd = np.where(has_mi, 1.0 + 0.75 * (ltv_lgd - 1.0), ltv_lgd)

    primary_key = _clean_key(df.get(geography_primary, pd.Series(pd.NA, index=df.index)))
    fallback_key = _clean_key(df.get(geography_fallback, pd.Series(pd.NA, index=df.index)))

    geo_pd_primary_map, geo_pd_primary_counts = _build_geography_lifts(
        primary_key, default_metric, min_count=2_000, shrinkage=8_000, clip_range=(0.80, 1.35)
    )
    geo_pd_fallback_map, geo_pd_fallback_counts = _build_geography_lifts(
        fallback_key, default_metric, min_count=5_000, shrinkage=20_000, clip_range=(0.85, 1.25)
    )
    geo_lgd_primary_map, geo_lgd_primary_counts = _build_geography_lifts(
        primary_key, lgd_default_metric, min_count=250, shrinkage=1_000, clip_range=(0.85, 1.30)
    )
    geo_lgd_fallback_map, geo_lgd_fallback_counts = _build_geography_lifts(
        fallback_key, lgd_default_metric, min_count=750, shrinkage=3_000, clip_range=(0.90, 1.20)
    )

    geo_pd = np.ones(n_loans, dtype=float)
    geo_lgd = np.ones(n_loans, dtype=float)
    geo_source = np.full(n_loans, "neutral", dtype=object)

    primary_lookup = primary_key.map(geo_pd_primary_map)
    primary_counts = primary_key.map(geo_pd_primary_counts).fillna(0).astype(int)
    use_primary = primary_lookup.notna() & (primary_counts >= 2_000)
    geo_pd[use_primary.to_numpy()] = primary_lookup[use_primary].astype(float).to_numpy()
    geo_source[use_primary.to_numpy()] = "msa"

    fallback_lookup = fallback_key.map(geo_pd_fallback_map)
    fallback_counts = fallback_key.map(geo_pd_fallback_counts).fillna(0).astype(int)
    use_fallback = (~use_primary) & fallback_lookup.notna() & (fallback_counts >= 5_000)
    geo_pd[use_fallback.to_numpy()] = fallback_lookup[use_fallback].astype(float).to_numpy()
    geo_source[use_fallback.to_numpy()] = "state"

    primary_lookup_lgd = primary_key.map(geo_lgd_primary_map)
    primary_counts_lgd = primary_key.map(geo_lgd_primary_counts).fillna(0).astype(int)
    use_primary_lgd = primary_lookup_lgd.notna() & (primary_counts_lgd >= 250)
    geo_lgd[use_primary_lgd.to_numpy()] = primary_lookup_lgd[use_primary_lgd].astype(float).to_numpy()

    fallback_lookup_lgd = fallback_key.map(geo_lgd_fallback_map)
    fallback_counts_lgd = fallback_key.map(geo_lgd_fallback_counts).fillna(0).astype(int)
    use_fallback_lgd = (~use_primary_lgd) & fallback_lookup_lgd.notna() & (fallback_counts_lgd >= 750)
    geo_lgd[use_fallback_lgd.to_numpy()] = fallback_lookup_lgd[use_fallback_lgd].astype(float).to_numpy()

    product_flags = _get_product_flags(df)
    product_pd = np.ones(n_loans, dtype=float)
    product_lgd = np.ones(n_loans, dtype=float)
    for flag_name, (pd_prior, lgd_prior) in PRODUCT_FLAG_PRIORS.items():
        flag_values = product_flags.get(flag_name, np.zeros(n_loans, dtype=bool))
        pd_lift = _product_flag_lift(
            flag_values=flag_values,
            metric=default_metric,
            prior_lift=pd_prior,
            shrinkage=20_000,
            clip_range=(0.90, 1.45),
        )
        lgd_lift = _product_flag_lift(
            flag_values=flag_values,
            metric=lgd_default_metric,
            prior_lift=lgd_prior,
            shrinkage=6_000,
            clip_range=(0.95, 1.25),
        )
        product_pd = np.where(flag_values, product_pd * pd_lift, product_pd)
        product_lgd = np.where(flag_values, product_lgd * lgd_lift, product_lgd)

    pd_raw = fico_pd * ltv_pd * geo_pd * product_pd
    lgd_raw = fico_lgd * ltv_lgd * geo_lgd * product_lgd

    pd_sensitivity = _normalize_with_weights(pd_raw, base_weights, 0.60, 1.85)
    lgd_sensitivity = _normalize_with_weights(lgd_raw, base_weights, 0.60, 2.00)

    summary_df = pd.DataFrame([
        {
            "metric": "pd_sensitivity",
            "weighted_mean": _weighted_mean(pd_sensitivity, base_weights),
            "min": float(np.min(pd_sensitivity)),
            "p05": float(np.percentile(pd_sensitivity, 5)),
            "p50": float(np.percentile(pd_sensitivity, 50)),
            "p95": float(np.percentile(pd_sensitivity, 95)),
            "max": float(np.max(pd_sensitivity)),
            "msa_share": float((geo_source == "msa").mean()),
            "state_share": float((geo_source == "state").mean()),
            "neutral_share": float((geo_source == "neutral").mean()),
        },
        {
            "metric": "lgd_sensitivity",
            "weighted_mean": _weighted_mean(lgd_sensitivity, base_weights),
            "min": float(np.min(lgd_sensitivity)),
            "p05": float(np.percentile(lgd_sensitivity, 5)),
            "p50": float(np.percentile(lgd_sensitivity, 50)),
            "p95": float(np.percentile(lgd_sensitivity, 95)),
            "max": float(np.max(lgd_sensitivity)),
            "msa_share": float((geo_source == "msa").mean()),
            "state_share": float((geo_source == "state").mean()),
            "neutral_share": float((geo_source == "neutral").mean()),
        },
    ])

    return pd_sensitivity.astype(np.float32), lgd_sensitivity.astype(np.float32), summary_df
