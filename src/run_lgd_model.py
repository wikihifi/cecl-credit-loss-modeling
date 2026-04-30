"""
Full LGD Model Training Pipeline
==================================

Trains and validates LGD models on the full 278K defaulted-loan dataset.

Author: Saurabh Chavan
"""

import argparse
import gc
import json
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from lgd_model import (
    prepare_lgd_dataset,
    train_lgd_ols,
    train_lgd_xgboost,
    validate_lgd_model,
    compute_lgd_by_segment,
    macro_sensitivity_check,
    LGD_FEATURES,
)

COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
MODEL_DIR = project_root / "models"


def _validate_lgd_training_inputs(data: dict, targets: dict) -> None:
    """Fail fast when the selected dataset cannot support supervised LGD training."""
    checks = [
        ("train", data["train"], targets["y_train"]),
        ("validation", data["val"], targets["y_val"]),
        ("test", data["test"], targets["y_test"]),
    ]
    problems = []
    for split_name, x_split, y_split in checks:
        if len(x_split) == 0 or len(y_split) == 0:
            problems.append(f"{split_name}: no defaulted loans with valid LGD")
            continue
        if y_split.notna().sum() == 0:
            problems.append(f"{split_name}: LGD target is entirely null")

    if problems:
        joined = "; ".join(problems)
        raise ValueError(
            "Selected dataset is not compatible with LGD model training. "
            f"Target distribution check failed: {joined}. "
            "LGD training requires defaulted loans with valid realized LGD values in the labeled splits. "
            "Use a labeled engineered training dataset, not a scoring-only fullbook."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="LGD Model Training Pipeline")
    parser.add_argument(
        "--portfolio-path", default=None,
        help="Path to the engineered loan-level parquet to train on.",
    )
    parser.add_argument(
        "--bundle-dir", default=None,
        help="Directory to write model bundle artifacts.",
    )
    parser.add_argument(
        "--also-write-global", action="store_true", default=True,
        help="Also write artifacts to models/ global paths for backward compatibility.",
    )
    parser.add_argument(
        "--no-write-global", action="store_true", default=False,
        help="Skip writing to global models/ paths.",
    )
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument("--dataset-fingerprint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    portfolio_path = Path(args.portfolio_path) if args.portfolio_path else COMBINED_PATH
    write_global = args.also_write_global and not args.no_write_global

    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = args.dataset_fingerprint or "unknown"
        bundle_id = f"lgd_{fp[:8]}_{ts}"
        bundle_dir = MODEL_DIR / "model_bundles" / bundle_id

    bundle_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - FULL LGD MODEL PIPELINE")
    print("=" * 70)
    print(f"Portfolio: {portfolio_path}")
    print(f"Bundle dir: {bundle_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load and prepare data
    # ------------------------------------------------------------------
    print("\nStep 1: Loading data...")
    df = pd.read_parquet(str(portfolio_path))
    data, targets, features = prepare_lgd_dataset(df)
    del df
    gc.collect()

    _validate_lgd_training_inputs(data, targets)

    X_train = data["train"]
    X_val = data["val"]
    X_test = data["test"]
    y_train = targets["y_train"]
    y_val = targets["y_val"]
    y_test = targets["y_test"]

    # ------------------------------------------------------------------
    # Step 2: Train OLS model (primary)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2: Training OLS LGD model on full training data")
    print("=" * 70)

    ols_model = train_lgd_ols(X_train, y_train)

    # ------------------------------------------------------------------
    # Step 3: Train XGBoost model (challenger)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3: Training XGBoost LGD model on full training data")
    print("=" * 70)

    xgb_available = False
    xgb_error = None
    xgb_model = None
    xgb_train_m = xgb_val_m = xgb_test_m = None
    xgb_train_p = xgb_val_p = xgb_test_p = None
    try:
        xgb_model = train_lgd_xgboost(X_train, y_train)
        xgb_available = True
    except Exception as exc:
        xgb_error = str(exc)
        print("  WARNING: XGBoost challenger unavailable; continuing with OLS only.")
        print(f"  XGBoost error: {exc}")

    # ------------------------------------------------------------------
    # Step 4: Full validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Full validation")
    print("=" * 70)

    print("\n--- OLS Model ---")
    ols_train_m, ols_train_p = validate_lgd_model(ols_model, X_train, y_train, "Train")
    ols_val_m, ols_val_p = validate_lgd_model(ols_model, X_val, y_val, "Validation")
    ols_test_m, ols_test_p = validate_lgd_model(ols_model, X_test, y_test, "Test")

    if xgb_available and xgb_model is not None:
        print("\n--- XGBoost Model ---")
        xgb_train_m, xgb_train_p = validate_lgd_model(xgb_model, X_train, y_train, "Train")
        xgb_val_m, xgb_val_p = validate_lgd_model(xgb_model, X_val, y_val, "Validation")
        xgb_test_m, xgb_test_p = validate_lgd_model(xgb_model, X_test, y_test, "Test")

    # ------------------------------------------------------------------
    # Step 5: Model comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LGD MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    if xgb_available and xgb_train_m and xgb_val_m and xgb_test_m:
        print(f"\n  {'Metric':<25s} {'OLS':>12s} {'XGBoost':>12s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'Train R2':<25s} {ols_train_m['r2']:>12.4f} {xgb_train_m['r2']:>12.4f}")
        print(f"  {'Validation R2':<25s} {ols_val_m['r2']:>12.4f} {xgb_val_m['r2']:>12.4f}")
        print(f"  {'Test R2':<25s} {ols_test_m['r2']:>12.4f} {xgb_test_m['r2']:>12.4f}")
        print(f"  {'Train RMSE':<25s} {ols_train_m['rmse']:>12.4f} {xgb_train_m['rmse']:>12.4f}")
        print(f"  {'Validation RMSE':<25s} {ols_val_m['rmse']:>12.4f} {xgb_val_m['rmse']:>12.4f}")
        print(f"  {'Test RMSE':<25s} {ols_test_m['rmse']:>12.4f} {xgb_test_m['rmse']:>12.4f}")
        print(f"  {'Train Calibration':<25s} {ols_train_m['mean_predicted']/ols_train_m['mean_actual']:>12.4f} {xgb_train_m['mean_predicted']/xgb_train_m['mean_actual']:>12.4f}")
        print(f"  {'Val Calibration':<25s} {ols_val_m['mean_predicted']/ols_val_m['mean_actual']:>12.4f} {xgb_val_m['mean_predicted']/xgb_val_m['mean_actual']:>12.4f}")
        print(f"  {'Test Calibration':<25s} {ols_test_m['mean_predicted']/ols_test_m['mean_actual']:>12.4f} {xgb_test_m['mean_predicted']/xgb_test_m['mean_actual']:>12.4f}")
    else:
        print("\n  XGBoost challenger was skipped.")
        print(f"  OLS Validation R2:   {ols_val_m['r2']:.4f}")
        print(f"  OLS Validation RMSE: {ols_val_m['rmse']:.4f}")
        print(f"  OLS Validation MAE:  {ols_val_m['mae']:.4f}")

    # ------------------------------------------------------------------
    # Step 6: Segment analysis (OLS model on validation set)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 6: Segment analysis")
    print(f"{'='*70}")

    # LTV segments
    ltv_buckets = pd.cut(
        X_val["original_ltv"].values,
        bins=[0, 60, 70, 80, 90, 200],
        labels=["<60", "60-70", "70-80", "80-90", "90+"],
    )
    compute_lgd_by_segment(y_val.values, ols_val_p, ltv_buckets, "LTV Bucket (OLS)")

    # FICO segments
    fico_buckets = pd.cut(
        X_val["borrower_credit_score"].values,
        bins=[0, 620, 660, 700, 740, 850],
        labels=["<620", "620-660", "660-700", "700-740", "740+"],
    )
    compute_lgd_by_segment(y_val.values, ols_val_p, fico_buckets, "FICO Bucket (OLS)")

    # MI segments
    compute_lgd_by_segment(
        y_val.values, ols_val_p,
        X_val["has_mortgage_insurance"].values, "Has MI (OLS)"
    )

    # ------------------------------------------------------------------
    # Step 7: Macro sensitivity (OLS model)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 7: Macro sensitivity analysis")
    print(f"{'='*70}")

    X_baseline = X_val.head(2000).copy()

    macro_sensitivity_check(
        ols_model, X_baseline,
        "unemployment_rate", [4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        features,
    )

    macro_sensitivity_check(
        ols_model, X_baseline,
        "hpi_national", [120, 140, 160, 180, 200, 220],
        features,
    )

    # ------------------------------------------------------------------
    # Step 8: Save models and artifacts
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Step 8: Saving models and artifacts")
    print(f"{'='*70}")

    import shutil

    bundle_dir.mkdir(parents=True, exist_ok=True)
    if write_global:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(ols_model, bundle_dir / "lgd_ols.pkl")
    if xgb_available and xgb_model is not None:
        joblib.dump(xgb_model, bundle_dir / "lgd_xgboost.pkl")
    if write_global:
        shutil.copy2(bundle_dir / "lgd_ols.pkl", MODEL_DIR / "lgd_ols.pkl")
        if xgb_available:
            shutil.copy2(bundle_dir / "lgd_xgboost.pkl", MODEL_DIR / "lgd_xgboost.pkl")
    print(f"  Saved LGD models to {bundle_dir}")

    # Save OLS coefficients
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": ols_model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(bundle_dir / "lgd_ols_coefficients.csv", index=False)
    if write_global:
        shutil.copy2(bundle_dir / "lgd_ols_coefficients.csv", MODEL_DIR / "lgd_ols_coefficients.csv")

    # Save XGBoost importance
    if xgb_available and xgb_model is not None:
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": xgb_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(bundle_dir / "lgd_xgb_feature_importance.csv", index=False)
        if write_global:
            shutil.copy2(bundle_dir / "lgd_xgb_feature_importance.csv", MODEL_DIR / "lgd_xgb_feature_importance.csv")

    # Save validation summary
    summary = {
        "ols": {
            "train_r2": ols_train_m["r2"],
            "val_r2": ols_val_m["r2"],
            "test_r2": ols_test_m["r2"],
            "train_rmse": ols_train_m["rmse"],
            "val_rmse": ols_val_m["rmse"],
            "test_rmse": ols_test_m["rmse"],
            "train_mae": ols_train_m["mae"],
            "val_mae": ols_val_m["mae"],
            "test_mae": ols_test_m["mae"],
        },
        "xgboost": {
            "train_r2": xgb_train_m["r2"] if xgb_train_m else np.nan,
            "val_r2": xgb_val_m["r2"] if xgb_val_m else np.nan,
            "test_r2": xgb_test_m["r2"] if xgb_test_m else np.nan,
            "train_rmse": xgb_train_m["rmse"] if xgb_train_m else np.nan,
            "val_rmse": xgb_val_m["rmse"] if xgb_val_m else np.nan,
            "test_rmse": xgb_test_m["rmse"] if xgb_test_m else np.nan,
            "train_mae": xgb_train_m["mae"] if xgb_train_m else np.nan,
            "val_mae": xgb_val_m["mae"] if xgb_val_m else np.nan,
            "test_mae": xgb_test_m["mae"] if xgb_test_m else np.nan,
        },
    }
    pd.DataFrame(summary).to_csv(bundle_dir / "lgd_validation_summary.csv")
    if write_global:
        shutil.copy2(bundle_dir / "lgd_validation_summary.csv", MODEL_DIR / "lgd_validation_summary.csv")

    # Save feature list
    with open(bundle_dir / "lgd_features.txt", "w") as f:
        for feat in features:
            f.write(feat + "\n")
    if write_global:
        shutil.copy2(bundle_dir / "lgd_features.txt", MODEL_DIR / "lgd_features.txt")

    # Write bundle metadata
    meta = {
        "bundle_id": bundle_dir.name,
        "model_type": "lgd",
        "dataset_path": str(portfolio_path.resolve()),
        "dataset_label": args.dataset_label or portfolio_path.name,
        "dataset_fingerprint": args.dataset_fingerprint or "",
        "training_timestamp": datetime.now().isoformat(),
        "lgd_features": features,
        "xgboost_available": xgb_available,
        "xgboost_error": xgb_error,
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  Wrote bundle metadata: {bundle_dir / 'metadata.json'}")
    print(f"  Saved all artifacts")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"LGD MODEL PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
