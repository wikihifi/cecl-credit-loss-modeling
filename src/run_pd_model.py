"""
Full PD Model Training Pipeline
================================

Trains and validates PD models on the full 3.8 million loan dataset.

Steps:
1. Load combined loan-level dataset
2. Prepare train/validation/test splits
3. Select non-redundant features, compute WoE/IV on full training data
4. Train logistic regression (primary) and XGBoost (challenger)
5. Run full validation (AUC, Gini, KS, calibration, PSI)
6. Save models and results

Run this AFTER test_pd_model.py passes all tests.

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

from pd_model import (
    calculate_woe_iv_all_features,
    apply_woe_transformation,
    train_logistic_regression,
    train_xgboost_challenger,
    run_full_validation,
    prepare_features_for_xgboost,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMBINED_PATH = project_root / "data" / "processed" / "loan_level_combined.parquet"
MODEL_DIR = project_root / "models"
TARGET = "default_flag"


def _validate_target_distribution(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail fast when the selected dataset cannot support supervised PD training."""
    split_frames = {
        "train": train,
        "validation": val,
        "test": test,
    }

    problems = []
    for split_name, split_df in split_frames.items():
        if TARGET not in split_df.columns:
            problems.append(f"{split_name}: missing target column `{TARGET}`")
            continue

        values = split_df[TARGET].dropna()
        unique_vals = set(values.unique().tolist())
        if not unique_vals:
            problems.append(f"{split_name}: target column `{TARGET}` is empty")
            continue
        if unique_vals <= {0}:
            problems.append(f"{split_name}: no positive defaults (all `{TARGET}` = 0)")
        elif unique_vals <= {1}:
            problems.append(f"{split_name}: no non-default observations (all `{TARGET}` = 1)")

    if problems:
        joined = "; ".join(problems)
        raise ValueError(
            "Selected dataset is not compatible with PD model training. "
            f"Target distribution check failed: {joined}. "
            "PD training requires both defaulted and non-defaulted loans in the training data. "
            "Use a labeled engineered training dataset, not an unlabeled scoring-only fullbook."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="PD Model Training Pipeline")
    parser.add_argument(
        "--portfolio-path", default=None,
        help="Path to the engineered loan-level parquet to train on. "
             "Defaults to data/processed/loan_level_combined.parquet.",
    )
    parser.add_argument(
        "--bundle-dir", default=None,
        help="Directory to write model bundle artifacts. "
             "Auto-generated under models/model_bundles/ if not provided.",
    )
    parser.add_argument(
        "--also-write-global", action="store_true", default=True,
        help="Also write artifacts to models/ global paths for backward compatibility.",
    )
    parser.add_argument(
        "--no-write-global", action="store_true", default=False,
        help="Skip writing to global models/ paths.",
    )
    parser.add_argument(
        "--dataset-label", default=None,
        help="Human-readable label for the dataset (stored in bundle metadata).",
    )
    parser.add_argument(
        "--dataset-fingerprint", default=None,
        help="Precomputed dataset fingerprint (stored in bundle metadata).",
    )
    return parser.parse_args()

# Feature selection: curated list removing redundant features.
#
# Removed redundancies:
#   fico_bucket: redundant with borrower_credit_score (same info, less granular)
#   ltv_bucket: redundant with original_ltv
#   dti_bucket: redundant with dti
#   fico_x_ltv: partially redundant with borrower_credit_score + original_ltv
#   fico_x_dti: partially redundant with borrower_credit_score + dti
#   number_of_borrowers: redundant with has_coborrower (same signal)
#
# Kept interactions:
#   fico_x_unemployment: captures the key macro-borrower interaction
#   (weak borrowers are disproportionately affected by job losses)
#
# These decisions are documented for model governance (Phase 9).
SELECTED_FEATURES = [
    # Borrower risk factors
    "borrower_credit_score",     # IV ~0.60, strongest single predictor
    "dti",                       # IV ~0.14, payment capacity
    "has_coborrower",            # IV ~0.05, income stability
    # Collateral risk factors
    "original_ltv",              # IV ~0.18, equity/leverage
    "original_cltv",             # IV ~0.16, total leverage including second liens
    "has_mortgage_insurance",    # IV ~0.05, loss mitigation
    # Loan characteristics
    "original_interest_rate",    # IV ~0.15, pricing reflects risk
    "original_loan_term",        # IV ~0.12, term structure
    "is_cashout_refi",           # IV ~0.03, equity extraction
    # Macroeconomic conditions at origination
    "fico_x_unemployment",       # IV ~0.59, borrower-macro interaction
    "gdp_growth_pct",            # IV ~0.02, economic cycle
    "fed_funds_rate",            # IV ~0.02, monetary policy
    "hpi_national",              # IV ~0.02, housing market conditions
]


def main():
    args = parse_args()

    # Resolve paths
    portfolio_path = Path(args.portfolio_path) if args.portfolio_path else COMBINED_PATH
    write_global = args.also_write_global and not args.no_write_global

    # Resolve bundle directory
    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = args.dataset_fingerprint or "unknown"
        bundle_id = f"pd_{fp[:8]}_{ts}"
        bundle_dir = MODEL_DIR / "model_bundles" / bundle_id

    bundle_dir.mkdir(parents=True, exist_ok=True)

    t_pipeline_start = time.time()

    print("=" * 70)
    print("CECL CREDIT RISK PROJECT - FULL PD MODEL PIPELINE")
    print("=" * 70)
    print(f"Portfolio: {portfolio_path}")
    print(f"Bundle dir: {bundle_dir}")

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\nStep 1: Loading combined loan-level dataset...")
    df = pd.read_parquet(str(portfolio_path))
    print(f"  Loaded {len(df):,} loans, {df.shape[1]} columns")

    # ------------------------------------------------------------------
    # Step 2: Prepare splits
    # ------------------------------------------------------------------
    print("\nStep 2: Preparing train/validation/test splits...")

    # Move pre-2005 "unknown" loans into training set
    df.loc[df["data_split"] == "unknown", "data_split"] = "train"

    train = df[df["data_split"] == "train"].copy()
    val = df[df["data_split"] == "validation"].copy()
    test = df[df["data_split"] == "test"].copy()

    print(f"  Train:      {len(train):>10,} loans, DR: {train[TARGET].mean():.4f}")
    print(f"  Validation: {len(val):>10,} loans, DR: {val[TARGET].mean():.4f}")
    print(f"  Test:       {len(test):>10,} loans, DR: {test[TARGET].mean():.4f}")

    _validate_target_distribution(train, val, test)

    # Free the full dataframe
    del df
    gc.collect()

    y_train = train[TARGET]
    y_val = val[TARGET]
    y_test = test[TARGET]

    # ------------------------------------------------------------------
    # Step 3: WoE/IV on full training data
    # ------------------------------------------------------------------
    print("\nStep 3: Computing WoE/IV on full training data...")
    print(f"  Features: {len(SELECTED_FEATURES)}")
    print(f"  Training samples: {len(train):,}")

    iv_summary, woe_results = calculate_woe_iv_all_features(
        train, SELECTED_FEATURES, TARGET
    )

    print(f"\n  IV Summary (full training data):")
    print(f"  {'Feature':<35s} {'IV':>10s} {'Interpretation':<15s}")
    print(f"  {'-'*35} {'-'*10} {'-'*15}")
    for _, row in iv_summary.iterrows():
        print(f"  {row['feature']:<35s} {row['iv']:>10.4f} {row['interpretation']:<15s}")

    # Drop any features with IV < 0.02 on full data
    final_features = iv_summary.loc[iv_summary["iv"] > 0.02, "feature"].tolist()
    dropped = [f for f in SELECTED_FEATURES if f not in final_features]
    if dropped:
        print(f"\n  Dropped features (IV < 0.02 on full data): {dropped}")
    print(f"  Final feature count: {len(final_features)}")
    if not final_features:
        raise ValueError(
            "PD training cannot continue because no features passed the IV > 0.02 filter. "
            "This usually means the selected dataset is not suitable for supervised PD training "
            "(for example, no default events in the labeled splits) or the engineered feature/target "
            "mapping is incompatible with the training pipeline."
        )

    # ------------------------------------------------------------------
    # Step 4: WoE transformation
    # ------------------------------------------------------------------
    print("\nStep 4: Applying WoE transformation...")

    X_train_woe = apply_woe_transformation(train, woe_results, final_features)
    X_val_woe = apply_woe_transformation(val, woe_results, final_features)
    X_test_woe = apply_woe_transformation(test, woe_results, final_features)

    print(f"  Train WoE shape: {X_train_woe.shape}")
    print(f"  Val WoE shape:   {X_val_woe.shape}")
    print(f"  Test WoE shape:  {X_test_woe.shape}")

    # Verify no NaN
    for label, xdf in [("Train", X_train_woe), ("Val", X_val_woe), ("Test", X_test_woe)]:
        n_nan = xdf.isna().sum().sum()
        if n_nan > 0:
            print(f"  WARNING: {label} has {n_nan} NaN values in WoE features")
        else:
            print(f"  {label}: 0 NaN values (clean)")

    # ------------------------------------------------------------------
    # Step 5: Train Logistic Regression (primary model)
    # ------------------------------------------------------------------
    # Note: We do NOT use class_weight='balanced' here because it distorts
    # predicted probabilities. Instead, the model learns the natural
    # default rate from the training data, producing properly calibrated
    # PD estimates. This is critical for CECL reserves -- the predicted
    # PD is directly used to compute dollar loss reserves.
    print("\nStep 5: Training models...")

    from sklearn.linear_model import LogisticRegression

    print("\n--- Logistic Regression (Primary Model) ---")
    lr_model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight=None,   # No reweighting: preserves calibration
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        n_jobs=-1,
    )
    t0 = time.time()
    lr_model.fit(X_train_woe, y_train)
    print(f"  Training completed in {time.time() - t0:.1f} seconds")

    # Print coefficients
    coef_df = pd.DataFrame({
        "feature": X_train_woe.columns,
        "coefficient": lr_model.coef_[0],
    }).sort_values("coefficient", key=abs, ascending=False)

    print(f"\n  Coefficients:")
    print(f"  {'Feature':<40s} {'Coefficient':>12s}")
    print(f"  {'-'*40} {'-'*12}")
    for _, row in coef_df.iterrows():
        print(f"  {row['feature']:<40s} {row['coefficient']:>12.4f}")
    print(f"  {'Intercept':<40s} {lr_model.intercept_[0]:>12.4f}")

    # ------------------------------------------------------------------
    # Step 6: Train XGBoost (challenger model)
    # ------------------------------------------------------------------
    print("\n--- XGBoost (Challenger Model) ---")
    xgb_available = False
    xgb_error = None
    xgb_model = None
    xgb_encoders = None
    xgb_results = None
    importance = pd.DataFrame(columns=["feature", "importance"])
    X_train_xgb = X_val_xgb = X_test_xgb = None

    try:
        X_train_xgb, xgb_encoders = prepare_features_for_xgboost(train, final_features)
        X_val_xgb, _ = prepare_features_for_xgboost(val, final_features)
        X_test_xgb, _ = prepare_features_for_xgboost(test, final_features)

        import xgboost as xgb

        n_default = y_train.sum()
        n_non_default = len(y_train) - n_default
        scale_pos_weight = n_non_default / n_default

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        t0 = time.time()
        xgb_model.fit(X_train_xgb, y_train, verbose=False)
        print(f"  Training completed in {time.time() - t0:.1f} seconds")

        # Feature importance
        importance = pd.DataFrame({
            "feature": X_train_xgb.columns,
            "importance": xgb_model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print(f"\n  Feature Importances:")
        print(f"  {'Feature':<40s} {'Importance':>12s}")
        print(f"  {'-'*40} {'-'*12}")
        for _, row in importance.iterrows():
            print(f"  {row['feature']:<40s} {row['importance']:>12.4f}")
        xgb_available = True
    except Exception as exc:
        xgb_error = str(exc)
        print("  WARNING: XGBoost challenger unavailable; continuing with Logistic Regression only.")
        print(f"  XGBoost error: {exc}")

    # ------------------------------------------------------------------
    # Step 7: Full validation
    # ------------------------------------------------------------------
    print("\n\nStep 7: Running full validation...")

    lr_results = run_full_validation(
        lr_model,
        X_train_woe, y_train,
        X_val_woe, y_val,
        X_test_woe, y_test,
        model_name="Logistic Regression (Primary)",
    )

    if xgb_available:
        xgb_results = run_full_validation(
            xgb_model,
            X_train_xgb, y_train,
            X_val_xgb, y_val,
            X_test_xgb, y_test,
            model_name="XGBoost (Challenger)",
        )

    # ------------------------------------------------------------------
    # Step 8: Model comparison summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")
    if xgb_available and xgb_results is not None:
        print(f"\n  {'Metric':<25s} {'Logistic Reg':>15s} {'XGBoost':>15s}")
        print(f"  {'-'*25} {'-'*15} {'-'*15}")

        for split_key, split_label in [
            ("train_metrics", "Train AUC"),
            ("val_metrics", "Validation AUC"),
            ("test_metrics", "Test AUC"),
        ]:
            lr_auc = lr_results[split_key]["auc"]
            xgb_auc = xgb_results[split_key]["auc"]
            print(f"  {split_label:<25s} {lr_auc:>15.4f} {xgb_auc:>15.4f}")

        for split_key, split_label in [
            ("train_metrics", "Train KS"),
            ("val_metrics", "Validation KS"),
            ("test_metrics", "Test KS"),
        ]:
            lr_ks = lr_results[split_key]["ks"]
            xgb_ks = xgb_results[split_key]["ks"]
            print(f"  {split_label:<25s} {lr_ks:>15.4f} {xgb_ks:>15.4f}")

        print(f"  {'PSI (Train vs Val)':<25s} {lr_results['psi_val']:>15.4f} {xgb_results['psi_val']:>15.4f}")
        print(f"  {'PSI (Train vs Test)':<25s} {lr_results['psi_test']:>15.4f} {xgb_results['psi_test']:>15.4f}")
    else:
        print("\n  XGBoost challenger was skipped.")
        print(f"  Logistic Regression Validation AUC: {lr_results['val_metrics']['auc']:.4f}")
        print(f"  Logistic Regression Validation KS:  {lr_results['val_metrics']['ks']:.4f}")
        print(f"  Logistic Regression PSI (Val):     {lr_results['psi_val']:.4f}")

    # ------------------------------------------------------------------
    # Step 9: Save models and artifacts
    # ------------------------------------------------------------------
    print(f"\n\nStep 9: Saving models and artifacts...")

    # Helper: write to bundle dir, optionally mirror to MODEL_DIR
    import shutil

    def _save(src_path: Path, filename: str) -> None:
        """Write to bundle dir; optionally mirror to global MODEL_DIR."""
        bundle_dest = bundle_dir / filename
        # src_path already points into bundle_dir in some cases; handle both
        if src_path != bundle_dest:
            shutil.copy2(src_path, bundle_dest)
        if write_global:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle_dest, MODEL_DIR / filename)

    # Save models to bundle
    joblib.dump(lr_model, bundle_dir / "pd_logistic_regression.pkl")
    joblib.dump(woe_results, bundle_dir / "woe_results.pkl")
    if xgb_available and xgb_model is not None and xgb_encoders is not None:
        joblib.dump(xgb_model, bundle_dir / "pd_xgboost.pkl")
        joblib.dump(xgb_encoders, bundle_dir / "xgb_label_encoders.pkl")
    if write_global:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        files_to_copy = ["pd_logistic_regression.pkl", "woe_results.pkl"]
        if xgb_available:
            files_to_copy += ["pd_xgboost.pkl", "xgb_label_encoders.pkl"]
        for fname in files_to_copy:
            shutil.copy2(bundle_dir / fname, MODEL_DIR / fname)
    print(f"  Saved models to {bundle_dir}")

    # Save IV summary
    iv_summary.to_csv(bundle_dir / "iv_summary.csv", index=False)
    if write_global:
        shutil.copy2(bundle_dir / "iv_summary.csv", MODEL_DIR / "iv_summary.csv")

    # Save coefficient table
    coef_df.to_csv(bundle_dir / "lr_coefficients.csv", index=False)
    if write_global:
        shutil.copy2(bundle_dir / "lr_coefficients.csv", MODEL_DIR / "lr_coefficients.csv")

    # Save feature importance
    if xgb_available:
        importance.to_csv(bundle_dir / "xgb_feature_importance.csv", index=False)
    if write_global and xgb_available:
        shutil.copy2(bundle_dir / "xgb_feature_importance.csv", MODEL_DIR / "xgb_feature_importance.csv")

    # Save selected features list
    with open(bundle_dir / "selected_features.txt", "w") as f:
        for feat in final_features:
            f.write(feat + "\n")
    if write_global:
        shutil.copy2(bundle_dir / "selected_features.txt", MODEL_DIR / "selected_features.txt")

    # Save calibration tables
    lr_results["calibration_table_val"].to_csv(bundle_dir / "lr_calibration_validation.csv", index=False)
    if xgb_available and xgb_results is not None:
        xgb_results["calibration_table_val"].to_csv(bundle_dir / "xgb_calibration_validation.csv", index=False)
    if write_global:
        shutil.copy2(bundle_dir / "lr_calibration_validation.csv", MODEL_DIR / "lr_calibration_validation.csv")
        if xgb_available:
            shutil.copy2(bundle_dir / "xgb_calibration_validation.csv", MODEL_DIR / "xgb_calibration_validation.csv")

    # Save validation summary
    summary = {
        "logistic_regression": {
            "train_auc": lr_results["train_metrics"]["auc"],
            "val_auc": lr_results["val_metrics"]["auc"],
            "test_auc": lr_results["test_metrics"]["auc"],
            "train_gini": lr_results["train_metrics"]["gini"],
            "val_gini": lr_results["val_metrics"]["gini"],
            "test_gini": lr_results["test_metrics"]["gini"],
            "train_ks": lr_results["train_metrics"]["ks"],
            "val_ks": lr_results["val_metrics"]["ks"],
            "test_ks": lr_results["test_metrics"]["ks"],
            "psi_val": lr_results["psi_val"],
            "psi_test": lr_results["psi_test"],
        },
        "xgboost": {
            "train_auc": xgb_results["train_metrics"]["auc"] if xgb_results else np.nan,
            "val_auc": xgb_results["val_metrics"]["auc"] if xgb_results else np.nan,
            "test_auc": xgb_results["test_metrics"]["auc"] if xgb_results else np.nan,
            "train_gini": xgb_results["train_metrics"]["gini"] if xgb_results else np.nan,
            "val_gini": xgb_results["val_metrics"]["gini"] if xgb_results else np.nan,
            "test_gini": xgb_results["test_metrics"]["gini"] if xgb_results else np.nan,
            "train_ks": xgb_results["train_metrics"]["ks"] if xgb_results else np.nan,
            "val_ks": xgb_results["val_metrics"]["ks"] if xgb_results else np.nan,
            "test_ks": xgb_results["test_metrics"]["ks"] if xgb_results else np.nan,
            "psi_val": xgb_results["psi_val"] if xgb_results else np.nan,
            "psi_test": xgb_results["psi_test"] if xgb_results else np.nan,
        },
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(bundle_dir / "validation_summary.csv")
    if write_global:
        shutil.copy2(bundle_dir / "validation_summary.csv", MODEL_DIR / "validation_summary.csv")
    print(f"  Saved validation summary")

    # Write bundle metadata
    meta = {
        "bundle_id": bundle_dir.name,
        "model_type": "pd",
        "dataset_path": str(portfolio_path.resolve()),
        "dataset_label": args.dataset_label or portfolio_path.name,
        "dataset_fingerprint": args.dataset_fingerprint or "",
        "training_timestamp": datetime.now().isoformat(),
        "row_count": len(train) + len(val) + len(test),
        "final_features": final_features,
        "xgboost_available": xgb_available,
        "xgboost_error": xgb_error,
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  Wrote bundle metadata: {bundle_dir / 'metadata.json'}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t_pipeline_start
    print(f"\n{'='*70}")
    print(f"PD MODEL PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
