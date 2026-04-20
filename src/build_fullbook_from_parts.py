"""
Build an engineered loan-level dataset from the raw partitioned parts dataset.

This script reads raw monthly Freddie-style parquet partitions under
`data/parts/`, converts them to the engineered loan-level shape used by the
rest of the project, and writes the output under `data/processed/fullbook/`.

Important notes:
- The raw parts dataset is not identical to the historical engineered training
  dataset. Some fields can be renamed directly, some can be derived, and some
  cannot be recovered from the raw parts source alone.
- When a feature cannot be derived from the raw source, the script writes the
  column with null values so the output can still align to the reference schema.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engine import (
    DEFAULT_ZERO_BALANCE_CODES,
    LGD_CAP,
    LGD_FLOOR,
    LOSS_COLS,
    PERFORMANCE_COLS,
    STATIC_COLS,
    create_derived_features,
    merge_macro_features,
)

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PARTS_ROOT = PROJECT_ROOT / "data" / "parts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "fullbook"
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "loan_level_combined.parquet"
DEFAULT_MACRO_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "macro" / "fred_macro_monthly.csv",
    PROJECT_ROOT / "data" / "macro" / "fred_macro_monthly.csv",
]

RAW_TO_ENGINEERED = {
    "Loan_Identifier": "loan_id",
    "Monthly_Reporting_Period": "monthly_reporting_period",
    "Channel": "channel",
    "Original_Interest_Rate": "original_interest_rate",
    "Original_UPB": "original_upb",
    "Original_Loan_Term": "original_loan_term",
    "Origination_Date": "origination_date",
    "First_Payment_Date": "first_payment_date",
    "Loan_Age": "loan_age",
    "Original_Loan_to_Value_Ratio_LTV": "original_ltv",
    "Original_Combined_Loan_to_Value_Ratio_CLTV": "original_cltv",
    "Number_of_Borrowers": "number_of_borrowers",
    "Debt_To_Income_DTI": "dti",
    "Borrower_Credit_Score_at_Origination": "borrower_credit_score",
    "Co_Borrower_Credit_Score_at_Origination": "coborrower_credit_score",
    "First_Time_Home_Buyer_Indicator": "first_time_home_buyer",
    "Loan_Purpose": "loan_purpose",
    "Property_Type": "property_type",
    "Number_of_Units": "number_of_units",
    "Occupancy_Status": "occupancy_status",
    "Property_State": "property_state",
    "MSA_or_MSDA": "msa",
    "Zip_Code_Short": "zip_code_short",
    "Mortgage_Insurance_Percentage": "mortgage_insurance_pct",
    "Amortization_Type": "amortization_type",
    "locf_Current_Actual_UPB": "current_actual_upb",
    "locf_Current_Loan_Delinquency_Status": "current_loan_delinquency_status",
    "locf_Modification_Flag": "modification_flag",
    "locf_Borrower_Assistance_Plan": "borrower_assistance_plan",
    "Total_Principal_Current": "total_principal_current",
    "Disposition_Date": "disposition_date",
    "Foreclosure_Date": "foreclosure_date",
    "Foreclosure_Costs": "foreclosure_costs",
    "Property_Preservation_and_Repair_Costs": "property_preservation_costs",
    "Asset_Recovery_Costs": "asset_recovery_costs",
    "Miscellaneous_Holding_Expenses_and_Credits": "misc_holding_expenses",
    "Associated_Taxes_for_Holding_Property": "holding_taxes",
    "Net_Sales_Proceeds": "net_sale_proceeds",
    "Credit_Enhancement_Proceeds": "credit_enhancement_proceeds",
    "Repurchase_Make_Whole_Proceeds": "repurchase_make_whole_proceeds",
    "Other_Foreclosure_Proceeds": "other_foreclosure_proceeds",
    "Zero_Balance_Code": "zero_balance_code",
}

DATE_COLUMNS = [
    "monthly_reporting_period",
    "origination_date",
    "first_payment_date",
    "disposition_date",
    "foreclosure_date",
]
NUMERIC_COLUMNS = [
    "original_interest_rate",
    "original_upb",
    "original_loan_term",
    "loan_age",
    "original_ltv",
    "original_cltv",
    "number_of_borrowers",
    "dti",
    "borrower_credit_score",
    "coborrower_credit_score",
    "number_of_units",
    "mortgage_insurance_pct",
    "current_actual_upb",
    "total_principal_current",
    "foreclosure_costs",
    "property_preservation_costs",
    "asset_recovery_costs",
    "misc_holding_expenses",
    "holding_taxes",
    "net_sale_proceeds",
    "credit_enhancement_proceeds",
    "repurchase_make_whole_proceeds",
    "other_foreclosure_proceeds",
]
RAW_REQUIRED_COLUMNS = sorted(RAW_TO_ENGINEERED.keys())
EMPTY_LGD_COLS = [
    "ead",
    "total_costs",
    "total_recovery",
    "total_loss",
    "lgd_raw",
    "lgd",
    "loan_age_at_default",
    "was_modified",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build engineered loan-level data from raw data/parts parquet partitions."
    )
    parser.add_argument(
        "--parts-root",
        default=str(DEFAULT_PARTS_ROOT),
        help="Root directory containing raw partitioned parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write engineered fullbook outputs.",
    )
    parser.add_argument(
        "--reference-path",
        default=str(DEFAULT_REFERENCE_PATH),
        help="Reference engineered parquet used to align output schema.",
    )
    parser.add_argument(
        "--macro-path",
        default=None,
        help="Optional macro CSV path. Defaults to the first existing project macro CSV.",
    )
    parser.add_argument(
        "--state-filter",
        default=None,
        help="Optional comma-separated state codes to limit processing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use across state partitions.",
    )
    return parser.parse_args()


def resolve_macro_path(explicit_macro_path: str | None) -> Path | None:
    if explicit_macro_path:
        path = Path(explicit_macro_path)
        if not path.exists():
            raise FileNotFoundError(f"Macro path not found: {path}")
        return path

    for candidate in DEFAULT_MACRO_PATHS:
        if candidate.exists():
            return candidate

    return None


def load_reference_columns(reference_path) -> list[str] | None:
    reference_path = Path(reference_path)
    if not reference_path.exists():
        return None

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading the reference parquet schema requires pyarrow.") from exc

    return pq.ParquetFile(reference_path).schema.names


def discover_state_dirs(parts_root: Path, state_filter: set[str] | None) -> list[Path]:
    state_dirs = sorted(
        path for path in parts_root.glob("Property_State=*") if path.is_dir()
    )
    if state_filter is None:
        return state_dirs

    return [
        path for path in state_dirs
        if path.name.split("=", 1)[-1] in state_filter
    ]


def normalize_raw_parts_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RAW_TO_ENGINEERED).copy()

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "current_loan_delinquency_status" in df.columns:
        df["current_loan_delinquency_status"] = pd.to_numeric(
            df["current_loan_delinquency_status"], errors="coerce"
        )

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def read_state_partition(state_dir: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading the raw parquet schema requires pyarrow.") from exc

    parquet_files = sorted(state_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {state_dir}")

    frames = []
    for parquet_file in parquet_files:
        available_columns = set(pq.ParquetFile(parquet_file).schema.names)
        selected_columns = [c for c in RAW_REQUIRED_COLUMNS if c in available_columns]
        frame = pd.read_parquet(parquet_file, columns=selected_columns)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return normalize_raw_parts_frame(combined)


def compute_default_and_lgd(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    dq_status = pd.to_numeric(
        df.get("current_loan_delinquency_status", pd.Series(index=df.index, dtype=float)),
        errors="coerce",
    )
    max_dq = dq_status.groupby(df["loan_id"]).max()
    ever_90dpd = (max_dq >= 3).astype(int)

    if "zero_balance_code" in df.columns:
        zb_default = (
            df.groupby("loan_id")["zero_balance_code"]
            .apply(lambda s: int(s.astype("string").isin(DEFAULT_ZERO_BALANCE_CODES).any()))
        )
    else:
        zb_default = pd.Series(0, index=max_dq.index, dtype=int)

    default_flag = ((ever_90dpd == 1) | (zb_default == 1)).astype(int)
    default_flag.name = "default_flag"
    max_dq.name = "max_delinquency_status"

    if "modification_flag" in df.columns:
        was_modified = (
            df.groupby("loan_id")["modification_flag"]
            .apply(lambda s: int((s.astype("string") == "Y").any()))
        )
    else:
        was_modified = pd.Series(np.nan, index=max_dq.index)

    if {"loan_age", "current_actual_upb"}.issubset(df.columns):
        delinquent_rows = df[dq_status >= 3].copy()
        if not delinquent_rows.empty:
            delinquent_rows = delinquent_rows.sort_values(
                ["loan_id", "monthly_reporting_period"]
            )
            first_default = delinquent_rows.groupby("loan_id").first()
            loan_age_at_default = first_default["loan_age"]
            ead = first_default["current_actual_upb"]
        else:
            loan_age_at_default = pd.Series(np.nan, index=max_dq.index)
            ead = pd.Series(np.nan, index=max_dq.index)
    else:
        loan_age_at_default = pd.Series(np.nan, index=max_dq.index)
        ead = pd.Series(np.nan, index=max_dq.index)

    lgd_summary = pd.DataFrame(index=max_dq.index, columns=EMPTY_LGD_COLS, dtype=float)
    lgd_summary["ead"] = ead.reindex(lgd_summary.index)
    lgd_summary["loan_age_at_default"] = loan_age_at_default.reindex(lgd_summary.index)
    lgd_summary["was_modified"] = was_modified.reindex(lgd_summary.index)
    lgd_summary["total_costs"] = np.nan
    lgd_summary["total_recovery"] = np.nan
    lgd_summary["total_loss"] = np.nan
    lgd_summary["lgd_raw"] = np.nan
    lgd_summary["lgd"] = np.nan

    available_loss_cols = [col for col in LOSS_COLS if col in df.columns]
    if available_loss_cols and "current_actual_upb" in df.columns:
        defaulted_ids = default_flag[default_flag == 1].index
        df_defaulted = df[df["loan_id"].isin(defaulted_ids)].copy()
        if not df_defaulted.empty:
            df_defaulted = df_defaulted.sort_values(["loan_id", "monthly_reporting_period"])
            df_defaulted = df_defaulted.set_index("loan_id")
            obs_counts = df_defaulted.groupby(level="loan_id").size()
            loans_with_two_rows = obs_counts[obs_counts >= 2].index
            df_defaulted = df_defaulted.loc[df_defaulted.index.isin(loans_with_two_rows)]

            if not df_defaulted.empty:
                ead_series = df_defaulted.groupby(level="loan_id")["current_actual_upb"].nth(-2)
                last_obs = df_defaulted.groupby(level="loan_id")[available_loss_cols].last()
                lgd_df = pd.DataFrame(index=ead_series.index)
                lgd_df["ead"] = pd.to_numeric(ead_series, errors="coerce")

                cost_cols = [c for c in [
                    "foreclosure_costs",
                    "property_preservation_costs",
                    "asset_recovery_costs",
                    "misc_holding_expenses",
                    "holding_taxes",
                ] if c in last_obs.columns]
                recovery_cols = [c for c in [
                    "net_sale_proceeds",
                    "credit_enhancement_proceeds",
                    "repurchase_make_whole_proceeds",
                    "other_foreclosure_proceeds",
                ] if c in last_obs.columns]

                for col in cost_cols:
                    lgd_df[col] = pd.to_numeric(last_obs[col], errors="coerce").fillna(0.0)
                lgd_df["total_costs"] = lgd_df[cost_cols].sum(axis=1) if cost_cols else np.nan

                for col in recovery_cols:
                    lgd_df[col] = pd.to_numeric(last_obs[col], errors="coerce")
                if recovery_cols:
                    has_valid_recovery = lgd_df[recovery_cols[0]].notna()
                    for col in recovery_cols:
                        lgd_df.loc[has_valid_recovery, col] = (
                            lgd_df.loc[has_valid_recovery, col].fillna(0.0)
                        )
                    lgd_df["total_recovery"] = lgd_df[recovery_cols].sum(axis=1)
                    lgd_df.loc[~has_valid_recovery, "total_recovery"] = np.nan
                else:
                    has_valid_recovery = pd.Series(False, index=lgd_df.index)
                    lgd_df["total_recovery"] = np.nan

                lgd_df["total_loss"] = lgd_df["ead"] - lgd_df["total_recovery"] + lgd_df["total_costs"]
                valid_mask = (lgd_df["ead"] > 0) & has_valid_recovery
                lgd_df["lgd_raw"] = np.nan
                lgd_df.loc[valid_mask, "lgd_raw"] = (
                    lgd_df.loc[valid_mask, "total_loss"] / lgd_df.loc[valid_mask, "ead"]
                )
                lgd_df["lgd"] = lgd_df["lgd_raw"].clip(lower=LGD_FLOOR, upper=LGD_CAP)
                lgd_summary.update(lgd_df[[c for c in EMPTY_LGD_COLS if c in lgd_df.columns]])

    return default_flag, max_dq, lgd_summary


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["data_split"] = "unknown"
    if "origination_year" in df.columns:
        df.loc[df["origination_year"] == 2005, "data_split"] = "train"
        df.loc[df["origination_year"] == 2006, "data_split"] = "validation"
        df.loc[df["origination_year"] == 2007, "data_split"] = "test"
    return df


def resolve_worker_count(workers: int | None, num_states: int) -> int:
    """Resolve a safe worker count for state-level parallelism."""
    cpu_count = os.cpu_count() or 1
    if workers is None:
        resolved = cpu_count
    else:
        resolved = max(1, int(workers))
    return max(1, min(resolved, cpu_count, num_states))


def align_to_reference_schema(df: pd.DataFrame, reference_columns: list[str] | None) -> pd.DataFrame:
    df = df.copy()
    if reference_columns is None:
        return df

    for column in reference_columns:
        if column not in df.columns:
            if column == "__index_level_0__":
                df[column] = np.arange(len(df))
            else:
                df[column] = np.nan

    extra_columns = [col for col in df.columns if col not in reference_columns]
    ordered_columns = [col for col in reference_columns if col in df.columns] + extra_columns
    return df[ordered_columns]


def engineer_state_partition(
    state_dir: Path,
    macro_df: pd.DataFrame | None,
    reference_columns: list[str] | None,
) -> pd.DataFrame:
    state_code = state_dir.name.split("=", 1)[-1]
    print(f"\n{'=' * 70}")
    print(f"Processing state {state_code}: {state_dir}")
    print(f"{'=' * 70}")
    t0 = time.time()

    df = read_state_partition(state_dir)
    print(f"  Loaded {len(df):,} monthly rows across {df['loan_id'].nunique():,} loans")

    df = df.sort_values(["loan_id", "monthly_reporting_period"])
    static_cols = [col for col in STATIC_COLS if col in df.columns]
    static = df.groupby("loan_id")[static_cols].first()
    default_flag, max_dq, lgd_summary = compute_default_and_lgd(df)

    loan_level = static.copy()
    loan_level["default_flag"] = default_flag
    loan_level["max_delinquency_status"] = max_dq
    loan_level = loan_level.join(lgd_summary, how="left")
    loan_level = create_derived_features(loan_level)

    if "origination_date" in loan_level.columns:
        orig_dates = pd.to_datetime(loan_level["origination_date"], errors="coerce")
        loan_level["origination_quarter"] = (
            orig_dates.dt.year.astype("Int64").astype("string")
            + "Q"
            + orig_dates.dt.quarter.astype("Int64").astype("string")
        )

    if macro_df is not None:
        loan_level = merge_macro_features(loan_level.reset_index(), macro_df).set_index("loan_id")
    else:
        print("  WARNING: Macro CSV not found. Macro-derived columns will be null.")

    loan_level = assign_splits(loan_level)
    loan_level = align_to_reference_schema(loan_level.reset_index(), reference_columns)
    print(f"  Output shape: {loan_level.shape}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    del df, static, lgd_summary
    gc.collect()

    return loan_level


def process_state_partition_task(task):
    """Worker task for one state partition."""
    state_dir = Path(task["state_dir"])
    macro_path = Path(task["macro_path"]) if task["macro_path"] else None
    reference_path = Path(task["reference_path"]) if task["reference_path"] else None
    state_output_dir = Path(task["state_output_dir"])

    macro_df = None
    if macro_path is not None and macro_path.exists():
        macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)

    reference_columns = load_reference_columns(reference_path) if reference_path and reference_path.exists() else None

    state_df = engineer_state_partition(state_dir, macro_df, reference_columns)
    state_code = state_dir.name.split("=", 1)[-1]
    state_path = state_output_dir / f"{state_code}_loan_level.parquet"
    state_df.to_parquet(state_path, index=False)

    return {
        "state_code": state_code,
        "state_path": str(state_path),
        "num_rows": len(state_df),
    }


def main():
    args = parse_args()
    parts_root = Path(args.parts_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_output_dir = output_dir / "loan_level_state"
    state_output_dir.mkdir(parents=True, exist_ok=True)

    state_filter = None
    if args.state_filter:
        state_filter = {part.strip().upper() for part in args.state_filter.split(",") if part.strip()}

    macro_path = resolve_macro_path(args.macro_path)
    if macro_path is not None:
        print(f"Loading macro data from {macro_path}")
        macro_df = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        print(f"  Macro shape: {macro_df.shape}")
    else:
        macro_df = None

    reference_columns = load_reference_columns(args.reference_path)
    if reference_columns is not None:
        print(f"Loaded reference schema from {args.reference_path} ({len(reference_columns)} columns)")
    else:
        print("Reference schema not found. Output will use discovered columns.")

    state_dirs = discover_state_dirs(parts_root, state_filter)
    if not state_dirs:
        raise FileNotFoundError(f"No state partitions found under {parts_root}")

    print(f"Found {len(state_dirs):,} state partitions")
    worker_count = resolve_worker_count(args.workers, len(state_dirs))
    print(f"Using {worker_count} worker process(es)")

    tasks = [
        {
            "state_dir": str(state_dir),
            "macro_path": str(macro_path) if macro_path is not None else None,
            "reference_path": str(args.reference_path) if args.reference_path else None,
            "state_output_dir": str(state_output_dir),
        }
        for state_dir in state_dirs
    ]

    if worker_count == 1:
        results = [process_state_partition_task(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            results = pool.map(process_state_partition_task, tasks, chunksize=1)

    results.sort(key=lambda item: item["state_code"])
    for result in results:
        print(
            f"  Saved {result['state_code']} loan-level parquet to {result['state_path']} "
            f"({result['num_rows']:,} rows)"
        )

    print(f"\n{'=' * 70}")
    print("Combining state-level loan data")
    print(f"{'=' * 70}")
    combined_parts = [
        pd.read_parquet(result["state_path"])
        for result in results
    ]
    combined = pd.concat(combined_parts, axis=0, ignore_index=True)
    combined = align_to_reference_schema(combined, reference_columns)

    combined_path = output_dir / "loan_level_combined.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"Saved combined fullbook loan-level dataset to {combined_path}")
    print(f"  Shape: {combined.shape}")
    if "default_flag" in combined.columns:
        print(
            f"  Defaults: {combined['default_flag'].fillna(0).sum():,.0f} "
            f"({combined['default_flag'].fillna(0).mean():.4f})"
        )


if __name__ == "__main__":
    main()
