# Build and Run Guide

This repository supports two local workflows:

1. Run the Streamlit dashboard from the checked-in summary outputs already in [`models/`](../models/).
2. Rebuild the full CECL pipeline from raw external data and regenerate model artifacts locally.

## 1. Base environment

The repo does not pin a Python version, so use a current Python 3 virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python packages required by the code:

- `streamlit`, `plotly` for the dashboard
- `pandas`, `numpy`, `pyarrow` for data processing and parquet I/O
- `scikit-learn`, `xgboost`, `joblib` for model training and artifact loading
- `fredapi`, `python-dotenv` for macro data retrieval via FRED and `.env` loading

## 2. External dependencies

A full rebuild requires data that is not committed to this repository:

- Fannie Mae Single-Family Loan Performance raw quarterly CSVs
  - Expected under `data/raw/<quarter>/*.csv`
  - Example quarter folder names: `2005Q1`, `2005Q2`, ...
- Federal Reserve 2025 stress scenario CSVs
  - Expected under `data/scenarios/`
  - Required filenames:
    - `baseline_2025.csv`
    - `severely_adverse_2025.csv`
- FRED API key for macroeconomic features
  - Put `FRED_API_KEY=...` in `.env` at the repo root
  - If `data/processed/macro/fred_macro_monthly.csv` already exists, the feature pipeline can reuse it instead of calling FRED

Operational requirements:

- Disk: plan for large local storage. The code comments describe raw quarterly files as multi-GB and the full raw dataset as tens of GB.
- Memory: the ingestion/feature pipeline is written for roughly 8-16 GB RAM, but full model training is still compute-heavy.

## 3. Run the dashboard only

This is the fastest path. The checked-in CSV outputs in [`models/`](../models/) are enough for the visualizations.

```bash
streamlit run dashboard/app.py
```

Notes:

- No `data/` directory is required for dashboard-only usage.
- The Loan Scorer page will use simplified fallback formulas unless locally generated `.pkl` model artifacts are present in [`models/`](../models/).

## 4. Rebuild the full pipeline

Create this directory layout first:

```text
data/
  raw/
    2005Q1/*.csv
    2005Q2/*.csv
    ...
  scenarios/
    baseline_2025.csv
    severely_adverse_2025.csv
```

Then run the pipeline from the repository root in this order:

```bash
python src/data_pipeline.py
python src/feature_engine.py
python src/run_pd_model.py
python src/run_lgd_model.py
python src/run_ecl.py
python src/run_stress_test.py
python src/run_monte_carlo.py
python src/generate_dashboard_data.py
streamlit run dashboard/app.py
```

What each step produces:

- `src/data_pipeline.py`
  - Reads `data/raw/...`
  - Writes quarter-level parquet files to `data/processed/quarterly/`
- `src/feature_engine.py`
  - Builds loan-level features
  - Writes `data/processed/loan_level_combined.parquet`
  - Caches FRED data to `data/processed/macro/fred_macro_monthly.csv`
- `src/run_pd_model.py`
  - Trains PD models
  - Writes PD artifacts to [`models/`](../models/)
- `src/run_lgd_model.py`
  - Trains LGD models
  - Writes LGD artifacts to [`models/`](../models/)
- `src/run_ecl.py`
  - Generates CECL outputs including `models/ecl_summary.csv`
- `src/run_stress_test.py`
  - Generates stress outputs including `models/stress_test_summary.csv`
- `src/run_monte_carlo.py`
  - Generates Monte Carlo outputs including `models/mc_risk_metrics.csv`
- `src/generate_dashboard_data.py`
  - Produces the small dashboard CSV summaries consumed by Streamlit

## 5. Common failure points

- `No parquet files found in data/processed/quarterly`
  - Run `python src/data_pipeline.py` first.
- `FRED_API_KEY not found`
  - Add it to `.env`, or provide a cached `data/processed/macro/fred_macro_monthly.csv`.
- Missing `baseline_2025.csv` or `severely_adverse_2025.csv`
  - Add both files under `data/scenarios/` before running stress testing.
- Dashboard starts but Loan Scorer uses fallback logic
  - This means `.pkl` model artifacts were not generated locally; run the PD and LGD training steps.
