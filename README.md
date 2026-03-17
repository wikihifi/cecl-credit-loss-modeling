# CECL Credit Risk Modeling Pipeline

An end-to-end credit risk modeling framework implementing CECL (Current Expected Credit Loss) lifetime loss estimation, Federal Reserve stress testing, and Monte Carlo simulation on 3.8 million Fannie Mae mortgage loans ($694B portfolio).

Built to demonstrate proficiency in quantitative credit risk analysis, model development, validation, and governance for financial institutions.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cecl-credit-loss-modeling.streamlit.app)

> **[Live Dashboard](https://cecl-credit-loss-modeling.streamlit.app)** | **[Framework Overview (Video)](https://youtu.be/qORtDigZkz0)** | **[Build & Run Guide](docs/build_and_run.md)** | **[Model Documentation](docs/model_documentation.md)**

---

## Business Context

Banks must estimate **lifetime expected losses** from day one of every loan under CECL (ASC 326). This framework answers three questions:

1. **How much should we reserve?** Scenario-weighted ECL = $158B (22.8% of portfolio)
2. **Can we survive a recession?** Severely adverse losses = $312B over 13 quarters
3. **What is our tail risk?** VaR 99.9% = $99B from 10,000 Monte Carlo simulations

## Architecture

![Architecture Diagram](docs/architecture.svg)

## Key Results

| Metric | Value |
|---|---|
| Portfolio | 3,804,801 loans, $693.8B outstanding |
| Default Rate | 14.4% (2005-2007 crisis-era vintages) |
| PD Model AUC | 0.6777 (Logistic Regression) / 0.7529 (XGBoost) |
| LGD Model R-squared | 0.1508 (OLS) / 0.2465 (XGBoost) |
| Baseline ECL | $115,562M (16.66%) |
| Severely Adverse ECL | $221,865M (31.98%) |
| Weighted ECL (60/40) | $158,083M (22.78%) |
| Stress Test 13Q Loss | $311,945M (44.96%) at peak stress |
| Monte Carlo VaR 99.9% | $99,208M (14.30%) |
| Monte Carlo ES 99% | $89,071M (12.84%) |

## Architecture Details

```
Expected Credit Loss = PD x LGD x EAD

PD (Probability of Default)
  Primary: Logistic Regression with WoE/IV transformation (10 features)
  Challenger: XGBoost (300 trees, depth 4)
  Validation: AUC, Gini, KS, calibration by decile, PSI

LGD (Loss Given Default)  
  Primary: OLS Linear Regression (13 features)
  Challenger: XGBoost Regressor
  Training: 278,220 defaulted loans with observed losses

EAD (Exposure at Default)
  Amortization schedule: Balance(t) = P x [(1+r)^n - (1+r)^t] / [(1+r)^n - 1]

ECL (Lifetime Expected Loss)
  PD term structure: 12-month PD extended via seasoning curve + survival function
  Scenario weighting: 60% baseline + 40% severely adverse
  Discounting: at loan contractual rate per CECL guidance

Stress Testing
  Fed 2025 scenarios (13 quarters, baseline + severely adverse)
  Macro-sensitivity overlay: PD +25%/+1% unemployment, LGD +1.5%/-1% HPI

Monte Carlo Simulation
  10,000 correlated scenarios via Cholesky decomposition
  Risk metrics: VaR 99%, VaR 99.9%, Expected Shortfall 99%
```

## Tech Stack

**Data:** Fannie Mae Single-Family Loan Performance Data (265M loan-month records), FRED API (macroeconomic series), Federal Reserve 2025 Stress Test Scenarios

**Modeling:** Python, scikit-learn, XGBoost, NumPy, pandas

**Dashboard:** Streamlit, Plotly

**Infrastructure:** Parquet columnar storage, chunked processing (500K rows), memory-optimized pipeline for 8-16GB RAM constraint

## Dashboard

**[Live Demo: cecl-credit-loss-modeling.streamlit.app](https://cecl-credit-loss-modeling.streamlit.app)**

| Page | What It Shows |
|---|---|
| **Portfolio Overview** | KPI cards, FICO/LTV/vintage composition, ECL summary, production deployment context |
| **PD Model** | Primary vs challenger AUC/KS comparison, calibration by decile, Information Value ranking, logistic regression coefficients, methodology rationale |
| **LGD Model** | OLS vs XGBoost R-squared comparison, coefficient interpretation, segment analysis by FICO and LTV, LGD calculation methodology |
| **Stress Testing** | Fed 2025 quarterly loss path with cumulative view, stress multiplier visualization, custom scenario builder with interactive sliders for unemployment/HPI/GDP |
| **Monte Carlo** | 10,000-simulation loss distribution histogram with VaR/ES markers, sensitivity tornado chart identifying dominant risk drivers, correlated scenario scatter plot |
| **Loan Scorer** | Production scoring simulation: input borrower FICO, LTV, DTI, loan amount, property value, and macro conditions to get real-time PD, LGD, lifetime ECL, and internal risk rating (1-10 scale) |

## Methodology Highlights

**WoE/IV Feature Selection:** 38 candidate features evaluated. 10 selected based on Information Value thresholds (>0.02) with redundancy removal. FICO (IV=0.60) and FICO x Unemployment interaction (IV=0.58) are dominant predictors.

**Time-Based Validation:** Train on 2005 vintages, validate on 2006 (out-of-time), test on 2007 (crisis period). Never random splits for credit risk.

**Calibration Over Discrimination:** Logistic regression trained without class reweighting to preserve probability calibration. Predicted PDs directly determine dollar reserves; miscalibration creates regulatory risk.

**Stress Testing Design:** Macro-sensitivity overlay approach chosen because origination-time features have inverted economic interpretation versus point-in-time features. Multipliers calibrated to 2008 crisis empirical relationships.

**Model Limitations Documented:** AUC below 0.75 target, PSI=0.29 indicating population shift, no geographic concentration risk, static portfolio assumption, linear stress multipliers. Full limitations analysis in model documentation.

## Deployment

The dashboard is deployed on **Streamlit Cloud**. Due to GitHub file size limits (100MB), trained model artifacts (`.pkl` files, `.parquet` loan-level data) are excluded from the repository. The dashboard uses pre-computed summary CSVs for all visualizations. The Loan Scorer uses simplified scoring formulas in the deployed version; locally with full model artifacts, it scores through the actual trained logistic regression and OLS models.

## Future Enhancements

| Enhancement | Description |
|---|---|
| Geographic risk | State-level unemployment and HPI to capture concentration risk |
| Dynamic LGD | Time-varying LGD based on loan seasoning and current LTV |
| Non-linear stress | Piece-wise polynomial multipliers for extreme scenarios |
| Retraining pipeline | MLflow tracking with PSI > 0.25 and calibration drift triggers |
| API endpoint | FastAPI wrapper for real-time scoring in production systems |

## Project Structure

```
cecl-credit-loss-modeling/
  src/
    data_pipeline.py            Raw Fannie Mae data ingestion (Phase 1)
    feature_engine.py           Loan-level feature engineering (Phase 2)
    pd_model.py                 WoE/IV, logistic regression, XGBoost (Phase 3)
    run_pd_model.py             Full PD training pipeline
    lgd_model.py                OLS and XGBoost LGD models (Phase 4)
    run_lgd_model.py            Full LGD training pipeline
    ecl_engine.py               EAD amortization + lifetime ECL (Phases 5-6)
    run_ecl.py                  Portfolio ECL calculation
    stress_testing.py           Fed scenario stress testing (Phase 7)
    run_stress_test.py          Stress test pipeline
    monte_carlo.py              Correlated MC simulation (Phase 8)
    run_monte_carlo.py          Monte Carlo pipeline
    generate_dashboard_data.py  Summary CSVs for deployment
  models/                       Model artifacts and result CSVs
  dashboard/
    app.py                      Streamlit dashboard entry point
    utils.py                    Shared chart styling
    views/                      6 dashboard pages
  docs/
    model_documentation.md      Model governance document
    architecture.png            Architecture diagram
  data/                         Raw and processed data (not in repo)
```

## Model Documentation

The complete model governance document is in [`docs/model_documentation.md`](docs/model_documentation.md), covering: executive summary, data description, PD/LGD methodology, validation results, stress testing, Monte Carlo, limitations, and monitoring plan.

## Author

**Saurabh Chavan**

Built as a comprehensive demonstration of CECL credit risk modeling capabilities, including PD/LGD/EAD development, lifetime loss estimation, regulatory stress testing, Monte Carlo simulation, model documentation, and interactive dashboard presentation.
