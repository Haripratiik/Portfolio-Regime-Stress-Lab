# Portfolio Regime & Stress Propagation Lab

**Course:** Quant Mentorship Program — Final Project
**Author:** Haripratiie
**Parent Project:** Stock Portfolio Optimizer V2 — [GitHub Repository](https://github.com/Haripratiik/Stock-Portfolio-Optimizer)

---

## Overview

A time-series study of how stress builds and spreads across an equal-weight basket of six
mega-cap technology stocks: AAPL, MSFT, GOOGL, AMZN, META, NVDA.

The analysis covers:
- **Market Regime Detection** via K-Means clustering on six rolling portfolio state features
- **Custom Stress Propagation Score** combining realized vol, pairwise correlation, breadth, dispersion, and lead-lag concentration
- **ARIMA Volatility Forecasting** with walk-forward evaluation (RMSE 0.0269 vs 0.0271 naive baseline)
- **GBM Monte Carlo Risk Simulation** producing VaR and CVaR tail-risk estimates

### Key Results

| Metric | Calm Expansion | Stress Contagion |
|--------|---------------|-----------------|
| Avg Realized Vol (20d) | 7.65% | 45.90% |
| Avg Stress Score | -0.93 | +1.49 |
| ARIMA RMSE | 0.0269 | — |
| Naive Baseline RMSE | 0.0271 | — |

---

## Deliverables

| File | Description |
|------|-------------|
| `portfolio_regime_stress_lab.ipynb` | Full analysis notebook — standalone, runs without any external `.py` files |
| `portfolio_regime_stress_lab_report.pdf` | Written summary report |
| `portfolio_regime_stress_lab_slide.pdf` | One-slide executive summary |
| `data/aligned_prices.csv` | Pre-downloaded daily adjusted-close prices (2021-04-19 to 2026-04-18) |
| `data/portfolio_features.csv` | Computed rolling features (vol, corr, breadth, stress score) |
| `data/metrics.json` | Exported model metrics from the research run |
| `figures/` | All charts referenced in the report and slide (PNG) |
| `tables/` | Regime summary, forecast metrics, transition probabilities (CSV) |

---

## How to Run

### Requirements

```
pip install numpy pandas matplotlib seaborn statsmodels scikit-learn yfinance
```

### Steps

1. Clone or download this repository.
2. Open `portfolio_regime_stress_lab.ipynb` in Jupyter Lab or Jupyter Notebook.
3. Run all cells from top to bottom (**Kernel > Restart & Run All**).

### Notes

- **No internet required.** Price data is pre-downloaded in `data/aligned_prices.csv`. The notebook loads from that file automatically; a live Yahoo Finance fetch only runs as a fallback if the CSV is missing.
- **No external scripts needed.** All constants and helper functions are defined inline in the notebook.
- **Expected runtime:** approximately 2-3 minutes. The ARIMA walk-forward loop re-fits on an expanding window for each test-set step and is the main source of runtime.
- **Reproducible.** Random seeds are fixed (`random_state=42`, `np.random.default_rng(42)`).

---

## Repository Structure

```
portfolio-regime-stress-lab/
├── portfolio_regime_stress_lab.ipynb       # Main analysis notebook
├── portfolio_regime_stress_lab_report.pdf  # Written report
├── portfolio_regime_stress_lab_slide.pdf   # One-slide summary
├── data/
│   ├── aligned_prices.csv                  # Pre-downloaded price data (1,305 trading days)
│   ├── portfolio_features.csv              # Computed rolling features
│   └── metrics.json                        # Model metrics from the research run
├── figures/
│   ├── normalized_prices.png
│   ├── rolling_state.png
│   ├── stress_regimes.png
│   ├── lead_lag_heatmap.png
│   ├── transition_heatmap.png
│   └── forecast_actual.png
├── tables/
│   ├── regime_summary.csv
│   ├── forecast_metrics.csv
│   ├── transition_probabilities.csv
│   ├── dwell_stats.csv
│   ├── forward_stats_by_regime.csv
│   ├── lead_lag_top10.csv
│   └── stress_entry_summary.csv
└── supporting_material/
    ├── rubric_alignment.md
    ├── speaker_notes.md
    └── submission_checklist.md
```

---

## Citations

Full citations with journal references are in the **References** section at the bottom of the notebook.

| Source | Used For |
|--------|----------|
| Box & Jenkins (1970) | ARIMA model selection, AIC/BIC grid search, walk-forward methodology |
| Dickey & Fuller (1979) | ADF stationarity test on price levels, log returns, and realized vol |
| Ljung & Box (1978) | Autocorrelation test on portfolio returns and ARIMA residuals |
| MacQueen (1967) | K-Means clustering algorithm underlying the three-regime segmentation |
| Rockafellar & Uryasev (2000) | CVaR tail-risk metric used in GBM simulation |
| Engle (1982) | ARCH persistence — motivation for modeling realized vol, not raw returns |
| Samuelson (1965) | GBM theoretical basis and absence of return autocorrelation |
| yfinance (Arora et al.) | Data source — Yahoo Finance daily OHLCV |
| scikit-learn (Pedregosa et al., 2011) | KMeans, StandardScaler |
| statsmodels (Seabold & Perktold, 2010) | ARIMA, adfuller, acorr_ljungbox |

---

## Data Source

Daily adjusted-close OHLCV data for AAPL, MSFT, GOOGL, AMZN, META, NVDA sourced from
**Yahoo Finance** via the `yfinance` library. Study period: **2021-04-18 to 2026-04-18**
(approximately 1,305 trading days). Data is pre-downloaded and committed to `data/aligned_prices.csv`
— no API call is needed to reproduce the analysis.
