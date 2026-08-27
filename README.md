# Time Series Forecasting — Asset Price Prediction from Signals

A time-series analysis and forecasting project that predicts an asset's price from seven auxiliary "signals," exploring how strongly each signal relates to price and comparing models to find the best predictor.

## Problem Statement

Given a time-series dataset containing an asset's daily price alongside 7 signals that may act as leading or coincident indicators, the goals are to:

- Clean and preprocess the raw data
- Identify relationships between the signals and the price
- Quantify the strength of those relationships with statistics and plots
- Use the signals to model and predict the asset's price

## Dataset

- ~3,000 daily observations spanning 2005–2016
- Columns: `Date`, `Price`, `Signal1` … `Signal7`
- A small number of missing values in some signal columns, handled via row-wise removal during preprocessing

## Approach

1. **Cleaning & preprocessing** — parsed dates, dropped an unneeded index column, removed rows with missing values, and re-indexed the dataframe.
2. **Exploratory analysis** — plotted price and all signals over time, computed rolling mean/std, and ran an Augmented Dickey-Fuller test to check stationarity.
3. **Correlation analysis** — converted dates to ordinal numbers and built a correlation matrix/heatmap to quantify the linear relationship between each signal and price. Signals with |correlation| > 0.25 were treated as meaningfully related; some signals (e.g. those tracking `Date` closely) move with price, while others move inversely.
4. **Modeling**
   - **Theil-Sen Regression** — a robust, outlier-resistant regression using date + all 7 signals as features, trained on an 80/20 chronological train/test split.
   - **ARIMA with exogenous variables** — used the signals as exogenous regressors and ran a grid search over `(p, d, q)` orders to minimize test-set MSE.
5. **Evaluation** — compared both models using RMSE and R².

## Results

| Model | RMSE | R² |
|---|---|---|
| Theil-Sen Regression | ~6.02 | ~0.998 |
| ARIMA (grid-searched order, with exogenous signals) | ~6.14 | ~0.998 |

Both models explain the price series extremely well; Theil-Sen regression edges out ARIMA slightly on this dataset while being simpler and faster to fit.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (`TheilSenRegressor`, metrics)
- statsmodels (`ARIMA`, `adfuller`)
- Matplotlib, Seaborn

## Notebook

All analysis, plots, and modeling steps are in [`Time-Series-Forecasting.ipynb`](./Time-Series-Forecasting.ipynb).

## Author

Diya Gupta
