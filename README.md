# 📈 GOOGL Stock Price Predictor — Time Series Forecasting

A machine learning project that forecasts Google (GOOGL) stock prices using three time-series techniques: **Moving Average**, **ARIMA**, and **Exponential Smoothing**.

---

## Results

| Model | MAE ($/day) | RMSE |
|---|---|---|
| **ARIMA(1,1,1)** | **$6.05** ✅ | **$7.67** |
| Exponential Smoothing | $13.71 | $17.00 |

**ARIMA outperformed Exponential Smoothing** — it better captured short-term momentum in the stock price.

---

## Project Structure

```
stock_predictor/
├── data/
│   ├── googl.csv               # Historical price data
│   ├── stationarity_check.png  # ADF test visualization
│   ├── forecast_results.png    # Forecast vs actual
│   └── dashboard.png           # Full 4-panel summary
├── src/
│   ├── fetch_data.py    # Download stock data via yfinance
│   ├── preprocess.py    # Stationarity test + differencing
│   ├── model.py         # MA, ARIMA, Exponential Smoothing
│   └── visualize.py     # Dashboard generator
├── main.py              # Run full pipeline
└── requirements.txt
```

---

## How It Works

### 1. Data Collection
Uses `yfinance` to pull 5 years of daily closing prices from Yahoo Finance.

### 2. Stationarity Testing (ADF Test)
Stock prices are **not stationary** (they trend upward). ARIMA requires stationary data.
- Applied **first differencing**: `price[t] - price[t-1]`
- Confirmed stationarity with the **Augmented Dickey-Fuller test** (p-value < 0.05)

### 3. Models

**Moving Average (Baseline)**
Average of last N days. Simple but always lags behind real price.

**ARIMA(1,1,1)**
- `AR(1)`: uses 1 past value to predict next
- `I(1)`: 1 round of differencing applied
- `MA(1)`: corrects using 1 past forecast error
- Used **rolling forecast** — retrained daily for realism

**Exponential Smoothing (Holt's)**
Like moving average but gives exponentially more weight to recent data. Also models the price trend.

### 4. Evaluation
- **MAE** (Mean Absolute Error): average dollar error per day
- **RMSE** (Root Mean Squared Error): penalizes large errors more

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

---

## Tech Stack
`Python` · `Pandas` · `NumPy` · `Statsmodels` · `Scikit-learn` · `Matplotlib` · `yfinance`

---

## Key Concepts Learned
- Time series stationarity and the ADF test
- ARIMA parameter selection (p, d, q)
- Rolling forecast strategy for realistic backtesting
- Exponential smoothing with trend component
- MAE vs RMSE as evaluation metrics
