"""
model.py
--------
Three forecasting models: Moving Average, ARIMA, and Exponential Smoothing.

HOW EACH MODEL WORKS:
---------------------

1. MOVING AVERAGE (MA)
   - Simplest baseline: average of the last N days
   - E.g., 30-day MA = average of last 30 closing prices
   - Smooths out short-term noise, shows the trend
   - Weakness: always "lags" behind real price

2. ARIMA (AutoRegressive Integrated Moving Average)
   - The gold standard for time series forecasting
   - Three components:
     AR(p): uses p past values to predict next value (like regression on own history)
     I(d) : differencing order — how many times we difference to get stationarity
     MA(q): uses q past forecast errors to correct predictions
   - ARIMA(1,1,1) means: use 1 lag, 1 differencing, 1 error term
   - We use "rolling forecast" — retrain on each new day for realism

3. EXPONENTIAL SMOOTHING (Holt-Winters)
   - Like moving average but gives MORE weight to recent data
   - Alpha (smoothing factor): close to 1 = very reactive, close to 0 = very smooth
   - Holt's method also models the TREND (direction of movement)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def moving_average(df, windows=[20, 50]):
    """
    Compute rolling moving averages.

    Parameters:
        windows: list of window sizes (e.g., 20-day, 50-day)
    """
    for w in windows:
        df[f"MA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def arima_forecast(train, test, order=(1, 1, 1)):
    """
    Rolling ARIMA forecast.

    Instead of training once and predicting all 60 days,
    we do a ROLLING FORECAST — we retrain after each prediction.
    This is more realistic: in real life, you'd retrain daily with new data.

    Parameters:
        train : training data (80% of history)
        test  : test data (last 60 days — what we try to predict)
        order : (p, d, q) for ARIMA
    """
    print(f"\nRunning ARIMA{order} rolling forecast on {len(test)} days...")
    history = list(train["Close"])
    predictions = []

    for i in range(len(test)):
        # Fit ARIMA on all available history
        model = ARIMA(history, order=order)
        result = model.fit()

        # Forecast 1 step ahead
        yhat = result.forecast(steps=1)[0]
        predictions.append(yhat)

        # Add actual value to history (rolling window)
        history.append(test["Close"].iloc[i])

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test)} days")

    return predictions


def exponential_smoothing_forecast(train, test):
    """
    Holt's Double Exponential Smoothing (handles trend).
    Fits on training data, forecasts the full test period at once.
    """
    print(f"\nRunning Exponential Smoothing forecast...")
    model = ExponentialSmoothing(
        train["Close"],
        trend="add",        # additive trend: price goes up/down linearly
        seasonal=None,      # no seasonality in daily stock data
        initialization_method="estimated"
    )
    result = model.fit()
    predictions = result.forecast(steps=len(test))
    return predictions.values


def evaluate(actual, predicted, model_name):
    """
    Compute MAE and RMSE to measure forecast accuracy.

    MAE  (Mean Absolute Error)      : average dollar error per day
    RMSE (Root Mean Squared Error)  : penalizes large errors more heavily
    Lower is better for both.
    """
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"\n{model_name} Performance:")
    print(f"  MAE  : ${mae:.2f}  (on average, off by ${mae:.2f} per day)")
    print(f"  RMSE : ${rmse:.2f}")
    return mae, rmse


def plot_results(df, test, arima_preds, es_preds):
    """
    Plot everything together: historical prices, moving averages,
    and both model forecasts vs actual test prices.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("GOOGL Stock Price Prediction", fontsize=15, fontweight='bold')

    # --- Top chart: Full history + Moving Averages ---
    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], color="#1a1a2e", linewidth=1, label="Actual Price", alpha=0.8)
    ax1.plot(df.index, df["MA_20"],  color="#e8710a", linewidth=1.5, label="20-Day MA", alpha=0.9)
    ax1.plot(df.index, df["MA_50"],  color="#1a73e8", linewidth=1.5, label="50-Day MA", alpha=0.9)
    ax1.set_title("Historical Price with Moving Averages")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Bottom chart: Zoom in on test period ---
    ax2 = axes[1]
    ax2.plot(test.index, test["Close"],   color="#1a1a2e", linewidth=2,   label="Actual Price",          marker='o', markersize=3)
    ax2.plot(test.index, arima_preds,     color="#e8710a", linewidth=1.8, label="ARIMA(1,1,1) Forecast", linestyle='--')
    ax2.plot(test.index, es_preds,        color="#0f9d58", linewidth=1.8, label="Exp. Smoothing Forecast",linestyle='-.')
    ax2.set_title("Forecast vs Actual — Last 60 Days (Test Period)")
    ax2.set_ylabel("Price (USD)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/forecast_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved forecast_results.png")


if __name__ == "__main__":
    from preprocess import load_data

    df = load_data()

    # Split: 80% train, last 60 days test
    test_size = 60
    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]
    print(f"Train: {len(train)} days | Test: {len(test)} days")

    # Moving averages on full dataset
    df = moving_average(df)

    # ARIMA forecast
    arima_preds = arima_forecast(train, test)
    evaluate(test["Close"], arima_preds, "ARIMA(1,1,1)")

    # Exponential Smoothing forecast
    es_preds = exponential_smoothing_forecast(train, test)
    evaluate(test["Close"], es_preds, "Exponential Smoothing")

    # Plot everything
    plot_results(df, test, arima_preds, es_preds)
