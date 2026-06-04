"""
preprocess.py
-------------
Cleans data and tests for stationarity — a key concept in time series.

HOW IT WORKS:
STATIONARITY means the statistical properties (mean, variance) of the
series do NOT change over time. ARIMA requires stationary data.

Raw stock prices are NOT stationary — they trend upward over time.
Solution: take the FIRST DIFFERENCE (today's price minus yesterday's price).
This converts "price levels" into "daily changes", which ARE stationary.

The ADF Test (Augmented Dickey-Fuller) is the standard statistical test
to check stationarity. p-value < 0.05 means the series IS stationary.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def load_data(filepath="data/googl.csv"):
    """Load CSV and parse the Date column as the index."""
    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    df.columns = ["Close"]
    return df


def adf_test(series, name="Series"):
    """
    Runs the Augmented Dickey-Fuller test.

    Null hypothesis: The series HAS a unit root (is NOT stationary)
    If p-value < 0.05: Reject null → series IS stationary
    If p-value > 0.05: Fail to reject → series is NOT stationary (needs differencing)
    """
    result = adfuller(series.dropna())
    print(f"\nADF Test — {name}")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Conclusion    : {'STATIONARY (ready for ARIMA)' if result[1] < 0.05 else 'NOT STATIONARY (needs differencing)'}")
    return result[1] < 0.05


def preprocess(df):
    """
    Creates differenced series and plots both original and differenced data.
    Returns the differenced series for use in ARIMA.
    """
    # First difference: removes the trend
    df["Close_Diff"] = df["Close"].diff()

    # Plot side by side
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("GOOGL Stock Price — Stationarity Check", fontsize=14, fontweight='bold')

    axes[0].plot(df.index, df["Close"], color="#1a73e8", linewidth=1.2)
    axes[0].set_title("Original Closing Price (NOT stationary — has upward trend)")
    axes[0].set_ylabel("Price (USD)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[1].plot(df.index, df["Close_Diff"], color="#e8710a", linewidth=0.8, alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1].set_title("First Difference (stationary — fluctuates around 0)")
    axes[1].set_ylabel("Daily Change (USD)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig("data/stationarity_check.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved stationarity_check.png")

    return df


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows")

    # Test original series
    adf_test(df["Close"], "Original Close Price")

    # Test differenced series
    df = preprocess(df)
    adf_test(df["Close_Diff"], "First Differenced Series")
