"""
fetch_data.py
-------------
Downloads historical GOOGL stock data using yfinance and saves it as a CSV.
If yfinance is unavailable (network restrictions), generates realistic synthetic data.

HOW IT WORKS:
- yfinance is a Python wrapper around Yahoo Finance's API
- We pull 5 years of daily OHLCV data (Open, High, Low, Close, Volume)
- We only keep the 'Close' price — that's what we predict
- Synthetic fallback uses geometric Brownian motion (same model used in options pricing)
"""

import pandas as pd
import numpy as np
import os


def generate_synthetic_googl(n_days=1260, save_path="data/googl.csv"):
    """
    Generates realistic GOOGL-like price data using Geometric Brownian Motion (GBM).

    GBM is the mathematical model behind Black-Scholes options pricing.
    It models stock prices as: dS = mu*S*dt + sigma*S*dW
    Where:
      mu    = drift (average daily return ~0.05% for GOOGL)
      sigma = volatility (daily std ~1.5% for GOOGL)
      dW    = random Wiener process (random shocks)
    """
    print("Generating synthetic GOOGL data (Geometric Brownian Motion)...")

    np.random.seed(42)  # for reproducibility
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)  # business days only

    # GOOGL parameters (approximate real values)
    S0    = 100.0   # starting price (normalized; real GOOGL ~$170 in 2024)
    mu    = 0.0005  # daily drift (~12% annual return)
    sigma = 0.015   # daily volatility (~24% annual)

    # Simulate price path
    returns = np.random.normal(mu, sigma, n_days)
    prices  = S0 * np.exp(np.cumsum(returns))

    # Scale to realistic GOOGL price range (~$90 to $200)
    prices = prices / prices[0] * 135.0

    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)

    print(f"Generated {len(df)} trading days of synthetic GOOGL data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    return df


def fetch_stock_data(ticker="GOOGL", period="5y", save_path="data/googl.csv"):
    """
    Attempts to download real data via yfinance; falls back to synthetic data.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if len(df) < 100:
            raise ValueError("Not enough data downloaded")
        df = df[["Close"]].copy()
        df.columns = ["Close"]
        df.dropna(inplace=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Downloaded {len(df)} rows from Yahoo Finance")
        return df
    except Exception as e:
        print(f"yfinance unavailable ({e}), using synthetic data...")
        return generate_synthetic_googl(save_path=save_path)


if __name__ == "__main__":
    df = fetch_stock_data()
    print("\nFirst 5 rows:")
    print(df.head())
