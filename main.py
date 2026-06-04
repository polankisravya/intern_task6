"""
main.py
-------
Entry point. Run this to execute the full pipeline:
  1. Download GOOGL data
  2. Preprocess + stationarity test
  3. Run all three models
  4. Generate dashboard
"""

import os
import sys

# Add src folder to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fetch_data import fetch_stock_data
from preprocess import load_data, adf_test, preprocess
from model import moving_average, arima_forecast, exponential_smoothing_forecast, evaluate, plot_results
from visualize import build_dashboard


def main():
    print("=" * 55)
    print("   GOOGL STOCK PRICE PREDICTOR — Full Pipeline")
    print("=" * 55)

    # Step 1: Download data
    print("\n[Step 1/4] Fetching Data...")
    fetch_stock_data(save_path="data/googl.csv")

    # Step 2: Preprocess
    print("\n[Step 2/4] Preprocessing + Stationarity Tests...")
    df = load_data("data/googl.csv")
    adf_test(df["Close"], "Original Prices")
    df = preprocess(df)
    adf_test(df["Close_Diff"], "Differenced Series")

    # Step 3 + 4: Models + Dashboard
    print("\n[Step 3/4] Training Models...")
    print("\n[Step 4/4] Building Dashboard...")
    arima_mae, es_mae, arima_rmse, es_rmse = build_dashboard()

    # Final summary
    print("\n" + "=" * 55)
    print("   RESULTS SUMMARY")
    print("=" * 55)
    print(f"   ARIMA(1,1,1)     MAE=${arima_mae:.2f}  RMSE=${arima_rmse:.2f}")
    print(f"   Exp. Smoothing   MAE=${es_mae:.2f}  RMSE=${es_rmse:.2f}")
    winner = "ARIMA" if arima_mae < es_mae else "Exponential Smoothing"
    print(f"\n   Best model: {winner}")
    print("\n   Output files:")
    print("   - data/googl.csv")
    print("   - data/stationarity_check.png")
    print("   - data/forecast_results.png")
    print("   - data/dashboard.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
