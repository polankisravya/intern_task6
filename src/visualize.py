"""
visualize.py
------------
Generates the final summary visualization — a 4-panel dashboard.

HOW IT WORKS:
- Combines all analysis into one clean figure
- Panel 1: Full price history
- Panel 2: Stationarity (original vs differenced)
- Panel 3: Model comparison (ARIMA vs Exp Smoothing)
- Panel 4: Model error bar chart (MAE comparison)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from preprocess import load_data, preprocess
from model import moving_average, arima_forecast, exponential_smoothing_forecast, evaluate


def build_dashboard():
    print("Building full dashboard...")

    # Load and prep data
    df = load_data()
    df = preprocess(df)
    df = moving_average(df, windows=[20, 50])

    # Train/test split
    test_size = 60
    train = df.iloc[:-test_size]
    test  = df.iloc[-test_size:]

    # Run models
    arima_preds = arima_forecast(train, test)
    es_preds    = exponential_smoothing_forecast(train, test)

    # Evaluate
    arima_mae,  arima_rmse  = evaluate(test["Close"], arima_preds, "ARIMA")
    es_mae,     es_rmse     = evaluate(test["Close"], es_preds,    "Exp Smoothing")

    # Build 4-panel figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("GOOGL Stock Price Predictor — Full Analysis Dashboard",
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Panel 1: Full price history + MAs
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df["Close"], color="#1a1a2e", linewidth=0.8, alpha=0.7, label="Close")
    ax1.plot(df.index, df["MA_20"], color="#e8710a", linewidth=1.5, label="20-Day MA")
    ax1.plot(df.index, df["MA_50"], color="#1a73e8", linewidth=1.5, label="50-Day MA")
    ax1.set_title("1. Price History + Moving Averages", fontweight='bold')
    ax1.set_ylabel("Price (USD)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Stationarity check
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df.index, df["Close_Diff"], color="#5f6368", linewidth=0.7, alpha=0.8)
    ax2.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.7)
    ax2.set_title("2. First Difference (Stationary Series)", fontweight='bold')
    ax2.set_ylabel("Daily Change (USD)")
    ax2.grid(True, alpha=0.2)

    # Panel 3: Forecast comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(test.index, test["Close"],  color="#1a1a2e", linewidth=2,   label="Actual",          marker='o', markersize=3)
    ax3.plot(test.index, arima_preds,    color="#e8710a", linewidth=1.8, label="ARIMA(1,1,1)",    linestyle='--')
    ax3.plot(test.index, es_preds,       color="#0f9d58", linewidth=1.8, label="Exp. Smoothing",  linestyle='-.')
    ax3.set_title("3. Forecast vs Actual (Last 60 Days)", fontweight='bold')
    ax3.set_ylabel("Price (USD)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # Panel 4: Model comparison bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    models = ["ARIMA(1,1,1)", "Exp. Smoothing"]
    mae_vals  = [arima_mae,  es_mae]
    rmse_vals = [arima_rmse, es_rmse]
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax4.bar(x - width/2, mae_vals,  width, label="MAE",  color="#1a73e8", alpha=0.85)
    bars2 = ax4.bar(x + width/2, rmse_vals, width, label="RMSE", color="#e8710a", alpha=0.85)
    ax4.set_title("4. Model Error Comparison (lower = better)", fontweight='bold')
    ax4.set_ylabel("Error (USD)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.grid(True, alpha=0.2, axis='y')
    # Annotate bars with values
    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"${bar.get_height():.1f}", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"${bar.get_height():.1f}", ha='center', va='bottom', fontsize=9)

    plt.savefig("data/dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nDashboard saved to data/dashboard.png")
    return arima_mae, es_mae, arima_rmse, es_rmse


if __name__ == "__main__":
    build_dashboard()
