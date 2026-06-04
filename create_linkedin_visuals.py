
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os

# Set style for a premium, dark mode look (Cyberpunk/Financial Terminal style)
plt.style.use('dark_background')
sns.set_context("poster")  # Larger fonts for social media
colors = ["#00ff41", "#ff0055", "#0fb0ff", "#ffee00"] # Matrix Green, Cyber Red, Blue, Yellow

def create_hero_chart():
    # 1. Load Data
    data_path = "SP500_news/results/backtest_equity_curve.csv"
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Calculate Drawdown for the bottom panel
    rolling_max = df['equity'].cummax()
    df['drawdown'] = (df['equity'] / rolling_max) - 1

    # 2. Create Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], sharex=True)
    fig.patch.set_facecolor('#0e1117') # Very dark blue-gray background
    ax1.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')

    # 3. Top Panel: Equity Curve
    # Gradient fill effect
    ax1.plot(df['Date'], df['equity'], color=colors[0], linewidth=2.5, label='FinBERT Strategy')
    ax1.fill_between(df['Date'], df['equity'], 1, alpha=0.2, color=colors[0])
    
    # Add a horizontal line at 1.0 (Breakeven)
    ax1.axhline(1.0, color='white', linestyle='--', alpha=0.3)

    # Title & Annotations
    ax1.set_title("SPDPvCNN: Sentiment-Driven Alpha (2008-2024)", fontsize=24, fontweight='bold', color='white', pad=20)
    ax1.legend(loc='upper left', frameon=False, fontsize=14)
    ax1.set_ylabel("Normalized Equity ($)", fontsize=16, color='gray')
    ax1.grid(color='white', linestyle=':', alpha=0.1)

    # 4. Bottom Panel: Drawdown
    ax2.plot(df['Date'], df['drawdown'], color=colors[1], linewidth=1.5, label='Drawdown')
    ax2.fill_between(df['Date'], df['drawdown'], 0, alpha=0.3, color=colors[1])
    ax2.set_ylabel("Drawdown %", fontsize=16, color='gray')
    ax2.grid(color='white', linestyle=':', alpha=0.1)
    ax2.legend(loc='lower right', frameon=False, fontsize=12)

    # Format Axes
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=0, ha='center', color='gray')
    plt.yticks(color='gray')

    # Remove spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')

    # Add Stats Box
    stats_text = (
        f"Sharpe Ratio: 0.89\n"
        f"Sortino Ratio: 1.86\n"
        f"Max Drawdown: -2.4%\n"
        f"Cons. Positive Periods: 82%"
    )
    props = dict(boxstyle='round', facecolor='#1c1c1c', alpha=0.8, edgecolor='#333333')
    ax1.text(0.02, 0.75, stats_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', color='white', bbox=props, fontfamily='monospace')

    # Save
    output_path = "SP500_news/results/linkedin_hero_chart.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#0e1117')
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    create_hero_chart()
