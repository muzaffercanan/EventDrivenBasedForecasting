"""
Compare individual models vs ensemble performance

This script runs backtests for each individual model and the ensemble
to compare their performance.
"""

import argparse
import subprocess
import os
import json
import pandas as pd
from pathlib import Path


def run_backtest(agg_csv: str, price_csv: str, model_name: str = "ensemble"):
    """Run backtest and return metrics"""
    cmd = [
        "python", "sp500_sentiment_backtest.py",
        "--agg_csv", agg_csv,
        "--price_csv", price_csv,
        "--date_col", "Date",
        "--price_col", "CP",
        "--tpos", "0.10",
        "--tneg", "0.10",
        "--nmin", "5",
        "--ret", "log"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Load metrics
        metrics_path = "SP500_news/results/backtest_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metrics["model_name"] = model_name
            return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest for {model_name}: {e}")
        return None
    
    return None


def compare_models():
    """Compare all models"""
    parser = argparse.ArgumentParser(description="Compare sentiment models")
    parser.add_argument("--raw_csv", type=str, 
                       default="SP500_news/raw/sp500_headlines_2008_2024.csv")
    parser.add_argument("--price_csv", type=str,
                       default="SP500_news/raw/sp500_headlines_2008_2024.csv")
    parser.add_argument("--date_col", type=str, default="Date")
    parser.add_argument("--title_col", type=str, default="Title")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    # First, run ensemble pipeline
    print("\n1. Running ensemble pipeline...")
    ensemble_cmd = [
        "python", "sp500_ensemble_sentiment_pipeline.py",
        "--raw_csv", args.raw_csv,
        "--date_col", args.date_col,
        "--title_col", args.title_col,
        "--ensemble_method", "weighted_average"
    ]
    
    try:
        subprocess.run(ensemble_cmd, check=True)
        print("✓ Ensemble pipeline completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Ensemble pipeline failed: {e}")
        return
    
    # Load ensemble daily aggregation
    ensemble_agg = "SP500_news/processed/sp500_headlines_daily_agg_ensemble.csv"
    
    if not os.path.exists(ensemble_agg):
        print(f"✗ Ensemble aggregation not found: {ensemble_agg}")
        return
    
    # Run backtests for each model
    models_to_test = ["ensemble"]
    
    # Check which individual models are available
    df_ensemble = pd.read_csv(ensemble_agg)
    available_models = [col.replace("_signed_mean", "") for col in df_ensemble.columns 
                       if col.endswith("_signed_mean")]
    
    print(f"\n2. Found individual models: {available_models}")
    
    all_metrics = []
    
    # Test ensemble
    print(f"\n3. Testing ensemble...")
    metrics = run_backtest(ensemble_agg, args.price_csv, "ensemble")
    if metrics:
        all_metrics.append(metrics)
        print(f"   Sharpe: {metrics['sharpe']:.3f}, CAGR: {metrics['cagr']:.3f}")
    
    # Test individual models (if we create separate aggregations)
    # For now, we'll just compare using the ensemble file
    # In a full implementation, you'd create separate daily aggregations per model
    
    # Save comparison results
    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        comparison_df = comparison_df.sort_values("sharpe", ascending=False)
        
        output_path = "SP500_news/results/model_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
        print(f"\nSaved to: {output_path}")
        
        # Save summary
        summary_path = "SP500_news/results/model_comparison_summary.txt"
        with open(summary_path, "w") as f:
            f.write("Model Comparison Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(comparison_df.to_string(index=False))
        
        print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    compare_models()

