"""
Optimize ensemble model weights using backtest performance

Tests different weight combinations and finds optimal weights
based on Sharpe ratio, CAGR, or other metrics.
"""

import argparse
import json
import os
import subprocess
import itertools
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')


def run_backtest_get_metrics(agg_csv: str, price_csv: str) -> Dict:
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
        
        metrics_path = "SP500_news/results/backtest_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return metrics
    except subprocess.CalledProcessError as e:
        print(f"Backtest error: {e.stderr}")
        return None
    
    return None


def create_custom_weights_config(weights: Dict[str, float], base_models: Dict) -> Dict:
    """Create model config with custom weights"""
    config = {}
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        # If all weights are zero, use equal weights
        total_weight = len(weights)
        weights = {k: 1.0 for k in weights.keys()}
    
    for model_name, weight in weights.items():
        if model_name in base_models:
            config[model_name] = base_models[model_name].copy()
            config[model_name]["weight"] = weight / total_weight  # Normalize
    
    return config


def run_pipeline_with_weights(weights: Dict[str, float], raw_csv: str, 
                               date_col: str, title_col: str, 
                               base_models: Dict, temp_config_file: str = "temp_weights.json"):
    """Run pipeline with custom weights"""
    # Create temporary config
    config = create_custom_weights_config(weights, base_models)
    
    # Save to temp file
    import json
    with open(temp_config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    # Run pipeline (we'll need to modify pipeline to accept config file)
    # For now, we'll use a workaround by modifying the script
    cmd = [
        "python", "sp500_ensemble_sentiment_pipeline.py",
        "--raw_csv", raw_csv,
        "--date_col", date_col,
        "--title_col", title_col,
        "--ensemble_method", "weighted_average"
    ]
    
    try:
        # We need to pass weights somehow - let's use environment variables or modify approach
        # Actually, better to modify the pipeline to accept weights as arguments
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Pipeline error: {e.stderr}")
        return False
    finally:
        if os.path.exists(temp_config_file):
            os.remove(temp_config_file)


def objective_function(weights_array: np.ndarray, model_names: List[str], 
                       raw_csv: str, price_csv: str, base_models: Dict,
                       metric: str = "sharpe") -> float:
    """Objective function for optimization (negative because we maximize)"""
    # Convert array to dict
    weights = {name: float(w) for name, w in zip(model_names, weights_array)}
    
    # Normalize weights
    total = sum(weights.values())
    if total == 0:
        return -999.0  # Penalty for invalid weights
    weights = {k: v/total for k, v in weights.items()}
    
    # Run pipeline with these weights
    # We'll need to modify the pipeline to accept weights
    # For now, let's use a simpler grid search approach
    return -999.0  # Placeholder


def grid_search_weights(raw_csv: str, price_csv: str, base_models: Dict,
                       metric: str = "sharpe", n_steps: int = 5):
    """Grid search for optimal weights"""
    print("=" * 60)
    print("GRID SEARCH WEIGHT OPTIMIZATION")
    print("=" * 60)
    
    model_names = list(base_models.keys())
    n_models = len(model_names)
    
    # Generate weight combinations
    # We'll test different weight distributions
    print(f"\nTesting {n_models} models with {n_steps} steps per dimension...")
    print(f"Total combinations: {n_steps ** n_models}")
    print("(This may take a while - consider reducing n_steps)")
    
    # Create weight grid
    weight_values = np.linspace(0.0, 1.0, n_steps)
    
    best_weights = None
    best_metric = -np.inf
    best_results = []
    
    # Test a subset of combinations (full grid is too large)
    # Instead, test common patterns
    test_patterns = [
        # Equal weights
        {name: 1.0/n_models for name in model_names},
        
        # Single model dominance
        *[{name: 1.0 if name == m else 0.0 for name in model_names} 
          for m in model_names],
        
        # Two-model combinations
        *[{name: 0.5 if name in [m1, m2] else 0.0 for name in model_names}
          for m1 in model_names for m2 in model_names if m1 < m2],
        
        # Graduated weights
        {name: (i+1)/sum(range(1, n_models+1)) for i, name in enumerate(model_names)},
        {name: (n_models-i)/sum(range(1, n_models+1)) for i, name in enumerate(model_names)},
    ]
    
    # Add some random combinations
    np.random.seed(42)
    for _ in range(10):
        w = np.random.dirichlet(np.ones(n_models))
        test_patterns.append({name: float(w[i]) for i, name in enumerate(model_names)})
    
    print(f"\nTesting {len(test_patterns)} weight combinations...")
    
    for idx, weights in enumerate(test_patterns):
        print(f"\n[{idx+1}/{len(test_patterns)}] Testing weights: {weights}")
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # We need to modify the pipeline to accept custom weights
        # For now, let's create a modified version that accepts weights
        try:
            # Import and modify the pipeline function
            from sp500_ensemble_sentiment_pipeline import run_ensemble_pipeline
            
            # Run with custom weights
            df, daily = run_ensemble_pipeline(
                raw_csv=raw_csv,
                date_col="Date",
                title_col="Title",
                model_configs=create_custom_weights_config(weights, base_models),
                ensemble_method="weighted_average",
                batch_size=32,
                max_length=128
            )
            
            # Run backtest
            agg_csv = "SP500_news/processed/sp500_headlines_daily_agg.csv"
            metrics = run_backtest_get_metrics(agg_csv, price_csv)
            
            if metrics:
                metric_value = metrics.get(metric, 0.0)
                print(f"  {metric}: {metric_value:.4f}")
                
                result = {
                    "weights": weights.copy(),
                    "metrics": metrics.copy(),
                    metric: metric_value
                }
                best_results.append(result)
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_weights = weights.copy()
                    print(f"  ✓ New best {metric}: {metric_value:.4f}")
            else:
                print(f"  ✗ Backtest failed")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return best_weights, best_metric, best_results


def main():
    parser = argparse.ArgumentParser(description="Optimize ensemble model weights")
    parser.add_argument("--raw_csv", type=str,
                       default="SP500_news/raw/sp500_headlines_2008_2024.csv")
    parser.add_argument("--price_csv", type=str,
                       default="SP500_news/raw/sp500_headlines_2008_2024.csv")
    parser.add_argument("--metric", type=str, default="sharpe",
                       choices=["sharpe", "sortino", "cagr", "calmar"],
                       help="Metric to optimize")
    parser.add_argument("--n_steps", type=int, default=5,
                       help="Number of steps for grid search (reduced for speed)")
    parser.add_argument("--output", type=str,
                       default="SP500_news/results/optimized_weights.json")
    
    args = parser.parse_args()
    
    # Import base models
    from sp500_ensemble_sentiment_pipeline import DEFAULT_MODELS
    
    # Check if files exist
    if not os.path.exists(args.raw_csv):
        print(f"Error: Raw CSV not found: {args.raw_csv}")
        return
    
    if not os.path.exists(args.price_csv):
        print(f"Error: Price CSV not found: {args.price_csv}")
        return
    
    # Run optimization
    best_weights, best_metric, all_results = grid_search_weights(
        args.raw_csv,
        args.price_csv,
        DEFAULT_MODELS,
        metric=args.metric,
        n_steps=args.n_steps
    )
    
    # Save results
    if best_weights:
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"\nBest {args.metric}: {best_metric:.4f}")
        print("\nOptimal weights:")
        for model, weight in best_weights.items():
            print(f"  {model}: {weight:.4f}")
        
        # Save to file
        output_data = {
            "metric": args.metric,
            "best_metric_value": float(best_metric),
            "optimal_weights": best_weights,
            "all_results": all_results[:10]  # Top 10 results
        }
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
        # Create comparison table
        if all_results:
            results_df = pd.DataFrame([
                {
                    **{f"{m}_weight": r["weights"].get(m, 0.0) for m in DEFAULT_MODELS.keys()},
                    **{k: v for k, v in r["metrics"].items() if k != "model_name"}
                }
                for r in sorted(all_results, key=lambda x: x[args.metric], reverse=True)[:10]
            ])
            
            csv_output = args.output.replace(".json", "_comparison.csv")
            results_df.to_csv(csv_output, index=False)
            print(f"Comparison table saved to: {csv_output}")
    else:
        print("\n✗ Optimization failed - no valid results")


if __name__ == "__main__":
    main()

