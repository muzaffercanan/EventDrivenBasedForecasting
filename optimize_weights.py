"""
Optimize ensemble model weights using backtest performance

OPTIMIZED VERSION: Uses pre-computed model scores instead of re-running pipeline.
This reduces optimization time from hours to minutes.
"""

import argparse
import json
import os
import subprocess
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def compute_ensemble_from_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Compute ensemble scores from pre-computed model scores using custom weights.
    
    This is MUCH faster than re-running the entire pipeline!
    """
    df = df.copy()
    
    # Get model signed scores
    model_scores = {}
    for model_name in weights.keys():
        signed_col = f"{model_name}_signed"
        if signed_col in df.columns:
            model_scores[model_name] = df[signed_col].values
        else:
            print(f"Warning: {signed_col} not found, using zeros")
            model_scores[model_name] = np.zeros(len(df))
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        # Equal weights if all zero
        weights = {k: 1.0/len(weights) for k in weights.keys()}
        total_weight = 1.0
    
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Compute weighted average
    ensemble_signed = np.zeros(len(df))
    for model_name, weight in normalized_weights.items():
        ensemble_signed += weight * model_scores[model_name]
    
    # Compute labels and scores
    ensemble_labels = []
    ensemble_scores = []
    
    for signed_val in ensemble_signed:
        if signed_val > 0.1:
            ensemble_labels.append("POSITIVE")
            ensemble_scores.append(abs(signed_val))
        elif signed_val < -0.1:
            ensemble_labels.append("NEGATIVE")
            ensemble_scores.append(abs(signed_val))
        else:
            ensemble_labels.append("NEUTRAL")
            ensemble_scores.append(abs(signed_val))
    
    df['ensemble_label'] = ensemble_labels
    df['ensemble_score'] = ensemble_scores
    df['ensemble_signed'] = ensemble_signed
    
    return df


def aggregate_daily(df: pd.DataFrame, time_decay: float = 0.1) -> pd.DataFrame:
    """Aggregate headlines to daily sentiment scores"""
    df = df.copy()
    df['__date'] = pd.to_datetime(df['__date'])
    
    # Coarse label
    def coarse_label(x: str) -> str:
        x = (x or "").upper()
        if "NEG" in x:
            return "NEGATIVE"
        if "POS" in x:
            return "POSITIVE"
        return "NEUTRAL"
    
    df['ensemble_label_coarse'] = df['ensemble_label'].apply(coarse_label)
    
    # Group by date
    grp = df.groupby(df['__date'].dt.date)
    
    daily = pd.DataFrame({
        'date': list(grp.groups.keys()),
        'news_count': grp.size().values,
        'pos_count': grp.apply(lambda g: (g['ensemble_label_coarse'] == 'POSITIVE').sum()).values,
        'neg_count': grp.apply(lambda g: (g['ensemble_label_coarse'] == 'NEGATIVE').sum()).values,
        'neu_count': grp.apply(lambda g: (g['ensemble_label_coarse'] == 'NEUTRAL').sum()).values,
    })
    
    # Time-weighted signed mean
    signed_means = []
    signed_stds = []
    signed_mins = []
    signed_maxs = []
    
    for date, group in grp:
        signed_vals = group['ensemble_signed'].values
        if len(signed_vals) > 0:
            # Time decay weighting (newer = higher weight)
            if time_decay > 0:
                weights = np.exp(-time_decay * np.arange(len(signed_vals))[::-1])
                weights = weights / weights.sum()
                signed_means.append(np.average(signed_vals, weights=weights))
            else:
                signed_means.append(np.mean(signed_vals))
            
            signed_stds.append(np.std(signed_vals))
            signed_mins.append(np.min(signed_vals))
            signed_maxs.append(np.max(signed_vals))
        else:
            signed_means.append(0.0)
            signed_stds.append(0.0)
            signed_mins.append(0.0)
            signed_maxs.append(0.0)
    
    daily['signed_mean'] = signed_means
    daily['signed_std'] = signed_stds
    daily['signed_min'] = signed_mins
    daily['signed_max'] = signed_maxs
    
    # Add individual model means
    model_names = ['finbert', 'finbert_tone', 'distilbert_sentiment', 'roberta_twitter']
    for model_name in model_names:
        signed_col = f"{model_name}_signed"
        if signed_col in df.columns:
            model_means = []
            for date, group in grp:
                vals = group[signed_col].values
                if len(vals) > 0:
                    if time_decay > 0:
                        weights = np.exp(-time_decay * np.arange(len(vals))[::-1])
                        weights = weights / weights.sum()
                        model_means.append(np.average(vals, weights=weights))
                    else:
                        model_means.append(np.mean(vals))
                else:
                    model_means.append(0.0)
            daily[f"{model_name}_signed_mean"] = model_means
    
    daily = daily.sort_values('date').reset_index(drop=True)
    
    return daily


def run_backtest_get_metrics(agg_csv: str, price_csv: str, 
                              tpos: float = 0.10, tneg: float = 0.10, 
                              nmin: int = 3) -> Dict:
    """Run backtest and return metrics (using direct import)"""
    try:
        # Import backtest function directly
        from sp500_sentiment_backtest import run_backtest
        
        # Run backtest (backtest expects date_col_sig="date" internally)
        df_bt, metrics = run_backtest(
            agg_csv=agg_csv,
            price_csv=price_csv,
            date_col="Date",  # Price CSV uses "Date", but backtest handles "date" internally
            price_col="CP",
            tpos=tpos,
            tneg=tneg,
            nmin=nmin,
            ret_type="log"
        )
        
        return metrics
    except Exception as e:
        print(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return None


def optimize_weights_fast(headlines_csv: str, price_csv: str, 
                         base_models: Dict, metric: str = "sharpe",
                         tpos: float = 0.10, tneg: float = 0.10, nmin: int = 3):
    """
    Fast weight optimization using pre-computed scores.
    
    This is MUCH faster than re-running the pipeline for each weight combination!
    """
    print("=" * 60)
    print("FAST WEIGHT OPTIMIZATION (Using Pre-computed Scores)")
    print("=" * 60)
    
    # Load pre-computed scores
    print(f"\nLoading pre-computed scores from {headlines_csv}...")
    df_headlines = pd.read_csv(headlines_csv)
    df_headlines['__date'] = pd.to_datetime(df_headlines['__date'])
    print(f"  Loaded {len(df_headlines)} headlines")
    
    model_names = list(base_models.keys())
    n_models = len(model_names)
    
    # Test weight patterns
    test_patterns = [
        # Equal weights
        {name: 1.0/n_models for name in model_names},
        
        # Single model dominance
        *[{name: 1.0 if name == m else 0.0 for name in model_names} 
          for m in model_names],
        
        # Two-model combinations
        *[{name: 0.5 if name in [m1, m2] else 0.0 for name in model_names}
          for m1 in model_names for m2 in model_names if m1 < m2],
        
        # Graduated weights (increasing)
        {name: (i+1)/sum(range(1, n_models+1)) for i, name in enumerate(model_names)},
        
        # Graduated weights (decreasing)
        {name: (n_models-i)/sum(range(1, n_models+1)) for i, name in enumerate(model_names)},
        
        # Current default weights
        {name: base_models[name].get('weight', 1.0/n_models) for name in model_names},
    ]
    
    # Add random combinations
    np.random.seed(42)
    for _ in range(15):
        w = np.random.dirichlet(np.ones(n_models) * 2)  # More diverse
        test_patterns.append({name: float(w[i]) for i, name in enumerate(model_names)})
    
    print(f"\nTesting {len(test_patterns)} weight combinations...")
    print("(This should take only a few minutes instead of hours!)")
    
    best_weights = None
    best_metric = -np.inf
    best_results = []
    
    # Temporary daily aggregation file
    temp_agg_csv = "SP500_news/processed/temp_daily_agg_optimization.csv"
    
    for idx, weights in enumerate(test_patterns):
        print(f"\n[{idx+1}/{len(test_patterns)}] Testing weights: {weights}")
        
        # Normalize
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        try:
            # Compute ensemble from pre-computed scores (FAST!)
            df_ensemble = compute_ensemble_from_scores(df_headlines, weights)
            
            # Aggregate to daily
            daily_agg = aggregate_daily(df_ensemble)
            
            # Save temporary daily aggregation
            daily_agg.to_csv(temp_agg_csv, index=False)
            
            # Keep date column as 'date' (backtest expects lowercase)
            daily_agg.to_csv(temp_agg_csv, index=False)
            
            # Run backtest
            metrics = run_backtest_get_metrics(temp_agg_csv, price_csv, tpos, tneg, nmin)
            
            if metrics:
                metric_value = metrics.get(metric, 0.0)
                print(f"  {metric}: {metric_value:.4f} | Sharpe: {metrics.get('sharpe', 0):.4f} | CAGR: {metrics.get('cagr', 0)*100:.2f}%")
                
                result = {
                    "weights": weights.copy(),
                    "metrics": metrics.copy(),
                    metric: metric_value
                }
                best_results.append(result)
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_weights = weights.copy()
                    print(f"  [NEW BEST] {metric}: {metric_value:.4f}")
            else:
                print(f"  ✗ Backtest failed")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cleanup
    if os.path.exists(temp_agg_csv):
        os.remove(temp_agg_csv)
    
    return best_weights, best_metric, best_results


def main():
    parser = argparse.ArgumentParser(description="Optimize ensemble model weights (FAST VERSION)")
    parser.add_argument("--headlines_csv", type=str,
                       default="SP500_news/processed/headlines_with_ensemble.csv",
                       help="CSV with pre-computed model scores")
    parser.add_argument("--price_csv", type=str,
                       default="SP500_news/raw/sp500_headlines_2008_2024.csv",
                       help="Price data CSV")
    parser.add_argument("--metric", type=str, default="sharpe",
                       choices=["sharpe", "sortino", "cagr", "calmar"],
                       help="Metric to optimize")
    parser.add_argument("--tpos", type=float, default=0.10,
                       help="Positive threshold")
    parser.add_argument("--tneg", type=float, default=0.10,
                       help="Negative threshold")
    parser.add_argument("--nmin", type=int, default=3,
                       help="Minimum news count")
    parser.add_argument("--output", type=str,
                       default="SP500_news/results/optimized_weights.json")
    
    args = parser.parse_args()
    
    # Import base models
    from sp500_ensemble_sentiment_pipeline import DEFAULT_MODELS
    
    # Check if files exist
    if not os.path.exists(args.headlines_csv):
        print(f"Error: Headlines CSV not found: {args.headlines_csv}")
        print("Run sp500_ensemble_sentiment_pipeline.py first!")
        return
    
    if not os.path.exists(args.price_csv):
        print(f"Error: Price CSV not found: {args.price_csv}")
        return
    
    # Run optimization
    best_weights, best_metric, all_results = optimize_weights_fast(
        args.headlines_csv,
        args.price_csv,
        DEFAULT_MODELS,
        metric=args.metric,
        tpos=args.tpos,
        tneg=args.tneg,
        nmin=args.nmin
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
