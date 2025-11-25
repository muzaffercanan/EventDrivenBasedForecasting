"""
SP500 Ensemble Sentiment Pipeline

Multiple sentiment models combined via ensemble methods:
- ProsusAI/finbert (financial domain)
- yiyanghkust/finbert-tone (financial tone analysis)
- distilbert/distilbert-base-uncased-finetuned-sst-2-english (general sentiment)
- cardiffnlp/twitter-roberta-base-sentiment-latest (social media sentiment)

Ensemble strategies:
- voting: Majority vote on labels
- weighted_average: Weighted average of signed scores
- stacking: Meta-learner approach (future enhancement)

Usage:
  python sp500_ensemble_sentiment_pipeline.py \
      --raw_csv SP500_news/raw/sp500_headlines_2008_2024.csv \
      --date_col Date --title_col Title \
      --ensemble_method weighted_average \
      --save_parquet
"""

import argparse
import os
from typing import Optional, List, Dict, Tuple
from collections import Counter

import pandas as pd
import numpy as np


def _ensure_dirs():
    os.makedirs("SP500_news/processed", exist_ok=True)
    os.makedirs("SP500_news/results", exist_ok=True)


# Default model configurations
DEFAULT_MODELS = {
    "finbert": {
        "model_id": "ProsusAI/finbert",
        "weight": 0.4,
        "description": "Financial domain BERT"
    },
    "finbert_tone": {
        "model_id": "yiyanghkust/finbert-tone",
        "weight": 0.3,
        "description": "Financial tone analysis"
    },
    "distilbert_sentiment": {
        "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "weight": 0.2,
        "description": "General sentiment (DistilBERT)"
    },
    "roberta_twitter": {
        "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "weight": 0.1,
        "description": "Twitter/social media sentiment"
    }
}


def _setup_pipeline(model_id: str, device: Optional[int] = None):
    """Setup a single model pipeline"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    import torch
    
    if device is None:
        device_idx = 0 if torch.cuda.is_available() else -1
    else:
        device_idx = device
    
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
        pipe = TextClassificationPipeline(
            model=mdl, 
            tokenizer=tok, 
            device=device_idx, 
            top_k=None, 
            truncation=True
        )
        return pipe
    except Exception as e:
        print(f"Warning: Failed to load {model_id}: {e}")
        return None


def _normalize_label(label: str) -> str:
    """Normalize label to POSITIVE/NEGATIVE/NEUTRAL"""
    label_upper = str(label).upper()
    
    # Positive indicators
    if any(x in label_upper for x in ["POS", "POSITIVE", "LABEL_1", "LABEL_2"]):
        # Some models use LABEL_1/LABEL_2, need to check context
        if "LABEL_1" in label_upper or "LABEL_2" in label_upper:
            # This depends on model - we'll handle in inference
            return "POSITIVE"
        return "POSITIVE"
    
    # Negative indicators
    if any(x in label_upper for x in ["NEG", "NEGATIVE", "LABEL_0"]):
        return "NEGATIVE"
    
    # Neutral
    return "NEUTRAL"


def _infer_single_model(pipe, texts: List[str], model_id: str, batch_size: int = 32, max_length: int = 128) -> Tuple[List[str], List[float], List[float]]:
    """Run inference for a single model"""
    if pipe is None:
        return None, None, None
    
    try:
        outs = pipe(texts, batch_size=batch_size, max_length=max_length)
        labels = []
        scores = []
        signed = []
        
        for o in outs:
            if isinstance(o, list):
                # Get the highest confidence prediction
                o = max(o, key=lambda x: x.get("score", 0.0))
            
            lab = str(o.get("label", "NEUTRAL"))
            sc = float(o.get("score", 0.0))
            
            # Normalize label
            normalized_lab = _normalize_label(lab)
            
            # Handle special cases for models with LABEL_0/LABEL_1/LABEL_2
            if "LABEL_0" in lab.upper() or "LABEL_1" in lab.upper() or "LABEL_2" in lab.upper():
                # For models like twitter-roberta: LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
                if "LABEL_0" in lab.upper():
                    normalized_lab = "NEGATIVE"
                elif "LABEL_2" in lab.upper():
                    normalized_lab = "POSITIVE"
                else:
                    normalized_lab = "NEUTRAL"
            
            # Calculate signed score
            if normalized_lab == "NEGATIVE":
                ssc = -sc
            elif normalized_lab == "POSITIVE":
                ssc = sc
            else:
                ssc = 0.0
            
            labels.append(normalized_lab)
            scores.append(sc)
            signed.append(ssc)
        
        return labels, scores, signed
    except Exception as e:
        print(f"Error in inference for {model_id}: {e}")
        return None, None, None


def ensemble_voting(all_predictions: List[Dict]) -> Tuple[List[str], List[float]]:
    """Ensemble via majority voting"""
    n_samples = len(all_predictions[0]["labels"])
    ensemble_labels = []
    ensemble_scores = []
    
    for i in range(n_samples):
        votes = [pred["labels"][i] for pred in all_predictions if pred["labels"] is not None]
        if not votes:
            ensemble_labels.append("NEUTRAL")
            ensemble_scores.append(0.0)
            continue
        
        # Majority vote
        vote_counts = Counter(votes)
        majority_label = vote_counts.most_common(1)[0][0]
        ensemble_labels.append(majority_label)
        
        # Average confidence of votes for majority class
        majority_scores = [pred["scores"][i] for pred, label in zip(all_predictions, votes) 
                          if label == majority_label and pred["scores"] is not None]
        ensemble_scores.append(np.mean(majority_scores) if majority_scores else 0.0)
    
    return ensemble_labels, ensemble_scores


def ensemble_weighted_average(all_predictions: List[Dict], weights: List[float]) -> Tuple[List[str], List[float], List[float]]:
    """Ensemble via weighted average of signed scores"""
    n_samples = len(all_predictions[0]["signed"])
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    ensemble_labels = []
    ensemble_scores = []
    ensemble_signed = []
    
    for i in range(n_samples):
        signed_scores = []
        valid_weights = []
        
        for pred, weight in zip(all_predictions, normalized_weights):
            if pred["signed"] is not None and i < len(pred["signed"]):
                signed_scores.append(pred["signed"][i])
                valid_weights.append(weight)
        
        if not signed_scores:
            ensemble_signed.append(0.0)
            ensemble_labels.append("NEUTRAL")
            ensemble_scores.append(0.0)
            continue
        
        # Weighted average
        weighted_avg = np.average(signed_scores, weights=valid_weights)
        ensemble_signed.append(weighted_avg)
        
        # Determine label from signed score
        if weighted_avg > 0.1:
            ensemble_labels.append("POSITIVE")
            ensemble_scores.append(abs(weighted_avg))
        elif weighted_avg < -0.1:
            ensemble_labels.append("NEGATIVE")
            ensemble_scores.append(abs(weighted_avg))
        else:
            ensemble_labels.append("NEUTRAL")
            ensemble_scores.append(abs(weighted_avg))
    
    return ensemble_labels, ensemble_scores, ensemble_signed


def run_ensemble_pipeline(
    raw_csv: str,
    date_col: str,
    title_col: str,
    model_configs: Optional[Dict] = None,
    ensemble_method: str = "weighted_average",
    device: Optional[int] = None,
    batch_size: int = 32,
    max_length: int = 128,
    save_parquet: bool = False
):
    """Run ensemble sentiment analysis pipeline"""
    _ensure_dirs()
    
    # Use default models if not provided
    if model_configs is None:
        model_configs = DEFAULT_MODELS
    
    # 1) Read data
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")
    df = pd.read_csv(raw_csv)
    
    if date_col not in df.columns or title_col not in df.columns:
        raise ValueError(f"Columns not found. Available: {list(df.columns)}; required: {date_col}, {title_col}")
    
    # Clean basic
    df[title_col] = df[title_col].astype(str).fillna("").str.strip()
    df = df[df[title_col] != ""].copy()
    
    # Parse date
    df["__date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[~df["__date"].isna()].copy()
    
    texts = df[title_col].tolist()
    n_texts = len(texts)
    
    print(f"Processing {n_texts} headlines with {len(model_configs)} models...")
    
    # 2) Load all models
    pipelines = {}
    for model_name, config in model_configs.items():
        print(f"Loading {model_name} ({config['model_id']})...")
        pipe = _setup_pipeline(config["model_id"], device)
        if pipe is not None:
            pipelines[model_name] = {
                "pipeline": pipe,
                "weight": config.get("weight", 1.0),
                "model_id": config["model_id"]
            }
        else:
            print(f"  Failed to load {model_name}, skipping...")
    
    if not pipelines:
        raise RuntimeError("No models loaded successfully!")
    
    print(f"Successfully loaded {len(pipelines)} models")
    
    # 3) Run inference for all models
    all_predictions = []
    weights = []
    
    for model_name, model_info in pipelines.items():
        print(f"Running inference with {model_name}...")
        labels, scores, signed = _infer_single_model(
            model_info["pipeline"], 
            texts, 
            model_info["model_id"],
            batch_size=batch_size,
            max_length=max_length
        )
        
        if labels is not None:
            all_predictions.append({
                "model_name": model_name,
                "labels": labels,
                "scores": scores,
                "signed": signed
            })
            weights.append(model_info["weight"])
            
            # Store individual model results
            df[f"{model_name}_label"] = labels
            df[f"{model_name}_score"] = scores
            df[f"{model_name}_signed"] = signed
    
    if not all_predictions:
        raise RuntimeError("No successful predictions from any model!")
    
    # 4) Ensemble predictions
    print(f"Ensembling predictions using {ensemble_method}...")
    
    if ensemble_method == "voting":
        ensemble_labels, ensemble_scores = ensemble_voting(all_predictions)
        ensemble_signed = [s if l == "POSITIVE" else (-s if l == "NEGATIVE" else 0.0) 
                          for l, s in zip(ensemble_labels, ensemble_scores)]
    elif ensemble_method == "weighted_average":
        ensemble_labels, ensemble_scores, ensemble_signed = ensemble_weighted_average(all_predictions, weights)
    else:
        raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    # Store ensemble results
    df["ensemble_label"] = ensemble_labels
    df["ensemble_score"] = ensemble_scores
    df["ensemble_signed"] = ensemble_signed
    
    # Also keep legacy column names for compatibility
    df["finbert_label"] = ensemble_labels
    df["finbert_score"] = ensemble_scores
    df["finbert_signed"] = ensemble_signed
    
    # 5) Persist enriched rows
    out_rows_csv = "SP500_news/processed/headlines_with_ensemble.csv"
    df.to_csv(out_rows_csv, index=False)
    print(f"Saved: {out_rows_csv}")
    
    if save_parquet:
        try:
            df.to_parquet(out_rows_csv.replace(".csv", ".parquet"), index=False)
            print(f"Saved: {out_rows_csv.replace('.csv', '.parquet')}")
        except Exception as e:
            print(f"Parquet save failed: {e}")
    
    # 6) Daily aggregation
    def coarse_label(x: str) -> str:
        x = (x or "").upper()
        if "NEG" in x:
            return "NEGATIVE"
        if "POS" in x:
            return "POSITIVE"
        return "NEUTRAL"
    
    df["ensemble_label_coarse"] = df["ensemble_label"].apply(coarse_label)
    
    grp = df.groupby(df["__date"].dt.date)
    
    daily = pd.DataFrame({
        "date": list(grp.groups.keys()),
        "news_count": grp.size().values,
        "pos_count": grp.apply(lambda g: (g["ensemble_label_coarse"] == "POSITIVE").sum()).values,
        "neg_count": grp.apply(lambda g: (g["ensemble_label_coarse"] == "NEGATIVE").sum()).values,
        "neu_count": grp.apply(lambda g: (g["ensemble_label_coarse"] == "NEUTRAL").sum()).values,
        "signed_mean": grp["ensemble_signed"].mean().values,
        "signed_std": grp["ensemble_signed"].std(ddof=0).values,
        "signed_min": grp["ensemble_signed"].min().values,
        "signed_max": grp["ensemble_signed"].max().values,
    })
    
    # Add individual model aggregations
    for model_name in pipelines.keys():
        if f"{model_name}_signed" in df.columns:
            daily[f"{model_name}_signed_mean"] = grp[f"{model_name}_signed"].mean().values
    
    daily = daily.sort_values("date").reset_index(drop=True)
    
    out_daily_csv = "SP500_news/processed/sp500_headlines_daily_agg_ensemble.csv"
    daily.to_csv(out_daily_csv, index=False)
    print(f"Saved: {out_daily_csv}")
    
    # Also save with legacy name for backtest compatibility
    legacy_daily_csv = "SP500_news/processed/sp500_headlines_daily_agg.csv"
    daily.to_csv(legacy_daily_csv, index=False)
    print(f"Saved (legacy): {legacy_daily_csv}")
    
    if save_parquet:
        try:
            daily.to_parquet(out_daily_csv.replace(".csv", ".parquet"), index=False)
        except Exception as e:
            print(f"Parquet save failed for daily: {e}")
    
    # 7) Results summary
    summary_lines = []
    summary_lines.append("SP500 Ensemble Sentiment Pipeline Summary\n")
    summary_lines.append("=" * 50 + "\n\n")
    summary_lines.append(f"Models used ({len(pipelines)}):\n")
    for model_name, model_info in pipelines.items():
        summary_lines.append(f"  - {model_name}: {model_info['model_id']} (weight: {model_info['weight']:.2f})\n")
    summary_lines.append(f"\nEnsemble method: {ensemble_method}\n\n")
    summary_lines.append(f"Raw rows read: {len(df)}\n")
    summary_lines.append(f"Date range: {daily['date'].min()} to {daily['date'].max()}\n")
    summary_lines.append(f"Days covered: {len(daily)}\n")
    summary_lines.append("\nEnsemble label distribution:\n")
    lbl_counts = df["ensemble_label_coarse"].value_counts()
    for k, v in lbl_counts.items():
        summary_lines.append(f"  {k}: {v}\n")
    summary_lines.append("\nDaily signed_mean stats:\n")
    summary_lines.append(str(daily["signed_mean"].describe()) + "\n")
    
    with open("SP500_news/results/ensemble_summary.txt", "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    
    print("\nSaved:")
    print(" -", out_rows_csv)
    print(" -", out_daily_csv)
    print(" -", legacy_daily_csv)
    print(" - SP500_news/results/ensemble_summary.txt")
    
    return df, daily


def main():
    p = argparse.ArgumentParser(description="SP500 Ensemble sentiment pipeline")
    p.add_argument("--raw_csv", type=str, default="SP500_news/raw/sp500_headlines_2008_2024.csv")
    p.add_argument("--date_col", type=str, default="Date")
    p.add_argument("--title_col", type=str, default="Title")
    p.add_argument("--ensemble_method", type=str, default="weighted_average", 
                   choices=["voting", "weighted_average"],
                   help="Ensemble method: voting or weighted_average")
    p.add_argument("--device", type=int, default=None, help="0=GPU, -1=CPU (default auto)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--save_parquet", action="store_true")
    p.add_argument("--models", type=str, nargs="+", default=None,
                   help="List of model names to use (default: all)")
    
    args = p.parse_args()
    
    # Filter models if specified
    model_configs = DEFAULT_MODELS.copy()
    if args.models:
        model_configs = {k: v for k, v in model_configs.items() if k in args.models}
        if not model_configs:
            print(f"Warning: No valid models found in {args.models}, using all defaults")
            model_configs = DEFAULT_MODELS.copy()
    
    run_ensemble_pipeline(
        raw_csv=args.raw_csv,
        date_col=args.date_col,
        title_col=args.title_col,
        model_configs=model_configs,
        ensemble_method=args.ensemble_method,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_parquet=args.save_parquet,
    )


if __name__ == "__main__":
    main()

