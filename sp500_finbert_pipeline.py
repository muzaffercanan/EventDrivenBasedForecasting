"""
SP500 FinBERT Sentiment Pipeline (standalone)

Steps:
1) Read raw headlines CSV from SP500_news/raw/sp500_headlines_2008_2024.csv
2) Run FinBERT sentiment (transformers) to get label and signed_score
3) Aggregate to daily features (news_count, pos/neg/neu counts, mean/std/min/max signed)
4) Save outputs to:
   - SP500_news/processed/headlines_with_finbert.csv
   - SP500_news/processed/sp500_headlines_daily_agg.csv
   - SP500_news/results/summary.txt (basic stats)

Requirements:
  pip install pandas numpy transformers torch pyarrow

Usage:
  python sp500_finbert_pipeline.py \
      --raw_csv SP500_news/raw/sp500_headlines_2008_2024.csv \
      --date_col Date --title_col Title \
      --save_parquet
"""

import argparse
import os
from typing import Optional, List

import pandas as pd
import numpy as np


def _ensure_dirs():
    os.makedirs("SP500_news/processed", exist_ok=True)
    os.makedirs("SP500_news/results", exist_ok=True)


def _setup_pipeline(model_id: str, device: Optional[int] = None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    import torch
    if device is None:
        device_idx = 0 if torch.cuda.is_available() else -1
    else:
        device_idx = device
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, device=device_idx, top_k=None, truncation=True)
    return pipe


def _infer(pipe, texts: List[str], batch_size: int = 32, max_length: int = 128):
    outs = pipe(texts, batch_size=batch_size, max_length=max_length)
    labels = []
    scores = []
    signed = []
    for o in outs:
        if isinstance(o, list):
            o = max(o, key=lambda x: x.get("score", 0.0))
        lab = str(o.get("label", "NEUTRAL")).upper()
        sc = float(o.get("score", 0.0))
        if "NEG" in lab:
            ssc = -sc
        elif "POS" in lab:
            ssc = sc
        else:
            ssc = 0.0
        labels.append(lab)
        scores.append(sc)
        signed.append(ssc)
    return labels, scores, signed


def run_pipeline(raw_csv: str,
                 date_col: str,
                 title_col: str,
                 model_id: str = "ProsusAI/finbert",
                 device: Optional[int] = None,
                 batch_size: int = 32,
                 max_length: int = 128,
                 save_parquet: bool = False):
    _ensure_dirs()

    # 1) Read
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

    # 2) FinBERT inference
    pipe = _setup_pipeline(model_id, device)
    texts = df[title_col].tolist()
    labels, scores, signed = _infer(pipe, texts, batch_size=batch_size, max_length=max_length)
    df["finbert_label"] = labels
    df["finbert_score"] = scores
    df["finbert_signed"] = signed

    # Persist enriched rows
    out_rows_csv = "SP500_news/processed/headlines_with_finbert.csv"
    df.to_csv(out_rows_csv, index=False)

    if save_parquet:
        try:
            df.to_parquet(out_rows_csv.replace(".csv", ".parquet"), index=False)
        except Exception as e:
            print(f"Parquet save failed for rows: {e}")

    # 3) Daily aggregation
    # Normalize label into coarse POS/NEG/NEU
    def coarse_label(x: str) -> str:
        x = (x or "").upper()
        if "NEG" in x:
            return "NEGATIVE"
        if "POS" in x:
            return "POSITIVE"
        return "NEUTRAL"

    df["finbert_label_coarse"] = df["finbert_label"].apply(coarse_label)

    grp = df.groupby(df["__date"].dt.date)

    daily = pd.DataFrame({
        "date": list(grp.groups.keys()),
        "news_count": grp.size().values,
        "pos_count": grp.apply(lambda g: (g["finbert_label_coarse"] == "POSITIVE").sum()).values,
        "neg_count": grp.apply(lambda g: (g["finbert_label_coarse"] == "NEGATIVE").sum()).values,
        "neu_count": grp.apply(lambda g: (g["finbert_label_coarse"] == "NEUTRAL").sum()).values,
        "signed_mean": grp["finbert_signed"].mean().values,
        "signed_std": grp["finbert_signed"].std(ddof=0).values,
        "signed_min": grp["finbert_signed"].min().values,
        "signed_max": grp["finbert_signed"].max().values,
    })
    daily = daily.sort_values("date").reset_index(drop=True)

    out_daily_csv = "SP500_news/processed/sp500_headlines_daily_agg.csv"
    daily.to_csv(out_daily_csv, index=False)

    if save_parquet:
        try:
            daily.to_parquet(out_daily_csv.replace(".csv", ".parquet"), index=False)
        except Exception as e:
            print(f"Parquet save failed for daily: {e}")

    # 4) Results summary
    summary_lines = []
    summary_lines.append("SP500 FinBERT Pipeline Summary\n===============================\n")
    summary_lines.append(f"Raw rows read: {len(df)}\n")
    summary_lines.append(f"Date range: {daily['date'].min()} to {daily['date'].max()}\n")
    summary_lines.append(f"Days covered: {len(daily)}\n")
    summary_lines.append("\nLabel distribution (coarse):\n")
    lbl_counts = df["finbert_label_coarse"].value_counts()
    for k, v in lbl_counts.items():
        summary_lines.append(f"  {k}: {v}\n")
    summary_lines.append("\nDaily signed_mean stats:\n")
    summary_lines.append(str(daily["signed_mean"].describe()) + "\n")

    with open("SP500_news/results/summary.txt", "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    print("Saved:")
    print(" -", out_rows_csv)
    print(" -", out_daily_csv)
    print(" - SP500_news/results/summary.txt")


def main():
    p = argparse.ArgumentParser(description="SP500 FinBERT sentiment pipeline (standalone)")
    p.add_argument("--raw_csv", type=str, default="SP500_news/raw/sp500_headlines_2008_2024.csv")
    p.add_argument("--date_col", type=str, default="Date")
    p.add_argument("--title_col", type=str, default="Title")
    p.add_argument("--model_id", type=str, default="ProsusAI/finbert")
    p.add_argument("--device", type=int, default=None, help="0=GPU, -1=CPU (default auto)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--save_parquet", action="store_true")
    args = p.parse_args()

    run_pipeline(
        raw_csv=args.raw_csv,
        date_col=args.date_col,
        title_col=args.title_col,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        save_parquet=args.save_parquet,
    )


if __name__ == "__main__":
    main()
