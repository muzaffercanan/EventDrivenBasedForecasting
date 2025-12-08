"""
Fine-tuning script for sentiment models on SP500 news

This script provides a framework for fine-tuning pre-trained sentiment models
on SP500-specific news data to improve prediction accuracy.

Usage:
    python fine_tune_sentiment_model.py \
        --model_name finbert \
        --train_data SP500_news/processed/headlines_with_ensemble.csv \
        --output_dir models/finetuned_finbert_sp500
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch


def prepare_training_data(df: pd.DataFrame, 
                         title_col: str = "Title",
                         label_col: str = "ensemble_label") -> Tuple[List[str], List[int]]:
    """
    Prepare training data from headlines and labels
    
    Converts POSITIVE/NEGATIVE/NEUTRAL labels to 0/1/2
    """
    texts = df[title_col].astype(str).tolist()
    
    # Convert labels to integers
    label_map = {"POSITIVE": 1, "NEGATIVE": 0, "NEUTRAL": 2}
    labels = [label_map.get(label.upper(), 2) for label in df[label_col]]
    
    # Filter out invalid entries
    valid_indices = [i for i, (t, l) in enumerate(zip(texts, labels)) 
                     if len(t.strip()) > 10 and l in [0, 1, 2]]
    
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    return texts, labels


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def fine_tune_model(
    model_name: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str] = None,
    val_labels: List[int] = None,
    output_dir: str = "models/finetuned_model",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128
):
    """
    Fine-tune a sentiment model on SP500 news data
    """
    print("=" * 60)
    print(f"FINE-TUNING {model_name.upper()} ON SP500 NEWS")
    print("=" * 60)
    
    # Model mapping
    model_map = {
        "finbert": "ProsusAI/finbert",
        "finbert_tone": "yiyanghkust/finbert-tone",
        "distilbert": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest"
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    
    model_id = model_map[model_name]
    
    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=3  # POSITIVE, NEGATIVE, NEUTRAL
    )
    
    # Prepare datasets
    print(f"\nPreparing datasets...")
    print(f"  Training samples: {len(train_texts)}")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
    
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'labels': train_labels
    })
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    
    # Validation set
    if val_texts and val_labels:
        print(f"  Validation samples: {len(val_texts)}")
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        val_dataset = val_dataset.map(tokenize_function, batched=True)
    else:
        # Split training data for validation
        split_idx = int(len(train_texts) * 0.9)
        val_texts = train_texts[split_idx:]
        val_labels = train_labels[split_idx:]
        train_texts = train_texts[:split_idx]
        train_labels = train_labels[:split_idx]
        
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })
        val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        warmup_steps=100,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    trainer.train()
    
    # Evaluate
    print(f"\nEvaluating...")
    eval_results = trainer.evaluate()
    print(f"  Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Validation F1: {eval_results['eval_f1']:.4f}")
    
    # Save
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "base_model": model_id,
        "num_train_samples": len(train_texts),
        "num_val_samples": len(val_texts),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "eval_results": {k: float(v) for k, v in eval_results.items()}
    }
    
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nFine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    
    return trainer, eval_results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentiment models on SP500 news")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=["finbert", "finbert_tone", "distilbert", "roberta"],
                       help="Model to fine-tune")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Training data CSV file")
    parser.add_argument("--output_dir", type=str,
                       default="models/finetuned_model",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Check if training data exists
    if not os.path.exists(args.train_data):
        print(f"Error: Training data not found: {args.train_data}")
        return
    
    # Load data
    print(f"Loading training data from {args.train_data}...")
    df = pd.read_csv(args.train_data)
    
    if 'Title' not in df.columns or 'ensemble_label' not in df.columns:
        print("Error: CSV must contain 'Title' and 'ensemble_label' columns")
        return
    
    # Prepare training data
    texts, labels = prepare_training_data(df)
    
    if len(texts) < 100:
        print(f"Warning: Only {len(texts)} training samples. Fine-tuning may not be effective.")
        print("Consider using more data or using the pre-trained model as-is.")
    
    # Split train/val
    split_idx = int(len(texts) * (1 - args.val_split))
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    val_texts = texts[split_idx:]
    val_labels = labels[split_idx:]
    
    # Fine-tune
    trainer, eval_results = fine_tune_model(
        model_name=args.model_name,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )
    
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"\nTo use the fine-tuned model, update sp500_ensemble_sentiment_pipeline.py")
    print(f"to use model path: {args.output_dir}")
    print(f"\nExample:")
    print(f'  "finbert": {{"model_id": "{args.output_dir}", "weight": 0.4}}')


if __name__ == "__main__":
    main()

