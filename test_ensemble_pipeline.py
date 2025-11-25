"""
Test script for ensemble sentiment pipeline

Tests the pipeline with a small sample to verify all models load correctly.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Import the ensemble pipeline functions
from sp500_ensemble_sentiment_pipeline import (
    _setup_pipeline,
    _infer_single_model,
    DEFAULT_MODELS,
    run_ensemble_pipeline
)


def test_model_loading():
    """Test if all models can be loaded"""
    print("=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)
    
    results = {}
    
    for model_name, config in DEFAULT_MODELS.items():
        print(f"\nTesting {model_name} ({config['model_id']})...")
        try:
            pipe = _setup_pipeline(config["model_id"], device=None)
            if pipe is not None:
                print(f"  [OK] {model_name} loaded successfully")
                results[model_name] = {"status": "success", "pipe": pipe}
            else:
                print(f"  [FAIL] {model_name} failed to load")
                results[model_name] = {"status": "failed", "pipe": None}
        except Exception as e:
            print(f"  [ERROR] {model_name} error: {e}")
            results[model_name] = {"status": "error", "error": str(e), "pipe": None}
    
    return results


def test_inference():
    """Test inference with sample texts"""
    print("\n" + "=" * 60)
    print("TEST 2: Model Inference")
    print("=" * 60)
    
    # Sample financial headlines
    test_texts = [
        "Stock market surges to record high on strong earnings",
        "Economic downturn fears grow as inflation rises",
        "Federal Reserve keeps interest rates unchanged",
        "Tech stocks plummet amid regulatory concerns",
        "Oil prices stabilize after volatile trading session"
    ]
    
    results = {}
    
    for model_name, config in DEFAULT_MODELS.items():
        print(f"\nTesting inference with {model_name}...")
        try:
            pipe = _setup_pipeline(config["model_id"], device=None)
            if pipe is None:
                print(f"  [SKIP] Cannot test - model failed to load")
                results[model_name] = {"status": "skipped"}
                continue
            
            labels, scores, signed = _infer_single_model(
                pipe, test_texts, config["model_id"], batch_size=2, max_length=128
            )
            
            if labels and scores and signed:
                print(f"  [OK] Inference successful")
                print(f"    Sample predictions:")
                for i, text in enumerate(test_texts[:2]):
                    print(f"      '{text[:50]}...' -> {labels[i]} ({signed[i]:.3f})")
                results[model_name] = {
                    "status": "success",
                    "sample_labels": labels[:2],
                    "sample_signed": signed[:2]
                }
            else:
                print(f"  [FAIL] Inference returned None")
                results[model_name] = {"status": "failed"}
        except Exception as e:
            print(f"  [ERROR] Inference error: {e}")
            results[model_name] = {"status": "error", "error": str(e)}
    
    return results


def test_pipeline_with_sample():
    """Test full pipeline with a small sample"""
    print("\n" + "=" * 60)
    print("TEST 3: Full Pipeline (Sample Data)")
    print("=" * 60)
    
    # Check if raw CSV exists
    raw_csv = "SP500_news/raw/sp500_headlines_2008_2024.csv"
    if not os.path.exists(raw_csv):
        print(f"  [INFO] Raw CSV not found: {raw_csv}")
        print("  Creating sample data for testing...")
        
        # Create sample data
        sample_data = {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "Title": [
                "Stock market reaches new all-time high",
                "Economic indicators show mixed signals",
                "Federal Reserve announces rate cut",
                "Tech sector faces regulatory challenges",
                "Oil prices surge on supply concerns"
            ]
        }
        
        os.makedirs("SP500_news/raw", exist_ok=True)
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(raw_csv, index=False)
        print(f"  [OK] Created sample CSV: {raw_csv}")
    
    try:
        print(f"\nRunning pipeline with sample data...")
        df, daily = run_ensemble_pipeline(
            raw_csv=raw_csv,
            date_col="Date",
            title_col="Title",
            ensemble_method="weighted_average",
            batch_size=2,
            max_length=128
        )
        
        print(f"\n  [OK] Pipeline completed successfully")
        print(f"    Processed {len(df)} headlines")
        print(f"    Generated {len(daily)} daily aggregations")
        print(f"\n  Sample ensemble results:")
        print(df[["Title", "ensemble_label", "ensemble_signed"]].head().to_string(index=False))
        
        return True
    except Exception as e:
        print(f"  [ERROR] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ENSEMBLE PIPELINE TEST SUITE")
    print("=" * 60)
    
    # Test 1: Model loading
    load_results = test_model_loading()
    
    # Test 2: Inference
    infer_results = test_inference()
    
    # Test 3: Full pipeline
    pipeline_success = test_pipeline_with_sample()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    successful_models = [name for name, res in load_results.items() 
                        if res.get("status") == "success"]
    failed_models = [name for name, res in load_results.items() 
                    if res.get("status") != "success"]
    
    print(f"\nModels loaded: {len(successful_models)}/{len(DEFAULT_MODELS)}")
    if successful_models:
        print(f"  [OK] Successful: {', '.join(successful_models)}")
    if failed_models:
        print(f"  [FAIL] Failed: {', '.join(failed_models)}")
    
    print(f"\nPipeline test: {'[PASSED]' if pipeline_success else '[FAILED]'}")
    
    if len(successful_models) > 0:
        print("\n[OK] At least one model works - pipeline can run")
    else:
        print("\n[FAIL] No models loaded - check dependencies and internet connection")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

