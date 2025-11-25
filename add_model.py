"""
Helper script to add new models to the ensemble pipeline

Makes it easy to add new sentiment models without editing the main pipeline file.
"""

import argparse
import json
import os
from typing import Dict


def load_model_configs(config_file: str = "model_configs.json") -> Dict:
    """Load model configurations from file"""
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        # Return default models
        from sp500_ensemble_sentiment_pipeline import DEFAULT_MODELS
        return DEFAULT_MODELS.copy()


def save_model_configs(configs: Dict, config_file: str = "model_configs.json"):
    """Save model configurations to file"""
    with open(config_file, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"✓ Configurations saved to {config_file}")


def add_model_interactive():
    """Interactive mode to add a new model"""
    print("=" * 60)
    print("ADD NEW SENTIMENT MODEL")
    print("=" * 60)
    
    model_name = input("\nModel name (e.g., 'bert_financial'): ").strip()
    if not model_name:
        print("✗ Model name cannot be empty")
        return
    
    model_id = input("HuggingFace model ID (e.g., 'ProsusAI/finbert'): ").strip()
    if not model_id:
        print("✗ Model ID cannot be empty")
        return
    
    weight_str = input("Weight (default 0.1, will be normalized): ").strip()
    try:
        weight = float(weight_str) if weight_str else 0.1
    except ValueError:
        print("✗ Invalid weight, using default 0.1")
        weight = 0.1
    
    description = input("Description (optional): ").strip() or f"Model: {model_id}"
    
    # Load existing configs
    configs = load_model_configs()
    
    # Check if model already exists
    if model_name in configs:
        overwrite = input(f"Model '{model_name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("✗ Cancelled")
            return
    
    # Add model
    configs[model_name] = {
        "model_id": model_id,
        "weight": weight,
        "description": description
    }
    
    # Save
    save_model_configs(configs)
    
    print(f"\n✓ Model '{model_name}' added successfully!")
    print(f"\nTo use this model, update sp500_ensemble_sentiment_pipeline.py")
    print(f"or use the --models argument with the new model name.")


def add_model_from_args(model_name: str, model_id: str, weight: float = 0.1, 
                        description: str = None, config_file: str = "model_configs.json"):
    """Add model from command line arguments"""
    configs = load_model_configs(config_file)
    
    if model_name in configs:
        print(f"Warning: Model '{model_name}' already exists. Overwriting...")
    
    configs[model_name] = {
        "model_id": model_id,
        "weight": weight,
        "description": description or f"Model: {model_id}"
    }
    
    save_model_configs(configs, config_file)
    print(f"✓ Model '{model_name}' added successfully!")


def list_models(config_file: str = "model_configs.json"):
    """List all configured models"""
    configs = load_model_configs(config_file)
    
    print("=" * 60)
    print("CONFIGURED MODELS")
    print("=" * 60)
    
    if not configs:
        print("\nNo models configured")
        return
    
    print(f"\nTotal models: {len(configs)}\n")
    
    for name, config in configs.items():
        print(f"Name: {name}")
        print(f"  Model ID: {config['model_id']}")
        print(f"  Weight: {config['weight']:.3f}")
        print(f"  Description: {config.get('description', 'N/A')}")
        print()


def remove_model(model_name: str, config_file: str = "model_configs.json"):
    """Remove a model from configuration"""
    configs = load_model_configs(config_file)
    
    if model_name not in configs:
        print(f"✗ Model '{model_name}' not found")
        return
    
    del configs[model_name]
    save_model_configs(configs, config_file)
    print(f"✓ Model '{model_name}' removed successfully!")


def test_model(model_name: str = None, model_id: str = None):
    """Test if a model can be loaded"""
    from sp500_ensemble_sentiment_pipeline import _setup_pipeline, _infer_single_model
    
    if model_id is None:
        if model_name is None:
            print("✗ Must provide either model_name or model_id")
            return
        
        configs = load_model_configs()
        if model_name not in configs:
            print(f"✗ Model '{model_name}' not found in configurations")
            return
        
        model_id = configs[model_name]["model_id"]
    
    print(f"Testing model: {model_id}")
    
    try:
        pipe = _setup_pipeline(model_id, device=None)
        if pipe is None:
            print("✗ Failed to load model")
            return
        
        print("✓ Model loaded successfully")
        
        # Test inference
        test_texts = [
            "Stock market reaches new high",
            "Economic downturn concerns grow"
        ]
        
        labels, scores, signed = _infer_single_model(
            pipe, test_texts, model_id, batch_size=2, max_length=128
        )
        
        if labels and scores:
            print("✓ Inference successful")
            for text, label, score in zip(test_texts, labels, signed):
                print(f"  '{text}' -> {label} ({score:.3f})")
        else:
            print("✗ Inference failed")
    
    except Exception as e:
        print(f"✗ Error: {e}")


def update_pipeline_file(config_file: str = "model_configs.json"):
    """Update the DEFAULT_MODELS in pipeline file with custom configs"""
    configs = load_model_configs(config_file)
    
    # Read pipeline file
    pipeline_file = "sp500_ensemble_sentiment_pipeline.py"
    if not os.path.exists(pipeline_file):
        print(f"✗ Pipeline file not found: {pipeline_file}")
        return
    
    with open(pipeline_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Generate new DEFAULT_MODELS dict
    models_str = "DEFAULT_MODELS = {\n"
    for name, config in configs.items():
        models_str += f'    "{name}": {{\n'
        models_str += f'        "model_id": "{config["model_id"]}",\n'
        models_str += f'        "weight": {config["weight"]},\n'
        models_str += f'        "description": "{config.get("description", "")}"\n'
        models_str += "    },\n"
    models_str += "}\n"
    
    # Find and replace DEFAULT_MODELS
    import re
    pattern = r'DEFAULT_MODELS = \{.*?\n\}'
    new_content = re.sub(pattern, models_str, content, flags=re.DOTALL)
    
    if new_content != content:
        # Backup original
        backup_file = pipeline_file + ".backup"
        with open(backup_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Backup saved to {backup_file}")
        
        # Write updated content
        with open(pipeline_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✓ Updated {pipeline_file} with new model configurations")
    else:
        print("✗ Could not find DEFAULT_MODELS to replace")


def main():
    parser = argparse.ArgumentParser(description="Manage sentiment models for ensemble pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add model
    parser_add = subparsers.add_parser("add", help="Add a new model")
    parser_add.add_argument("--name", type=str, required=True, help="Model name")
    parser_add.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser_add.add_argument("--weight", type=float, default=0.1, help="Model weight")
    parser_add.add_argument("--description", type=str, help="Model description")
    parser_add.add_argument("--config", type=str, default="model_configs.json", help="Config file")
    
    # List models
    parser_list = subparsers.add_parser("list", help="List all models")
    parser_list.add_argument("--config", type=str, default="model_configs.json", help="Config file")
    
    # Remove model
    parser_remove = subparsers.add_parser("remove", help="Remove a model")
    parser_remove.add_argument("--name", type=str, required=True, help="Model name")
    parser_remove.add_argument("--config", type=str, default="model_configs.json", help="Config file")
    
    # Test model
    parser_test = subparsers.add_parser("test", help="Test a model")
    parser_test.add_argument("--name", type=str, help="Model name")
    parser_test.add_argument("--model_id", type=str, help="HuggingFace model ID")
    parser_test.add_argument("--config", type=str, default="model_configs.json", help="Config file")
    
    # Update pipeline
    parser_update = subparsers.add_parser("update", help="Update pipeline file with configs")
    parser_update.add_argument("--config", type=str, default="model_configs.json", help="Config file")
    
    # Interactive add
    parser_interactive = subparsers.add_parser("add-interactive", help="Add model interactively")
    
    args = parser.parse_args()
    
    if args.command == "add":
        add_model_from_args(
            args.name, args.model_id, args.weight, args.description, args.config
        )
    elif args.command == "list":
        list_models(args.config)
    elif args.command == "remove":
        remove_model(args.name, args.config)
    elif args.command == "test":
        test_model(args.name, args.model_id)
    elif args.command == "update":
        update_pipeline_file(args.config)
    elif args.command == "add-interactive":
        add_model_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

