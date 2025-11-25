import wandb
from architectures.helpers.constants import hyperparameters, selected_model, threshold

hyperparameters = hyperparameters[selected_model]

def initialize_wandb():
    print(f"Initializing W&B with selected_model: {selected_model}")
    config_base = {
        "model": f"{selected_model}",
        "learning_rate": hyperparameters["learning_rate_type"],
        "epochs": hyperparameters["num_epochs"],
        "batch_size": hyperparameters["batch_size"],
        "threshold": f"0.{threshold}",
        "image_size": hyperparameters["image_size"]
    }

    if selected_model in ["convmixer", "convmixer_tf"]:
        config = {
            **config_base,
            "weight_decay": hyperparameters["weight_decay"],
            "filters": hyperparameters["filters"],
            "depth": hyperparameters["depth"],
            "kernel_size": hyperparameters["kernel_size"],
            "patch_size": hyperparameters["patch_size"]
        }
    elif selected_model in ["vision_transformer", "vit"]:
        config = {
            **config_base,
            "weight_decay": hyperparameters["weight_decay"],
            "projection_dim": hyperparameters["projection_dim"],
            "num_heads": hyperparameters["num_heads"],
            "patch_size": hyperparameters["patch_size"],
            "transformer_layers": hyperparameters["transformer_layers"],
            "layer_norm_eps": hyperparameters.get("layer_norm_eps", 1e-6)
        }
    elif selected_model == "mlp_mixer":
        config = {
            **config_base,
            "weight_decay": hyperparameters["weight_decay"],
            "dropout_rate": hyperparameters["dropout_rate"],
            "embedding_dim": hyperparameters["embedding_dim"],
            "patch_size": hyperparameters["patch_size"],
            "num_blocks": hyperparameters["num_blocks"]
        }
    elif selected_model == "cnn_ta":
        config = {
            **config_base,
            "first_dropout_rate": hyperparameters["first_dropout_rate"],
            "second_dropout_rate": hyperparameters["second_dropout_rate"]
        }
    else:
        config = config_base

    wandb.init(
        project=f"{selected_model}",
        entity="muzcaaa-student",  # Burayı güncelledik
        config=config,
        settings=wandb.Settings(disable_code=True)
    )
    print(f"W&B initialized for project: {selected_model}, entity: muzcaaa-student")