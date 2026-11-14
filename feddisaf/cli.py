"""Configuration for seven base-method + FedDisAF combinations."""

import argparse
import json
import random
import math
from pathlib import Path

from .combinations import COMBINATION_NAMES


DEFAULTS = {
    "base_algorithm": "FedAvg",
    "dataset": "PlantVillage", "data_dir": "./data", "backbone": "resnet18",
    "client_num_in_total": 10, "client_num_per_round": 5,
    "comm_round": 100, "local_epochs": 1, "batch_size": 32,
    "lr": 0.001, "momentum": 0.9, "wd": 0.001,
    "partition_method": "hetero", "partition_alpha": 0.1,
    "dp_noise_scale": 0.15, "proto_loss_weight": 1.0,
    "feddisaf_proto_temperature": 0.1,
    "feddisaf_num_heads": 8, "feddisaf_attention_dim": 256,
    "fedprox_mu": 0.01,
    "fedala_consistency_threshold": 0.8, "fedala_performance_decay": 0.9,
    "fedala_local_adaptation_steps": 3, "fedala_consistency_weight": 0.1,
    "fedala_personalization_factor": 0.3, "fedala_adaptive_weight_momentum": 0.8,
    "fedala_performance_history_length": 10,
    "fedftg_fine_tune_lr": 0.001, "fedftg_fine_tune_epochs": 1,
    "fedftg_gradient_momentum": 0.9, "fedftg_gradient_clip": 1.0,
    "fedftg_adaptive_weighting": True, "fedftg_consistency_weight": 0.1,
    "fedftg_performance_decay": 0.9,
    "fedpop_population_size": 30, "fedpop_mutation_rate": 0.15,
    "fedpop_crossover_rate": 0.7, "fedpop_elite_size": 5,
    "fedpop_evolution_interval": 3, "fedpop_fitness_decay": 0.95,
    "fedpop_sample_weight": 0.3, "fedpop_fitness_weight": 0.7,
    "fedpop_performance_history_length": 15, "fedpop_exploration_factor": 0.1,
    "fedpop_min_selection_prob": 0.01,
    "seed": 5, "device": "auto", "pretrained": True,
    "save_model": True, "log_dir": "./outputs", "summary_last_rounds": 5,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Train a base-method + FedDisAF combination.", allow_abbrev=False)
    parser.add_argument("--algorithm", dest="base_algorithm", choices=COMBINATION_NAMES + ("all",),
                        default=argparse.SUPPRESS,
                        help="Base method to combine with FedDisAF; all runs seven combinations")
    parser.add_argument("--config", type=Path, help="YAML configuration (CLI overrides YAML)")
    parser.add_argument("--print-config", action="store_true", help="Show configuration and exit")
    options = [
        ("dataset", "dataset", str), ("data-dir", "data_dir", str),
        ("backbone", "backbone", str), ("comm-round", "comm_round", int),
        ("num-clients", "client_num_in_total", int),
        ("num-selected", "client_num_per_round", int),
        ("local-epochs", "local_epochs", int), ("batch-size", "batch_size", int),
        ("lr", "lr", float), ("momentum", "momentum", float), ("wd", "wd", float),
        ("partition-method", "partition_method", str),
        ("partition-alpha", "partition_alpha", float),
        ("dp-noise", "dp_noise_scale", float),
        ("proto-loss-weight", "proto_loss_weight", float),
        ("temperature", "feddisaf_proto_temperature", float),
        ("feddisaf-num-heads", "feddisaf_num_heads", int),
        ("fedprox-mu", "fedprox_mu", float),
        ("seed", "seed", int), ("device", "device", str),
        ("log-dir", "log_dir", str),
        ("summary-last-rounds", "summary_last_rounds", int),
    ]
    for flag, dest, kind in options:
        flags = ["--" + flag]
        if "-" in flag:
            flags.append("--" + flag.replace("-", "_"))
        parser.add_argument(*flags, dest=dest, type=kind, default=argparse.SUPPRESS)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        default=argparse.SUPPRESS, help="Skip ImageNet weight download")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false",
                        default=argparse.SUPPRESS)
    return parser


def load_config(args):
    config = DEFAULTS.copy()
    if args.config:
        import yaml
        with args.config.open(encoding="utf-8") as stream:
            supplied = yaml.safe_load(stream) or {}
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must contain a YAML mapping")
        unknown = set(supplied) - set(DEFAULTS)
        if unknown:
            raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        config.update(supplied)
    config.update({k: v for k, v in vars(args).items() if k in DEFAULTS})
    if config["dataset"] not in ("PlantVillage", "ip102_v1.1"):
        raise ValueError("dataset must be PlantVillage or ip102_v1.1")
    if config["backbone"] not in ("resnet18", "resnet34", "resnet50"):
        raise ValueError("backbone must be resnet18, resnet34, or resnet50")
    for key in ("comm_round", "local_epochs", "client_num_in_total",
                "client_num_per_round", "summary_last_rounds"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if config["client_num_per_round"] > config["client_num_in_total"]:
        raise ValueError("Selected clients cannot exceed total clients")
    if not isinstance(config["batch_size"], int) or config["batch_size"] < 2:
        raise ValueError("batch_size must be an integer of at least 2")
    if config["partition_method"] not in ("iid", "hetero"):
        raise ValueError("partition_method must be iid or hetero")
    for key in ("lr", "partition_alpha", "feddisaf_proto_temperature"):
        if config[key] <= 0:
            raise ValueError(key + " must be positive")
    for key in ("dp_noise_scale", "wd", "proto_loss_weight"):
        if config[key] < 0:
            raise ValueError(key + " cannot be negative")
    if not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("pretrained", "save_model"):
        if not isinstance(config[key], bool):
            raise ValueError(key + " must be a YAML boolean")
    if config["base_algorithm"] not in COMBINATION_NAMES + ("all",):
        raise ValueError("Unknown base_algorithm; only FedDisAF combinations are supported")
    for key, value in config.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(key + " must be finite")
    heads = config["feddisaf_num_heads"]
    width = config["feddisaf_attention_dim"]
    if not isinstance(heads, int) or isinstance(heads, bool) or heads < 0:
        raise ValueError("feddisaf_num_heads must be a nonnegative integer")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1 or (heads and width % heads):
        raise ValueError("feddisaf_attention_dim must be positive and divisible by nonzero heads")
    for key in ("fedprox_mu", "fedftg_gradient_clip", "fedftg_consistency_weight"):
        if config[key] < 0:
            raise ValueError(key + " cannot be negative")
    for key in ("fedala_local_adaptation_steps", "fedala_performance_history_length",
                "fedpop_population_size", "fedpop_elite_size", "fedpop_evolution_interval",
                "fedpop_performance_history_length", "fedftg_fine_tune_epochs"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if config["fedpop_elite_size"] > config["fedpop_population_size"]:
        raise ValueError("fedpop_elite_size cannot exceed fedpop_population_size")
    for key in ("fedpop_mutation_rate", "fedpop_crossover_rate", "fedpop_fitness_decay",
                "fedpop_sample_weight", "fedpop_exploration_factor", "fedpop_min_selection_prob",
                "fedala_performance_decay", "fedala_adaptive_weight_momentum",
                "fedftg_gradient_momentum"):
        if not 0 <= config[key] <= 1:
            raise ValueError(key + " must be in [0, 1]")
    return config


def validate_data_layout(config):
    """Require explicit evaluation data to avoid implicit, changing split rules."""
    root = Path(config["data_dir"]) / config["dataset"]
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist: " + str(root))
    if config["dataset"] == "ip102_v1.1" and (root / "train.txt").is_file():
        if not (root / "val.txt").is_file():
            raise ValueError("IP102 TXT layout requires train.txt and val.txt")
        vocabularies = []
        for split in ("train.txt", "val.txt"):
            labels = {int(line.split()[1]) for line in (root / split).read_text(
                encoding="utf-8").splitlines() if line.strip()}
            vocabularies.append(labels)
    else:
        vocabularies = []
        for split in ("train", "val"):
            path = root / split
            if not path.is_dir():
                raise ValueError("Explicit train/ and val/ directories are required")
            vocabularies.append({p.name for p in path.iterdir() if p.is_dir()})
    if not vocabularies[0] or vocabularies[0] != vocabularies[1]:
        raise ValueError("Training and evaluation splits must have the same class vocabulary")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args)
        if args.print_config:
            print(json.dumps(config, indent=2))
            return
        validate_data_layout(config)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    # Import training dependencies only after --help / configuration validation.
    import numpy as np
    import torch
    from .data import get_data_loaders
    from .training import run_training

    if config["device"] == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    client_loaders, evaluation_loader, num_classes = get_data_loaders(config)
    config["num_classes"] = num_classes
    if len(client_loaders) != config["client_num_in_total"]:
        parser.error("Partition produced empty clients; use more data or fewer clients")
    from .data import describe_partition
    partition = describe_partition(client_loaders)
    names = COMBINATION_NAMES if config["base_algorithm"] == "all" else (config["base_algorithm"],)
    for name in names:
        # Reuse the partition and restart model/training RNGs for each combination.
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        run_config = dict(config, base_algorithm=name)
        result = run_training(run_config, client_loaders, evaluation_loader, config["log_dir"],
                              partition=partition)
        print("Results saved to:", result["experiment_dir"])


if __name__ == "__main__":
    main()
