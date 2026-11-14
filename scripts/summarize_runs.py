"""Aggregate independent runs, reducing rounds within each run before seeds."""

import argparse
import json
import math
from pathlib import Path


MATCH_KEYS = (
    "algorithm", "base_algorithm", "dataset", "backbone", "num_classes", "comm_round",
    "client_num_in_total", "client_num_per_round", "local_epochs", "batch_size",
    "lr", "momentum", "wd", "partition_method", "partition_alpha",
    "dp_noise_scale", "proto_loss_weight", "feddisaf_proto_temperature", "pretrained",
    "feddisaf_num_heads", "feddisaf_attention_dim",
)


def mean_std(values, ddof=1):
    if ddof not in (0, 1) or len(values) <= ddof or not values:
        raise ValueError("Need more independent values than ddof (0 or 1)")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Metric values must be finite")
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((v - mean)**2 for v in values) / (len(values) - ddof))
    return mean, std


def summarize_directories(directories, last_rounds=5, tail_fraction=None, ddof=1):
    directories = [Path(p).resolve() for p in directories]
    if len(set(directories)) != len(directories) or len(directories) < 2:
        raise ValueError("Supply at least two different independent run directories")
    if tail_fraction is not None and not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    if last_rounds < 1:
        raise ValueError("last_rounds must be positive")
    seed_results = []
    reference = None
    used_seeds = set()
    for directory in directories:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        keys = set(MATCH_KEYS) | {key for key in config if key.startswith(("fedala_", "fedprox_", "fedftg_", "fedpop_"))}
        signature = {key: config.get(key) for key in sorted(keys)}
        if reference is None:
            reference = signature
        elif reference != signature:
            raise ValueError("Run configurations differ: " + str(directory))
        seed = config["seed"]
        if seed in used_seeds:
            raise ValueError("Each run must have a distinct random seed")
        used_seeds.add(seed)
        history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        total = config["comm_round"]
        if history["rounds"] != list(range(1, total + 1)):
            raise ValueError("Run is incomplete or round ordering is invalid: " + str(directory))
        count = max(1, math.ceil(total * tail_fraction)) if tail_fraction is not None else min(last_rounds, total)
        values = {"seed": seed, "directory": str(directory),
                  "first_round": total - count + 1, "last_round": total,
                  "round_count": count}
        for source, name, scale in (("test_accuracy", "accuracy_percent", 1),
                                    ("test_recall", "recall_percent", 100),
                                    ("test_loss", "loss", 1)):
            if len(history[source]) != total:
                raise ValueError("History metric length does not match rounds")
            metrics = [float(v) * scale for v in history[source][-count:]]
            values[name] = mean_std(metrics, ddof=0)[0]
        seed_results.append(values)
    result = {"method": reference["algorithm"], "config": reference, "num_seeds": len(seed_results),
              "ddof": ddof, "sd_unit": "independent seed-level means",
              "window": {"last_rounds": None if tail_fraction is not None else last_rounds,
                         "tail_fraction": tail_fraction}, "seed_results": seed_results}
    for metric in ("accuracy_percent", "recall_percent", "loss"):
        mean, std = mean_std([r[metric] for r in seed_results], ddof=ddof)
        result[metric] = {"mean": mean, "std": std}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Directories with config.json and history.json")
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument("--last-rounds", type=int, default=5)
    windows.add_argument("--tail-fraction", type=float, help="For example, 0.5 for the latter half")
    parser.add_argument("--ddof", choices=(0, 1), type=int, default=1,
                        help="SD denominator N-ddof; default is sample SD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = summarize_directories(args.runs, args.last_rounds, args.tail_fraction, args.ddof)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
