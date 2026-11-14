"""Start independent FedDisAF processes and aggregate the resulting run means."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

from summarize_runs import summarize_directories


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_seed"))
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument("--last-rounds", type=int, default=5)
    windows.add_argument("--tail-fraction", type=float)
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=1)
    args, training_args = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from feddisaf.cli import build_parser, load_config
    training_parser = build_parser()
    try:
        resolved = load_config(training_parser.parse_args(training_args))
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if resolved["base_algorithm"] == "all":
        parser.error("Run one combination per multi-seed invocation; choose a specific --algorithm")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Supply at least two distinct seeds")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be integers in [0, 2**32)")
    if args.last_rounds < 1 or (args.tail_fraction is not None and not 0 < args.tail_fraction <= 1):
        parser.error("Invalid round summary window")
    reserved = ("--seed", "--log-dir", "--log_dir", "--summary-last-rounds", "--summary_last_rounds", "--print-config")
    if any(token.split("=", 1)[0] in reserved for token in training_args):
        parser.error("Seed, log directory and summary window are managed by this script")
    destination = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runs = []
    for seed in args.seeds:
        log_dir = destination / ("seed_%d" % seed)
        subprocess.run([sys.executable, str(root / "main.py")] + training_args + [
            "--seed", str(seed), "--log-dir", str(log_dir),
            "--summary-last-rounds", str(args.last_rounds)], check=True)
        candidates = list(log_dir.glob("*/history.json"))
        if len(candidates) != 1:
            raise RuntimeError("Expected exactly one completed run for seed %d" % seed)
        runs.append(candidates[0].parent)
    result = summarize_directories(runs, args.last_rounds, args.tail_fraction, args.ddof)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (destination / "aggregate.json").write_text(text, encoding="utf-8")
    print(text)
    print("Multi-seed summary:", destination / "aggregate.json")


if __name__ == "__main__":
    main()
