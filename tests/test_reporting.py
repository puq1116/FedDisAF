"""Configuration, archive scope, and independent-run reduction without torch."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_runs import summarize_directories
from feddisaf.cli import build_parser, load_config
from feddisaf.combinations import COMBINATION_NAMES


class ReportingTests(unittest.TestCase):
    def make_runs(self, root):
        runs = []
        for seed, base in ((1, 10), (2, 20), (3, 30)):
            directory = Path(root) / str(seed)
            directory.mkdir()
            config = {"algorithm": "FedAvg+FedDisAF", "base_algorithm": "FedAvg",
                      "dataset": "PlantVillage", "comm_round": 6, "seed": seed}
            history = {"rounds": list(range(1, 7)), "test_accuracy": [base + t for t in range(6)],
                       "test_recall": [seed * 0.1] * 6, "test_loss": [1.] * 6}
            (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (directory / "history.json").write_text(json.dumps(history), encoding="utf-8")
            runs.append(directory)
        return runs

    def test_final_five_then_sample_sd_of_three_run_means(self):
        with tempfile.TemporaryDirectory() as temp:
            result = summarize_directories(self.make_runs(temp))
            self.assertEqual(result["method"], "FedAvg+FedDisAF")
            self.assertEqual(result["accuracy_percent"], {"mean": 23., "std": 10.})
            self.assertAlmostEqual(result["recall_percent"]["mean"], 20.)
            self.assertEqual(result["seed_results"][0]["first_round"], 2)

    def test_other_combinations_and_duplicate_seeds_cannot_be_pooled(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = self.make_runs(temp)
            path = runs[1] / "config.json"
            config = json.loads(path.read_text())
            config["base_algorithm"] = "FedProx"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "configurations differ"):
                summarize_directories(runs)
            config["base_algorithm"] = "FedAvg"
            config["seed"] = 1
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "distinct random seed"):
                summarize_directories(runs)

    def test_incomplete_histories_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = self.make_runs(temp)
            path = runs[0] / "history.json"
            history = json.loads(path.read_text())
            history["rounds"] = history["rounds"][:-1]
            path.write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summarize_directories(runs)

    def test_cli_has_exactly_seven_combinations_and_all(self):
        parser = build_parser()
        action = next(a for a in parser._actions if a.dest == "base_algorithm")
        self.assertEqual(tuple(action.choices), COMBINATION_NAMES + ("all",))
        self.assertEqual(load_config(parser.parse_args(["--algorithm", "FedProx"]))["base_algorithm"], "FedProx")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--feddisaf_enabled", "False"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--algorithm", "FedFed"])

    def test_config_rejects_invalid_heads_clients_and_nonfinite_values(self):
        parser = build_parser()
        for args in (["--feddisaf-num-heads", "3"], ["--num-clients", "2", "--num-selected", "3"],
                     ["--lr", "nan"], ["--temperature", "inf"]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                load_config(parser.parse_args(args))

    def test_archive_exports_no_standalone_baseline_classes(self):
        root = Path(__file__).resolve().parents[1] / "feddisaf"
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ClassDef):
                    self.assertFalse(node.name.endswith("Base"), (path, node.name))


if __name__ == "__main__":
    unittest.main()
