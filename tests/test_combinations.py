"""Exercise every integrated combination, not standalone baseline algorithms."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feddisaf.combinations import COMBINATION_NAMES, build_combination
from feddisaf.training import run_training
from test_core import TinyModel


class ModelWithBuffers(TinyModel):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))


def evaluate_tiny(model, loader, device):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x)[0].argmax(1) == y).sum().item()
            total += len(y)
    return correct * 100 / total, 1.0, correct / total


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, name):
        return {"base_algorithm": name, "device": "cpu", "num_classes": 2,
                "local_epochs": 1, "lr": 0.01, "wd": 0.0, "momentum": 0.0,
                "dp_noise_scale": 0.05, "feddisaf_num_heads": 2,
                "feddisaf_attention_dim": 8, "feddisaf_feature_dim": 4,
                "feature_dim": 4, "client_num_in_total": 2, "client_num_per_round": 2,
                "comm_round": 4, "save_model": True, "seed": 11,
                "fedpop_population_size": 4, "fedpop_elite_size": 1,
                "fedpop_evolution_interval": 1}

    def test_all_seven_train_aggregate_and_transfer_prototypes(self):
        for name in COMBINATION_NAMES:
            with self.subTest(name=name):
                torch.manual_seed(13)
                np.random.seed(13)
                config = self.config(name)
                algorithm = build_combination(config)
                self.assertTrue(type(algorithm).__name__.endswith("WithFedDisAF"))
                loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=4)
                library = {0: None, 1: None}
                updates, features = [], []
                for client in range(2):
                    update, feature = algorithm.local_update(ModelWithBuffers(), loader, client, library)
                    updates.append(update)
                    features.append(feature)
                weights = algorithm.aggregate(updates, features, library)
                self.assertIsNotNone(library[0])
                self.assertIsNotNone(library[1])
                restored = ModelWithBuffers()
                restored.load_state_dict(weights)
                update, _ = algorithm.local_update(restored, loader, 0, library)
                self.assertTrue(all(torch.isfinite(v).all() for v in update["weights"].values()))
                self.assertGreater(algorithm.plugin.prototype_loss(restored, library).item(), 0)

    def test_four_round_training_loop_for_every_combination(self):
        loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=4)
        for name in COMBINATION_NAMES:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                torch.manual_seed(11)
                np.random.seed(11)
                config = self.config(name)
                with patch("feddisaf.training.get_client_model", side_effect=lambda cfg: ModelWithBuffers()), \
                     patch("feddisaf.training.evaluate", side_effect=evaluate_tiny):
                    result = run_training(config, [loader, loader], loader, temp)
                self.assertEqual(result["algorithm"], name + "+FedDisAF")
                directory = Path(result["experiment_dir"])
                self.assertEqual(json.loads((directory / "history.json").read_text())["rounds"], [1, 2, 3, 4])
                self.assertTrue((directory / "final_model.pth").is_file())
                self.assertIn("within_run_round_std_accuracy", result)

    def test_fedavg_uses_sample_weights_and_input_states_are_unchanged(self):
        algorithm = build_combination(dict(self.config("FedAvg"), feddisaf_num_heads=0))
        updates = [{"num_samples": 1, "weights": {"weight": torch.tensor([2.])}},
                   {"num_samples": 3, "weights": {"weight": torch.tensor([10.])}}]
        actual = algorithm.aggregate(updates)
        self.assertAlmostEqual(actual["weight"].item(), 8.)
        self.assertEqual(updates[0]["weights"]["weight"].item(), 2.)

    def test_factory_rejects_plugin_off_and_unknown_modes(self):
        with self.assertRaisesRegex(ValueError, "combinations only"):
            build_combination(dict(self.config("FedAvg"), feddisaf_enabled=False))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            build_combination(dict(self.config("FedAvg"), base_algorithm="FedFed"))

    def test_cli_all_dispatches_only_combinations_with_the_same_partition(self):
        from feddisaf.cli import main
        loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=4)
        manifest = {"client_indices": [[0, 1], [2, 3]]}
        with tempfile.TemporaryDirectory() as temp:
            for split in ("train", "val"):
                for label in ("a", "b"):
                    (Path(temp) / "PlantVillage" / split / label).mkdir(parents=True)
            with patch("feddisaf.data.get_data_loaders", return_value=([loader, loader], loader, 2)), \
                 patch("feddisaf.data.describe_partition", return_value=manifest), \
                 patch("feddisaf.training.run_training", return_value={"experiment_dir": temp}) as run:
                main(["--algorithm", "all", "--data-dir", temp, "--num-clients", "2",
                      "--num-selected", "2", "--device", "cpu", "--no-pretrained"])
            self.assertEqual([call.args[0]["base_algorithm"] for call in run.call_args_list], list(COMBINATION_NAMES))
            self.assertTrue(all(call.kwargs["partition"] is manifest for call in run.call_args_list))

    def test_partition_manifest_records_unassigned_and_duplicate_indices(self):
        from torch.utils.data import Subset
        from feddisaf.data import describe_partition
        data = TensorDataset(torch.randn(5, 3), torch.zeros(5, dtype=torch.long))
        data.samples = [("image_%d.jpg" % i, 0) for i in range(5)]
        loaders = [DataLoader(Subset(data, [0, 1])), DataLoader(Subset(data, [1, 3]))]
        manifest = describe_partition(loaders)
        self.assertEqual(manifest["unassigned_indices"], [2, 4])
        self.assertEqual(manifest["duplicate_assignments"], 1)
        self.assertEqual(manifest["assigned_samples"], 4)


if __name__ == "__main__":
    unittest.main()
