"""Numerical behavior of the FedDisAF prototype pathway."""
import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feddisaf.plugin import FedDisAFPlugin, fuse_prototypes
from feddisaf.combinations import build_combination
from feddisaf.models import ClientModel


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.generator = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)
        self.feature_dim = 4

    def forward(self, data):
        h = self.backbone(data)
        private = self.generator(h)
        return self.classifier(private), private, h - private


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.config = {"device": "cpu", "num_classes": 2, "local_epochs": 1,
                       "lr": 0.01, "wd": 0.0, "momentum": 0.0,
                       "dp_noise_scale": 0.0, "feddisaf_num_heads": 0}

    def test_cosine_fusion_downweights_opposite_outlier_and_is_permutation_invariant(self):
        prototypes = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]], requires_grad=True)
        fused = fuse_prototypes(prototypes)
        self.assertGreater(fused[0].item(), 0.99)
        self.assertTrue(torch.allclose(fused, fuse_prototypes(prototypes[[2, 0, 1]])))
        self.assertFalse(fused.requires_grad)
        self.assertEqual(fused.device.type, "cpu")

    def test_fusion_empty_singleton_and_zero_vectors(self):
        self.assertIsNone(fuse_prototypes(torch.empty(0, 4)))
        self.assertTrue(torch.equal(fuse_prototypes(torch.tensor([[2., -3.]])), torch.tensor([2., -3.])))
        self.assertTrue(torch.equal(fuse_prototypes(torch.zeros(3, 2)), torch.zeros(2)))

    def test_missing_upload_retains_library_and_empty_update_does_not_erase_it(self):
        old = torch.tensor([8., 9.])
        library = {0: None, 1: old.clone()}
        plugin = FedDisAFPlugin(self.config)
        plugin.update_library([{"prototypes": {0: torch.tensor([1., 2.])}}], library)
        plugin.update_library([], library)
        self.assertTrue(torch.equal(library[1], old))
        self.assertTrue(torch.equal(library[0], torch.tensor([1., 2.])))

    def test_prototype_loss_is_class_average_detaches_input_and_restores_modes(self):
        model = TinyModel()
        model.classifier = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.4), nn.Linear(4, 2))
        model.train()
        model.classifier[0].eval()
        modes = [m.training for m in model.classifier.modules()]
        vectors = {0: torch.randn(4, requires_grad=True), 1: torch.randn(4, requires_grad=True)}
        model.classifier.eval()
        expected = sum(nn.functional.cross_entropy(model.classifier(v.unsqueeze(0)), torch.tensor([c]))
                       for c, v in vectors.items()) / 2
        for module, mode in zip(model.classifier.modules(), modes):
            module.training = mode
        actual = FedDisAFPlugin(self.config).prototype_loss(model, vectors)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(modes, [m.training for m in model.classifier.modules()])
        actual.backward()
        self.assertTrue(all(v.grad is None for v in vectors.values()))
        self.assertIsNone(model.backbone.weight.grad)
        self.assertIsNone(model.generator.weight.grad)
        self.assertIsNotNone(model.classifier[-1].weight.grad)

    def test_local_prototypes_use_visited_preupdate_features_and_omit_absent_class(self):
        model = TinyModel()
        x = torch.randn(4, 3)
        y = torch.zeros(4, dtype=torch.long)
        with torch.no_grad():
            shared = model(x)[2].clone()
        update, _ = build_combination(self.config).local_update(
            model, DataLoader(TensorDataset(x, y), batch_size=4))
        self.assertEqual(set(update["prototypes"]), {0})
        self.assertTrue(torch.allclose(update["prototypes"][0], shared.mean(0)))

    def test_local_prototype_supervision_changes_classifier_direct_update_only(self):
        plain = TinyModel()
        guided = copy.deepcopy(plain)
        loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=4)
        algorithm = build_combination(self.config)
        a, _ = algorithm.local_update(plain, loader)
        b, _ = algorithm.local_update(guided, loader, global_prototypes={0: torch.ones(4), 1: None})
        for name in a["weights"]:
            if name.startswith(("backbone.", "generator.")):
                self.assertTrue(torch.equal(a["weights"][name], b["weights"][name]))
        self.assertFalse(torch.equal(a["weights"]["classifier.weight"], b["weights"]["classifier.weight"]))

    def test_resnet_singleton_keeps_gradients_and_restores_batchnorm(self):
        model = ClientModel(num_classes=2, pretrained=False).train()
        logits, private, shared = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(tuple(private.shape), (1, 512))
        self.assertEqual(tuple(shared.shape), (1, 512))
        logits.sum().backward()
        self.assertIsNotNone(model.generator[0].weight.grad)
        self.assertTrue(model.generator[1].training)


if __name__ == "__main__":
    unittest.main()
