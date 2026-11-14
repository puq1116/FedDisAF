"""The shared FedDisAF prototype pathway, independent of model aggregation."""
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


def fuse_prototypes(prototypes: torch.Tensor, temperature: float = 0.1):
    """Cosine weights relative to the class centroid; return a detached CPU vector."""
    if prototypes is None or prototypes.numel() == 0:
        return None
    if prototypes.size(0) == 1:
        return prototypes[0].detach().cpu()
    centroid = prototypes.mean(dim=0, keepdim=True)
    similarity = F.cosine_similarity(prototypes, centroid.expand_as(prototypes), dim=-1)
    weights = torch.softmax(similarity / max(float(temperature), 1e-6), dim=0)
    return (weights.unsqueeze(-1) * prototypes).sum(dim=0).detach().cpu()


class FedDisAFPlugin:
    """Prototype supervision and library updates used by all seven combinations.

    The combination owns its local optimizer and model-state aggregation. The
    client model supplies the backbone/private/complementary feature decomposition.
    """

    def __init__(self, config):
        self.device = config.get("device", "cpu")
        self.num_classes = config.get("num_classes", 10)
        self.temperature = config.get("feddisaf_proto_temperature", 0.1)
        self.criterion = nn.CrossEntropyLoss()

    def prototype_loss(self, model: nn.Module, library: Optional[Dict]):
        available = [(label, library[label]) for label in range(self.num_classes)
                     if library and library.get(label) is not None]
        if not available:
            return 0.0
        # Restore each submodule's original mode, including mixed train/eval modes.
        modes = [(module, module.training) for module in model.classifier.modules()]
        model.classifier.eval()
        try:
            loss = sum(self.criterion(
                model.classifier(vector.detach().to(self.device).unsqueeze(0)),
                torch.tensor([label], device=self.device),
            ) for label, vector in available)
            return loss / len(available)
        finally:
            for module, training in modes:
                module.training = training

    def update_library(self, client_updates: List[Dict], library: Optional[Dict]):
        if library is None:
            return
        for label in range(self.num_classes):
            uploads = [update.get("prototypes", {}).get(label) for update in client_updates]
            uploads = [vector.to(self.device) for vector in uploads if vector is not None]
            if uploads:
                library[label] = fuse_prototypes(torch.stack(uploads), self.temperature)

