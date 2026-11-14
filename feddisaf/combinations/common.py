"""Shared interface for combinations; no standalone baseline implementation."""
import torch
from torch import nn

from ..plugin import FedDisAFPlugin


class FedDisAFCombination:
    def __init__(self, config):
        if config.get("feddisaf_enabled", True) is not True:
            raise ValueError("This archive provides FedDisAF combinations only")
        self.config = config
        self.device = config.get("device", "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.plugin = FedDisAFPlugin(config)

    def extract_client_features(self, model, loader):
        """Retain the source's post-training private-feature summary."""
        if self.attention_aggregator is not None:
            return self.attention_aggregator.extract_client_features(model, loader)
        model.eval()
        features = []
        with torch.no_grad():
            for data, _ in loader:
                features.append(model(data.to(self.device))[1].detach().cpu())
                if len(features) * data.size(0) >= 100:
                    break
        return torch.cat(features).mean(dim=0) if features else None

