"""Backbone-level additive feature decomposition and local classification."""

from typing import Any, Dict, Tuple

import torch
from torch import nn
from torchvision import models


class ClientModel(nn.Module):
    """Return class logits, private features, and complementary shared features."""

    def __init__(self, backbone="resnet18", num_classes=10, pretrained=True):
        super().__init__()
        dimensions = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}
        if backbone not in dimensions:
            raise ValueError("Supported backbones: resnet18, resnet34, resnet50")
        constructor = getattr(models, backbone)
        enum = getattr(models, backbone.replace("resnet", "ResNet") + "_Weights", None)
        if enum is not None:
            self.backbone = constructor(weights=enum.IMAGENET1K_V1 if pretrained else None)
        else:
            self.backbone = constructor(pretrained=pretrained)
        self.backbone.fc = nn.Identity()
        self.feature_dim = dimensions[backbone]
        self.generator = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim), nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # A final singleton batch can occur in a small non-IID client partition.
        # Use running BatchNorm statistics for that batch, while keeping gradients.
        frozen = []
        if self.training and x.size(0) == 1:
            frozen = [m for m in self.modules()
                      if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training]
            for module in frozen:
                module.eval()
        try:
            h = self.backbone(x)
            x_p = self.generator(h)
            x_s = h - x_p
            return self.classifier(x_p), x_p, x_s
        finally:
            for module in frozen:
                module.train()


def get_client_model(config: Dict[str, Any]) -> ClientModel:
    return ClientModel(
        backbone=config.get("backbone", "resnet18"),
        num_classes=config.get("num_classes", 10),
        pretrained=config.get("pretrained", True),
    )
