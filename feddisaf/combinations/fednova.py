"""Source-derived FedNovaWithFedDisAF; see docs/PROVENANCE.md."""
import copy
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from .common import FedDisAFCombination


class FedNovaWithFedDisAF(FedDisAFCombination):

    def __init__(self, config: Dict[str, Any], attention_aggregator=None):
        super().__init__(config)
        self.config = config
        self.device = config.get('device', 'cuda')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.proto_loss_weight = config.get('proto_loss_weight', 1.0)

        self.attention_aggregator = attention_aggregator

    def aggregate(self, client_updates: List[Dict[str, Any]],
                  client_features: List[torch.Tensor] = None,
                  global_prototypes: Dict[int, torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        global_weights = self._aggregate_models(client_updates)

        self.plugin.update_library(client_updates, global_prototypes)

        return global_weights

    def local_update(self, model: torch.nn.Module, train_loader: DataLoader,
                     client_id: int = 0,
                     global_prototypes: Dict[int, torch.Tensor] = None) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        model.train()
        model.to(self.device)

        import torch.optim as optim
        optimizer = optim.SGD(model.parameters(), lr=self.config.get('lr', 0.01))

        total_loss = 0
        num_steps = 0

        noisy_shared_features = []
        labels_list = []

        for epoch in range(self.config.get('local_epochs', 5)):
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                logits, x_p, x_s = model(data)
                loss_local = self.criterion(logits, targets)

                loss_global = self.plugin.prototype_loss(model, global_prototypes)

                loss = loss_local + self.proto_loss_weight * loss_global
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_steps += 1

                dp_noise_scale = self.config.get('dp_noise_scale', 0.0)
                if dp_noise_scale > 0:
                    noise = torch.randn_like(x_s) * dp_noise_scale
                    x_s_noisy = x_s + noise
                else:
                    x_s_noisy = x_s
                noisy_shared_features.append(x_s_noisy.detach().cpu())
                labels_list.append(targets.detach().cpu())

        num_epochs = self.config.get('local_epochs', 5)
        rho = 0.9
        a_k = (num_epochs - rho * (1 - rho**num_epochs) / (1 - rho)) / (1 - rho)

        prototypes = {}
        if noisy_shared_features:
            all_features = torch.cat(noisy_shared_features, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            num_classes = self.config.get('num_classes', 10)
            for c in range(num_classes):
                mask = (all_labels == c)
                if mask.sum() > 0:
                    prototypes[c] = all_features[mask].mean(dim=0)
                else:
                    prototypes[c] = torch.zeros(self.config.get('feature_dim', 512))

        update = {
            'weights': copy.deepcopy(model.state_dict()),
            'num_samples': len(train_loader.dataset),
            'loss': total_loss / max(num_steps, 1),
            'coeff': a_k,
            'train_loader': train_loader,
            'client_id': client_id,
            'prototypes': prototypes
        }

        features = self.extract_client_features(model, train_loader)

        return update, features

    def _aggregate_models(self, client_updates):
        if not client_updates:
            return {}

        total_coeff = sum(u.get('coeff', 1.0) for u in client_updates)
        global_weights = {}

        for key in client_updates[0]['weights'].keys():
            global_weights[key] = torch.zeros_like(client_updates[0]['weights'][key], dtype=torch.float32)

        for update in client_updates:
            weight = update.get('coeff', 1.0) / total_coeff
            for key in global_weights.keys():
                global_weights[key] += weight * update['weights'][key].float()

        return global_weights
