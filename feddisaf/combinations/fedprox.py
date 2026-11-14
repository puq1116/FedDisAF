"""Source-derived FedProxWithFedDisAF; see docs/PROVENANCE.md."""
import copy
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from .common import FedDisAFCombination


class FedProxWithFedDisAF(FedDisAFCombination):

    def __init__(self, config: Dict[str, Any], attention_aggregator=None):
        super().__init__(config)
        self.config = config
        self.device = config.get('device', 'cuda')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mu = config.get('fedprox_mu', 0.01)
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
        num_batches = 0

        noisy_shared_features = []
        labels_list = []
        proto_loss_sum = 0.0

        global_model_params = copy.deepcopy({k: v.detach() for k, v in model.named_parameters()})

        for epoch in range(self.config.get('local_epochs', 5)):
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                logits, x_p, x_s = model(data)

                loss_local = self.criterion(logits, targets)

                proximal_loss = 0.0
                for name, param in model.named_parameters():
                    if name in global_model_params:
                        proximal_loss += torch.norm(param - global_model_params[name], p=2) ** 2
                loss_prox = (self.mu / 2) * proximal_loss

                loss_global = self.plugin.prototype_loss(model, global_prototypes)

                loss = loss_local + loss_prox + self.proto_loss_weight * loss_global
                proto_loss_sum += loss_global.item() if isinstance(loss_global, torch.Tensor) else float(loss_global)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                dp_noise_scale = self.config.get('dp_noise_scale', 0.0)
                if dp_noise_scale > 0:
                    noise = torch.randn_like(x_s) * dp_noise_scale
                    x_s_noisy = x_s + noise
                else:
                    x_s_noisy = x_s
                noisy_shared_features.append(x_s_noisy.detach().cpu())
                labels_list.append(targets.detach().cpu())

                prototypes = {}
                if noisy_shared_features:
                    all_features = torch.cat(noisy_shared_features, dim=0)
                    all_labels = torch.cat(labels_list, dim=0)
                    num_classes = self.config.get('num_classes', 10)
                    feature_dim = getattr(model, 'feature_dim',
                                          self.config.get('feature_dim', 512))
                    for c in range(num_classes):
                        mask = (all_labels == c)
                        if mask.sum() > 0:
                            prototypes[c] = all_features[mask].mean(dim=0)

                update = {
                    'weights': copy.deepcopy(model.state_dict()),
                    'num_samples': len(train_loader.dataset),
                    'loss': total_loss / max(num_batches, 1),
                    'proto_loss': proto_loss_sum / max(num_batches, 1),
                    'train_loader': train_loader,
                    'client_id': client_id,
                    'prototypes': prototypes
                }

        features = self.extract_client_features(model, train_loader)

        return update, features

    def _aggregate_models(self, client_updates, client_weights=None):
        """
        聚合客户端更新

        Args:
            client_updates: 客户端更新列表
            client_weights: 客户端权重列表，如果为None则使用均匀权重
        """
        if not client_updates:
            return {}

        if client_weights is None:
            client_weights = [1.0 / len(client_updates)] * len(client_updates)
        elif isinstance(client_weights, list):
            total = sum(client_weights)
            if total > 0:
                client_weights = [w / total for w in client_weights]

        global_weights = {}
        first_update = client_updates[0]['weights']

        for key, value in first_update.items():
            if isinstance(value, torch.Tensor):
                if value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    global_weights[key] = torch.zeros_like(value, dtype=torch.float32)
                else:
                    global_weights[key] = torch.zeros_like(value, dtype=torch.float32)
            else:
                global_weights[key] = torch.tensor(0.0, dtype=torch.float32)

        total_weight = 0
        for weight, update in zip(client_weights, client_updates):
            total_weight += weight
            for key in global_weights.keys():
                if key not in update['weights']:
                    continue

                update_value = update['weights'][key]
                if isinstance(update_value, torch.Tensor):
                    if update_value.dtype not in [torch.float32, torch.float64]:
                        update_value = update_value.float()
                else:
                    update_value = torch.tensor(update_value, dtype=torch.float32)

                global_weights[key] += weight * update_value

        if total_weight > 0:
            for key in global_weights.keys():
                global_weights[key] = global_weights[key] / total_weight

        return global_weights
