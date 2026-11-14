"""Source-derived FedALAWithFedDisAF; see docs/PROVENANCE.md."""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from .common import FedDisAFCombination


class FedALAWithFedDisAF(FedDisAFCombination):
    """
    FedALA + FedDisAF

    FedALA核心思想: 自适应本地聚合
    - 为每个客户端动态计算聚合权重
    - 基于历史性能和当前损失
    """

    def __init__(self, config: Dict[str, Any], attention_aggregator=None):
        super().__init__(config)
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.proto_loss_weight = config.get('proto_loss_weight', 1.0)

        self.consistency_threshold = config.get('fedala_consistency_threshold', 0.8)
        self.performance_decay = config.get('fedala_performance_decay', 0.9)
        self.local_adaptation_steps = config.get('fedala_local_adaptation_steps', 3)
        self.consistency_weight = config.get('fedala_consistency_weight', 0.1)
        self.personalization_factor = config.get('fedala_personalization_factor', 0.3)
        self.adaptive_weight_momentum = config.get('fedala_adaptive_weight_momentum', 0.8)
        self.performance_history_length = config.get('fedala_performance_history_length', 10)

        self.attention_aggregator = attention_aggregator

        self.client_performance_history = {}
        self.client_adaptive_weights = {}

    def aggregate(self, client_updates: List[Dict[str, Any]],
                  client_features: List[torch.Tensor] = None,
                  global_prototypes: Dict[int, torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        """
        FedALA聚合 + FedDisAF多头注意力
        """
        if not client_updates:
            return {}

        adaptive_weights = self._compute_adaptive_weights(client_updates)

        attention_weights = None
        if client_features and self.config.get('feddisaf_enabled', True) and self.attention_aggregator:
            attention_weights = self.attention_aggregator.compute_attention_weights(client_features)

        final_weights = self._fuse_weights(adaptive_weights, attention_weights, client_updates)

        global_weights = self._weighted_aggregate(client_updates, final_weights)

        self.plugin.update_library(client_updates, global_prototypes)

        return global_weights

    def _compute_adaptive_weights(self, client_updates: List[Dict[str, Any]]) -> List[float]:
        """
        计算FedALA自适应权重
        基于: 1) 历史性能 2) 当前损失 3) 本地适应性
        """
        num_clients = len(client_updates)
        weights = []

        for idx, update in enumerate(client_updates):
            client_id = update.get('client_id', idx)
            current_loss = update.get('loss', 1.0)

            history = self.client_performance_history.get(client_id, [])

            if len(history) > 0:
                recent_performance = np.mean(history[-self.performance_history_length:]) if len(history) >= self.performance_history_length else np.mean(history)
                performance_ratio = current_loss / (recent_performance + 1e-8)

                adaptive_weight = 1.0 / (performance_ratio + 0.1)
                adaptive_weight = np.clip(adaptive_weight, 0.1, 2.0)
            else:
                adaptive_weight = 1.0

            local_adaptation = self._compute_local_adaptation(update)
            adaptive_weight *= (1 + self.local_adaptation_steps * local_adaptation)

            weights.append(adaptive_weight)

            self.client_performance_history.setdefault(client_id, []).append(current_loss)
            if len(self.client_performance_history[client_id]) > self.performance_history_length:
                self.client_performance_history[client_id].pop(0)

        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / num_clients] * num_clients

        return weights

    def _compute_local_adaptation(self, update: Dict[str, Any]) -> float:
        """计算本地适应性得分"""
        weights = update.get('weights', {})
        if not weights:
            return 0.0

        update_norms = []
        for param in weights.values():
            if hasattr(param, 'norm'):
                try:
                    if param.dtype != torch.float32:
                        param_float = param.float()
                    else:
                        param_float = param
                    update_norms.append(param_float.norm().item())
                except Exception:
                    continue

        if update_norms:
            avg_norm = np.mean(update_norms)
            adaptation = min(1.0, avg_norm / 10.0)
            return adaptation

        return 0.5

    def _fuse_weights(self, adaptive_weights: List[float],
                  attention_weights: Optional[torch.Tensor],
                  client_updates: List[Dict[str, Any]]) -> List[float]:
        """融合FedALA自适应权重和FedDisAF注意力权重"""
        num_clients = len(adaptive_weights)
        adaptive = np.array(adaptive_weights)

        if attention_weights is not None:
            attn = attention_weights.detach().cpu().numpy()

            sample_counts = np.array([u.get('num_samples', 1) for u in client_updates])
            sample_weights = sample_counts / sample_counts.sum()

            alpha = self.adaptive_weight_momentum
            beta = self.personalization_factor
            gamma = 0.1  # 样本权重系数

            fused = alpha * adaptive + (1 - alpha) * attn + gamma * sample_weights

            for idx in range(num_clients):
                client_id = client_updates[idx].get('client_id', idx)
                prev_weight = self.client_adaptive_weights.get(client_id, fused[idx])
                fused[idx] = self.performance_decay * prev_weight + (1 - self.performance_decay) * fused[idx]
                self.client_adaptive_weights[client_id] = fused[idx]
        else:
            fused = adaptive

        fused = fused / fused.sum()

        return fused.tolist()

    def _weighted_aggregate(self, client_updates: List[Dict[str, Any]],
                            weights: List[float]) -> Dict[str, torch.Tensor]:
        """加权聚合"""
        if not client_updates:
            return {}

        global_weights = {}
        weights_list = [update['weights'] for update in client_updates]

        for key in weights_list[0].keys():
            param = weights_list[0][key]
            global_weights[key] = torch.zeros_like(param, dtype=torch.float32)

        for idx, w in enumerate(weights_list):
            weight = weights[idx]
            for key in global_weights.keys():
                param = w[key]
                if param.dtype != torch.float32:
                    param = param.float()
                device = global_weights[key].device
                if param.device != device:
                    param = param.to(device)
                global_weights[key] += weight * param

        return global_weights

    def local_update(self, model: nn.Module, train_loader: DataLoader,
                     client_id: int = 0,
                     global_prototypes: Dict[int, torch.Tensor] = None) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        """
        FedALA本地训练
        """
        model.train()
        model.to(self.device)

        import torch.optim as optim
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.get('lr', 0.01),
            momentum=self.config.get('momentum', 0.9),
            weight_decay=self.config.get('wd', 1e-4)
        )

        total_loss = 0
        num_batches = 0
        proto_loss_sum = 0.0
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
