"""Source-derived FedFTGWithFedDisAF; see docs/PROVENANCE.md."""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from .common import FedDisAFCombination


class FedFTGWithFedDisAF(FedDisAFCombination):
    """
    FedFTG + FedDisAF

    FedFTG核心思想: 特征传输引导
    - 维护全局特征表示
    - 使用特征引导损失约束本地训练
    - 自适应特征聚合
    """

    def __init__(self, config: Dict[str, Any], attention_aggregator=None):
        super().__init__(config)
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.proto_loss_weight = config.get('proto_loss_weight', 1.0)

        self.fine_tune_lr = config.get('fedftg_fine_tune_lr', 0.001)
        self.fine_tune_epochs = config.get('fedftg_fine_tune_epochs', 1)
        self.gradient_momentum = config.get('fedftg_gradient_momentum', 0.9)
        self.gradient_clip = config.get('fedftg_gradient_clip', 1.0)
        self.adaptive_weighting = config.get('fedftg_adaptive_weighting', True)
        self.consistency_weight = config.get('fedftg_consistency_weight', 0.1)
        self.performance_decay = config.get('fedftg_performance_decay', 0.9)

        self.attention_aggregator = attention_aggregator

        self.global_features = None
        self.client_feature_history = {}

    def aggregate(self, client_updates: List[Dict[str, Any]],
                  client_features: List[torch.Tensor] = None,
                  global_prototypes: Dict[int, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        FedFTG聚合 + FedDisAF多头注意力
        """
        if not client_updates:
            return {}

        client_feats = []
        for idx, update in enumerate(client_updates):
            feat = update.get('features')
            if feat is not None:
                client_feats.append(feat)
            elif client_features and idx < len(client_features):
                client_feats.append(client_features[idx])

        if client_feats:
            self._update_global_features(client_feats, client_updates)

        ftg_weights = self._compute_ftg_weights(client_updates)

        attention_weights = None
        if client_features and self.config.get('feddisaf_enabled', True) and self.attention_aggregator:
            attention_weights = self.attention_aggregator.compute_attention_weights(client_features)

        final_weights = self._fuse_weights(ftg_weights, attention_weights, client_updates)

        global_weights = self._weighted_aggregate(client_updates, final_weights)

        self.plugin.update_library(client_updates, global_prototypes)

        return global_weights

    def _update_global_features(self, client_features: List[torch.Tensor],
                                 client_updates: List[Dict[str, Any]]):
        """
        更新全局特征表示 (FedFTG特征引导)
        使用加权平均更新全局特征
        """
        if not client_features:
            return

        total_loss = sum(u.get('loss', 1.0) for u in client_updates)
        weights = [1.0 / (u.get('loss', 1.0) + 1e-8) for u in client_updates]
        weights = [w / sum(weights) for w in weights]

        device = self.device
        for i, feat in enumerate(client_features):
            if feat is not None and feat.device != device:
                client_features[i] = feat.to(device)

        valid_features = [f for f in client_features if f is not None]
        if not valid_features:
            return

        new_features = torch.zeros_like(valid_features[0], device=device, dtype=torch.float32)

        for feat, weight in zip(client_features, weights):
            if feat is not None:
                if feat.device != device:
                    feat = feat.to(device)
                new_features += weight * feat

        if self.global_features is None:
            self.global_features = new_features
        else:
            if self.global_features.device != device:
                self.global_features = self.global_features.to(device)
            self.global_features = (self.gradient_momentum * self.global_features +
                                    (1 - self.gradient_momentum) * new_features)

    def _compute_ftg_weights(self, client_updates: List[Dict[str, Any]]) -> List[float]:
        """
        计算FedFTG自适应权重
        基于: 1) 特征一致性 2) 损失 3) 历史性能
        """
        weights = []
        device = self.device

        for idx, update in enumerate(client_updates):
            client_id = update.get('client_id', idx)
            loss = update.get('loss', 1.0)

            base_weight = 1.0 / (loss + 0.1)

            consistency_weight = 1.0
            if self.global_features is not None and update.get('features') is not None:
                client_feat = update['features']
                if client_feat.device != device:
                    client_feat = client_feat.to(device)
                similarity = torch.cosine_similarity(
                    client_feat.unsqueeze(0),
                    self.global_features.unsqueeze(0)
                ).item()
                consistency_weight = np.exp(similarity * self.consistency_weight)

            history = self.client_feature_history.get(client_id, [])
            history_weight = 1.0
            if len(history) > 0:
                recent_perf = np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
                history_weight = 1.0 / (recent_perf + 0.1)

            weight = base_weight * consistency_weight * history_weight
            weights.append(weight)

            self.client_feature_history.setdefault(client_id, []).append(loss)
            if len(self.client_feature_history[client_id]) > 20:
                self.client_feature_history[client_id].pop(0)

        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(client_updates)] * len(client_updates)

        return weights

    def _fuse_weights(self, ftg_weights: List[float],
                      attention_weights: Optional[torch.Tensor],
                      client_updates: List[Dict[str, Any]]) -> List[float]:
        """融合FedFTG权重和注意力权重"""
        num_clients = len(ftg_weights)
        ftg = np.array(ftg_weights)

        if attention_weights is not None:
            attn = attention_weights.detach().cpu().numpy()

            feature_guide_strength = []
            for update in client_updates:
                if self.global_features is not None and update.get('features') is not None:
                    client_feat = update['features']
                    if client_feat.device != self.device:
                        client_feat = client_feat.to(self.device)
                    similarity = torch.cosine_similarity(
                        client_feat.unsqueeze(0),
                        self.global_features.unsqueeze(0)
                    ).item()
                    feature_guide_strength.append(np.exp(similarity))
                else:
                    feature_guide_strength.append(1.0)

            feature_guide = np.array(feature_guide_strength)
            feature_guide = feature_guide / feature_guide.sum()

            alpha = 0.4  # FedFTG权重系数
            beta = 0.4   # 注意力权重系数
            gamma = 0.2  # 特征引导系数

            fused = alpha * ftg + beta * attn + gamma * feature_guide
        else:
            fused = ftg

        fused = fused / fused.sum()

        return fused.tolist()

    def _weighted_aggregate(self, client_updates: List[Dict[str, Any]],
                            weights: List[float]) -> Dict[str, torch.Tensor]:
        """加权聚合 - 确保使用 float32 类型"""
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
        FedFTG本地训练 - 带特征引导损失
        """
        model.train()
        model.to(self.device)

        import torch.optim as optim
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.get('lr', 0.01),
            momentum=self.gradient_momentum,
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

                if self.global_features is not None:
                    global_feat = self.global_features.to(self.device)
                    feature_guide_loss = self.consistency_weight * torch.norm(
                        x_p.mean(dim=0) - global_feat
                    )
                    loss_local += feature_guide_loss

                loss_global = self.plugin.prototype_loss(model, global_prototypes)

                loss = loss_local + self.proto_loss_weight * loss_global
                proto_loss_sum += loss_global.item() if isinstance(loss_global, torch.Tensor) else float(loss_global)
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
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
