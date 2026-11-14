"""Source-derived FedPOPWithFedDisAF; see docs/PROVENANCE.md."""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from .common import FedDisAFCombination


class FedPOPWithFedDisAF(FedDisAFCombination):
    """
    FedPOP + FedDisAF

    FedPOP核心思想: 基于种群的优化
    - 维护模型种群
    - 使用进化策略 (交叉、变异)
    - 精英保留机制
    """

    def __init__(self, config: Dict[str, Any], attention_aggregator=None):
        super().__init__(config)
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.proto_loss_weight = config.get('proto_loss_weight', 1.0)

        self.population_size = config.get('fedpop_population_size', 30)
        self.mutation_rate = config.get('fedpop_mutation_rate', 0.15)
        self.crossover_rate = config.get('fedpop_crossover_rate', 0.7)
        self.elite_size = config.get('fedpop_elite_size', 5)
        self.evolution_interval = config.get('fedpop_evolution_interval', 3)
        self.fitness_decay = config.get('fedpop_fitness_decay', 0.95)
        self.sample_weight = config.get('fedpop_sample_weight', 0.3)
        self.fitness_weight = config.get('fedpop_fitness_weight', 0.7)
        self.performance_history_length = config.get('fedpop_performance_history_length', 15)
        self.exploration_factor = config.get('fedpop_exploration_factor', 0.1)
        self.min_selection_prob = config.get('fedpop_min_selection_prob', 0.01)

        self.attention_aggregator = attention_aggregator

        self.population = []  # 存储 (weights, fitness, client_id)
        self.elite_population = []
        self.generation = 0

        self.client_fitness_history = {}

    def aggregate(self, client_updates: List[Dict[str, Any]],
                  client_features: List[torch.Tensor] = None,
                  global_prototypes: Dict[int, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        FedPOP聚合 + FedDisAF多头注意力
        """
        if not client_updates:
            return {}

        self._update_population(client_updates)

        if self.generation % self.evolution_interval == 0 and self.generation > 0:
            self._evolve_population()

        attention_weights = None
        if client_features and self.config.get('feddisaf_enabled', True) and self.attention_aggregator:
            attention_weights = self.attention_aggregator.compute_attention_weights(client_features)

        selection_probs = self._compute_selection_probabilities(client_updates, attention_weights)

        global_weights = self._weighted_aggregate(client_updates, selection_probs)

        self.plugin.update_library(client_updates, global_prototypes)

        self.generation += 1
        return global_weights

    def _update_population(self, client_updates: List[Dict[str, Any]]):
        """更新种群 - 将客户端更新加入种群"""
        for idx, update in enumerate(client_updates):
            client_id = update.get('client_id', idx)
            fitness = self._compute_fitness(update)

            self.client_fitness_history.setdefault(client_id, []).append(fitness)
            if len(self.client_fitness_history[client_id]) > self.performance_history_length:
                self.client_fitness_history[client_id].pop(0)

            smoothed_fitness = self._get_smoothed_fitness(client_id, fitness)

            self.population.append({
                'weights': update['weights'],
                'fitness': smoothed_fitness,
                'loss': update.get('loss', 1.0),
                'num_samples': update.get('num_samples', 1),
                'client_id': client_id
            })

        if len(self.population) > self.population_size:
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            self.population = self.population[:self.population_size]

    def _compute_fitness(self, update: Dict[str, Any]) -> float:
        """计算适应度 - 基于损失和样本数量"""
        loss = update.get('loss', 1.0)
        num_samples = update.get('num_samples', 1)

        fitness = 1.0 / (loss + 1e-8)

        fitness *= np.log(num_samples + 1)

        return fitness

    def _get_smoothed_fitness(self, client_id: int, current_fitness: float) -> float:
        """获取平滑后的适应度 (考虑历史)"""
        history = self.client_fitness_history.get(client_id, [])
        if len(history) > 0:
            alpha = self.fitness_decay
            historical_fitness = np.mean(history[-5:]) if len(history) >= 5 else np.mean(history)
            smoothed = alpha * historical_fitness + (1 - alpha) * current_fitness
        else:
            smoothed = current_fitness

        return smoothed

    def _evolve_population(self):
        """执行种群进化 - 选择、交叉、变异"""
        if len(self.population) < 2:
            return

        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        self.elite_population = sorted_pop[:self.elite_size]

        new_population = []

        new_population.extend(self.elite_population)

        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection(sorted_pop)
            parent2 = self._tournament_selection(sorted_pop)

            if np.random.random() < self.crossover_rate:
                child_weights = self._crossover(parent1['weights'], parent2['weights'])
            else:
                child_weights = copy.deepcopy(parent1['weights'])

            child_weights = self._mutate(child_weights)

            child_fitness = (parent1['fitness'] + parent2['fitness']) / 2

            new_population.append({
                'weights': child_weights,
                'fitness': child_fitness,
                'client_id': -1  # 虚拟客户端ID
            })

        self.population = new_population

    def _tournament_selection(self, sorted_pop: List) -> Dict:
        """锦标赛选择"""
        tournament_size = min(3, len(sorted_pop))
        candidates = np.random.choice(len(sorted_pop), tournament_size, replace=False)
        best_idx = max(candidates, key=lambda i: sorted_pop[i]['fitness'])
        return sorted_pop[best_idx]

    def _crossover(self, weights1: Dict, weights2: Dict) -> Dict:
        """交叉操作 - 对模型权重进行均匀交叉"""
        child_weights = {}

        for key in weights1.keys():
            if key in weights2:
                mask = torch.rand_like(weights1[key], dtype=torch.float32) < self.crossover_rate
                child_weights[key] = torch.where(mask, weights1[key], weights2[key])
            else:
                child_weights[key] = weights1[key].clone()

        return child_weights

    def _mutate(self, weights: Dict) -> Dict:
        """变异操作 - 对模型权重添加高斯噪声"""
        mutated_weights = {}

        for key, param in weights.items():
            if np.random.random() < self.mutation_rate and param.is_floating_point():
                noise = torch.randn_like(param) * self.exploration_factor
                mutated_weights[key] = param + noise
            else:
                mutated_weights[key] = param.clone()

        return mutated_weights

    def _compute_selection_probabilities(self, client_updates: List[Dict[str, Any]],
                                          attention_weights: Optional[torch.Tensor]) -> List[float]:
        """计算选择概率 - 结合种群适应度和注意力权重"""
        num_clients = len(client_updates)
        probs = []

        for idx, update in enumerate(client_updates):
            client_id = update.get('client_id', idx)

            population_fitness = self._get_population_fitness(client_id, update['weights'])

            base_prob = population_fitness

            loss = update.get('loss', 1.0)
            loss_weight = 1.0 / (loss + 0.1)
            base_prob *= loss_weight

            probs.append(base_prob)

        probs = np.array(probs)
        probs = probs / probs.sum()

        probs = np.maximum(probs, self.min_selection_prob)
        probs = probs / probs.sum()

        if attention_weights is not None:
            attn = attention_weights.cpu().numpy()
            probs = self.sample_weight * probs + (1 - self.sample_weight) * attn
            probs = probs / probs.sum()

        return probs.tolist()

    def _get_population_fitness(self, client_id: int, weights: Dict) -> float:
        """从种群中获取客户端的适应度"""
        for individual in self.population:
            if individual.get('client_id') == client_id:
                return individual['fitness']

        history = self.client_fitness_history.get(client_id, [])
        if len(history) > 0:
            return np.mean(history[-3:])

        return 0.5

    def _weighted_aggregate(self, client_updates: List[Dict[str, Any]],
                            probabilities: List[float]) -> Dict[str, torch.Tensor]:
        """基于概率的加权聚合"""
        if not client_updates:
            return {}

        global_weights = {}
        weights_list = [update['weights'] for update in client_updates]

        for key in weights_list[0].keys():
            global_weights[key] = torch.zeros_like(weights_list[0][key], dtype=torch.float32)

        for idx, w in enumerate(weights_list):
            prob = probabilities[idx]
            for key in global_weights.keys():
                global_weights[key] += prob * w[key].float()

        return global_weights

    def local_update(self, model: nn.Module, train_loader: DataLoader,
                     client_id: int = 0,
                     global_prototypes: Dict[int, torch.Tensor] = None) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        """
        FedPOP本地训练 - 带探索噪声
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

                if self.exploration_factor > 0 and np.random.random() < self.exploration_factor:
                    for param in model.parameters():
                        if param.grad is not None:
                            noise = torch.randn_like(param.grad) * 0.01
                            param.grad += noise

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
