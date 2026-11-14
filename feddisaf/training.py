"""One independent combination run with per-round outputs."""
import json
import os
from typing import Tuple

import numpy as np
import torch

from .logger import ExperimentLogger
from .combinations import build_combination
from .models import get_client_model


def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
             device: str) -> Tuple[float, float, float]:
    """
    评估模型性能

    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 计算设备

    Returns:
        accuracy: 准确率 (%)
        avg_loss: 平均交叉熵损失
        recall: 召回率
    """
    model.eval()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs, _, _ = model(data)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)

    from sklearn.metrics import recall_score
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    return accuracy, avg_loss, recall

def run_training(config, client_loaders, test_loader, base_log_dir, partition=None):
    config['algorithm'] = config.get('base_algorithm', 'FedAvg') + '+FedDisAF'
    with ExperimentLogger(base_log_dir, config) as logger:
        return _run_training(config, client_loaders, test_loader, logger, partition)


def _run_training(config, client_loaders, test_loader, logger, partition=None):
    algorithm_name = config.get('base_algorithm', 'FedAvg') + '+FedDisAF'
    config['algorithm'] = algorithm_name
    logger.log_config(config)

    print(f"\n{'='*60}")
    print(f"Running {algorithm_name}")
    print(f"Experiment Dir: {logger.get_experiment_path()}")
    print(f"{'='*60}")


    if partition is not None:
        with open(os.path.join(logger.experiment_dir, 'partition.json'), 'w', encoding='utf-8') as stream:
            json.dump(partition, stream, indent=2)
    algorithm = build_combination(config)
    global_model = get_client_model(config)

    num_classes = config.get('num_classes', 10)
    global_prototypes = {c: None for c in range(num_classes)}

    best_accuracy = 0
    best_round = 0
    best_recall = 0
    best_loss = 0

    all_accuracies = []
    all_losses = []
    all_recalls = []
    all_train_losses = []

    for round_num in range(1, config.get('comm_round', 200) + 1):
        num_selected = min(config.get('client_num_per_round', 5), len(client_loaders))
        selected_indices = np.random.choice(len(client_loaders), num_selected, replace=False)

        client_updates = []
        client_features = []
        total_train_loss = 0
        total_proto_loss = 0.0

        for idx in selected_indices:
            model = get_client_model(config)
            model.load_state_dict(global_model.state_dict())

            update, features = algorithm.local_update(
                model, client_loaders[idx], idx,
                global_prototypes=global_prototypes)

            client_updates.append(update)
            client_features.append(features if features is not None else None)
            total_train_loss += update.get('loss', 0)
            total_proto_loss += update.get('proto_loss', 0.0)

        avg_train_loss = total_train_loss / len(selected_indices) if len(selected_indices) > 0 else 0
        avg_proto_loss = total_proto_loss / len(selected_indices) if len(selected_indices) > 0 else 0
        all_train_losses.append(avg_train_loss)

        global_weights = algorithm.aggregate(
            client_updates, client_features,
            global_prototypes=global_prototypes)

        global_model.load_state_dict(global_weights)

        if round_num % 10 == 0 or round_num == 1:
            valid_protos = sum(1 for v in global_prototypes.values() if v is not None)
            proto_norm = sum(
                v.norm().item() for v in global_prototypes.values() if v is not None
            ) / max(valid_protos, 1)
            logger.logger.info(
                f"[Proto-Check] Round {round_num}: "
                f"valid={valid_protos}/{num_classes}, "
                f"avg_norm={proto_norm:.4f}, "
                f"proto_loss={avg_proto_loss:.4f}"
            )

        test_accuracy, test_loss, test_recall = evaluate(
            global_model, test_loader, config['device']
        )

        all_accuracies.append(test_accuracy)
        all_losses.append(test_loss)
        all_recalls.append(test_recall)

        logger.log_round(round_num, avg_train_loss, test_accuracy, test_loss, test_recall)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_round = round_num
            best_recall = test_recall
            best_loss = test_loss
            if config.get('save_model', True):
                model_path = os.path.join(logger.experiment_dir, 'best_model.pth')
                torch.save({
                    'round': round_num,
                    'model_state_dict': global_model.state_dict(),
                    'accuracy': test_accuracy,
                    'loss': test_loss,
                    'recall': test_recall
                }, model_path)

    last_n = min(config.get('summary_last_rounds', 5), len(all_accuracies))
    recent_acc = all_accuracies[-last_n:] if last_n > 0 else all_accuracies
    recent_loss = all_losses[-last_n:] if last_n > 0 else all_losses
    recent_recall = all_recalls[-last_n:] if last_n > 0 else all_recalls

    mean_accuracy = np.mean(recent_acc) if recent_acc else 0
    std_accuracy = np.std(recent_acc) if recent_acc else 0
    mean_loss = np.mean(recent_loss) if recent_loss else 0
    mean_recall = np.mean(recent_recall) if recent_recall else 0
    std_recall = np.std(recent_recall) if recent_recall else 0

    results = {
        'algorithm': algorithm_name,
        'dataset': config.get('dataset'),
        'partition_method': config.get('partition_method'),
        'num_clients': config.get('client_num_in_total'),
        'num_selected': config.get('client_num_per_round'),
        'local_epochs': config.get('local_epochs'),
        'batch_size': config.get('batch_size'),
        'learning_rate': config.get('lr'),
        'dp_noise': config.get('dp_noise_scale', 0),
        'alpha': config.get('partition_alpha'),
        'num_rounds': config.get('comm_round'),
        'mean_accuracy': mean_accuracy,
        'within_run_round_std_accuracy': std_accuracy,
        'mean_loss': mean_loss,
        'final_loss': recent_loss[-1] if recent_loss else 0,
        'mean_recall': mean_recall,
        'within_run_round_std_recall': std_recall,
        'final_recall': recent_recall[-1] if recent_recall else 0,
        'best_accuracy': best_accuracy,
        'best_recall': best_recall,
        'best_loss': best_loss,
        'best_round': best_round,
        'experiment_dir': logger.get_experiment_path()
    }

    logger.log_final_results(results)

    if config.get('save_model', True):
        final_model_path = os.path.join(logger.experiment_dir, 'final_model.pth')
        torch.save({
            'round': config.get('comm_round', 200),
            'model_state_dict': global_model.state_dict(),
            'accuracy': all_accuracies[-1] if all_accuracies else 0.0,
            'loss': all_losses[-1] if all_losses else 0.0,
            'recall': all_recalls[-1] if all_recalls else 0.0
        }, final_model_path)

    summary_file = os.path.join(logger.experiment_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    return results
