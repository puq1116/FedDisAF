# utils/logger.py
import os
import json
import logging
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


class ExperimentLogger:
    """实验日志记录器 - 每个实验独立文件夹"""

    def __init__(self, base_log_dir: str, config: Dict[str, Any]):
        """
        初始化日志记录器

        Args:
            base_log_dir: 基础日志目录
            config: 实验配置
        """
        self.config = config
        self.base_log_dir = base_log_dir

        # 生成实验文件夹名称
        self.experiment_name = self._generate_experiment_name(config)

        # 创建实验文件夹
        self.experiment_dir = os.path.join(base_log_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 定义文件路径
        self.log_file = os.path.join(self.experiment_dir, 'training.log')
        self.results_file = os.path.join(self.experiment_dir, 'results.json')
        self.metrics_file = os.path.join(self.experiment_dir, 'metrics.csv')
        self.config_file = os.path.join(self.experiment_dir, 'config.json')

        # 设置日志
        self._setup_logging()

        # 保存配置
        self._save_config()

        # 存储训练历史
        self.history = {
            'rounds': [],
            'train_loss': [],
            'test_accuracy': [],
            'test_loss': [],
            'test_recall': []
        }

        # 写入CSV头
        self._init_csv()

        self.logger.info(f"Experiment directory created: {self.experiment_dir}")
        self.logger.info(f"Experiment name: {self.experiment_name}")

    def _generate_experiment_name(self, config: Dict[str, Any]) -> str:
        """
        生成实验文件夹名称
        格式: 算法名称_数据集_客户端数_选中数_本地轮数_差分隐私噪声尺度_批大小_学习率_alpha_时间
        """
        # 获取配置参数
        algorithm = config.get('algorithm', 'FedDisAF')
        dataset = config.get('dataset', 'PlantVillage')
        num_clients = config.get('client_num_in_total', 10)
        num_selected = config.get('client_num_per_round', 5)
        local_epochs = config.get('local_epochs', 1)

        # 差分隐私噪声尺度
        dp_noise = config.get('dp_noise_scale', config.get('feddisaf_dp_noise', 0.15))
        dp_str = f"dp{dp_noise}" if dp_noise > 0 else "no_dp"

        batch_size = config.get('batch_size', 32)
        learning_rate = config.get('lr', config.get('learning_rate', 0.001))
        alpha = config.get('partition_alpha', config.get('feddisaf_alpha', 0.75))

        # 时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 生成文件夹名
        experiment_name = (
            f"{algorithm}_{dataset}_"
            f"{num_clients}c_{num_selected}s_{local_epochs}e_"
            f"{dp_str}_{batch_size}b_{learning_rate}lr_{alpha}a_"
            f"seed{config.get('seed', 5)}_{timestamp}"
        )

        # 替换特殊字符
        experiment_name = experiment_name.replace('.', '_')

        return experiment_name

    def _setup_logging(self):
        """设置日志格式"""
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # 配置文件handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        # 配置控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(message)s'))

        # 获取logger
        self.logger = logging.getLogger(__name__ + '.' + self.experiment_name)
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
            handler.close()
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def close(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _save_config(self):
        """保存配置到JSON文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, default=str)

    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'train_loss', 'test_accuracy', 'test_loss', 'test_recall'
            ])

    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        self.logger.info("=" * 70)
        self.logger.info("Experiment Configuration")
        self.logger.info("=" * 70)
        self.logger.info(f"Experiment Name: {self.experiment_name}")
        self.logger.info(f"Experiment Dir: {self.experiment_dir}")
        self.logger.info("-" * 70)

        # 分类显示配置
        sections = {
            'Method': ['algorithm', 'proto_loss_weight', 'feddisaf_proto_temperature'],
            'Dataset': ['dataset', 'data_dir', 'num_classes', 'partition_method', 'partition_alpha'],
            'Training': ['comm_round', 'local_epochs', 'batch_size', 'lr', 'wd'],
            'Clients': ['client_num_in_total', 'client_num_per_round'],
            'Perturbation': ['dp_noise_scale'],
            'Device': ['device', 'seed']
        }

        for section, keys in sections.items():
            self.logger.info(f"\n[{section}]")
            for key in keys:
                if key in config:
                    value = config[key]
                    if isinstance(value, float):
                        self.logger.info(f"  {key}: {value:.6f}")
                    else:
                        self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 70)

    def log_round(self, round_num: int, train_loss: float,
                  test_accuracy: float, test_loss: float, test_recall: float):
        """记录每轮结果"""
        # 保存到历史
        self.history['rounds'].append(round_num)
        self.history['train_loss'].append(train_loss)
        self.history['test_accuracy'].append(test_accuracy)
        self.history['test_loss'].append(test_loss)
        self.history['test_recall'].append(test_recall)

        # 记录到日志
        self.logger.info(f"Round {round_num:4d}: "
                        f"Train Loss={train_loss:.6f}, "
                        f"Test Acc={test_accuracy:6.2f}%, "
                        f"Test Loss={test_loss:.6f}, "
                        f"Test Recall={test_recall:.6f}")

        # 实时保存到CSV
        self._append_to_csv(round_num, train_loss, test_accuracy, test_loss, test_recall)

    def _append_to_csv(self, round_num: int, train_loss: float,
                       test_accuracy: float, test_loss: float, test_recall: float):
        """追加数据到CSV文件"""
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                f"{train_loss:.6f}",
                f"{test_accuracy:.2f}",
                f"{test_loss:.6f}",
                f"{test_recall:.6f}"
            ])

    def log_final_results(self, results: Dict[str, Any]):
        """记录最终结果"""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("FINAL RESULTS")
        self.logger.info("=" * 70)

        # 主要结果
        main_results = ['algorithm', 'dataset', 'partition_method', 'final_accuracy',
                       'final_loss', 'final_recall', 'mean_accuracy', 'within_run_round_std_accuracy',
                       'mean_recall', 'within_run_round_std_recall', 'best_accuracy', 'best_round']

        for key in main_results:
            if key in results:
                value = results[key]
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.6f}")
                else:
                    self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 70)

        # 保存最终结果到JSON
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        # 保存历史数据
        history_file = os.path.join(self.experiment_dir, 'history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)

    def log_error(self, error_msg: str):
        """记录错误"""
        self.logger.error(error_msg)

    def log_info(self, info_msg: str):
        """记录信息"""
        self.logger.info(info_msg)

    def get_best_results(self) -> Dict[str, float]:
        """获取最佳结果"""
        if not self.history['test_accuracy']:
            return {}

        best_idx = np.argmax(self.history['test_accuracy'])
        return {
            'best_round': self.history['rounds'][best_idx],
            'best_accuracy': self.history['test_accuracy'][best_idx],
            'best_recall': self.history['test_recall'][best_idx],
            'best_loss': self.history['test_loss'][best_idx]
        }

    def get_final_stats(self, last_n: int = 5) -> Dict[str, float]:
        """获取最后N轮的统计"""
        if len(self.history['test_accuracy']) < last_n:
            last_n = len(self.history['test_accuracy'])

        if last_n == 0:
            return {
                'mean_accuracy': 0, 'std_accuracy': 0,
                'mean_loss': 0, 'std_loss': 0,
                'mean_recall': 0, 'std_recall': 0,
                'final_accuracy': 0, 'final_loss': 0, 'final_recall': 0
            }

        recent_acc = self.history['test_accuracy'][-last_n:]
        recent_loss = self.history['test_loss'][-last_n:]
        recent_recall = self.history['test_recall'][-last_n:]

        return {
            'mean_accuracy': np.mean(recent_acc),
            'std_accuracy': np.std(recent_acc),
            'mean_loss': np.mean(recent_loss),
            'std_loss': np.std(recent_loss),
            'mean_recall': np.mean(recent_recall),
            'std_recall': np.std(recent_recall),
            'final_accuracy': recent_acc[-1] if recent_acc else 0,
            'final_loss': recent_loss[-1] if recent_loss else 0,
            'final_recall': recent_recall[-1] if recent_recall else 0
        }

    def get_experiment_path(self) -> str:
        """获取实验文件夹路径"""
        return self.experiment_dir
