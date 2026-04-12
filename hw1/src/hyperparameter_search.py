"""
超参数搜索模块
支持网格搜索和随机搜索
"""

import numpy as np
import os
import json
import itertools
from datetime import datetime
from model import ThreeLayerMLP
from trainer import Trainer
from optimizers import SGDOptimizer, LearningRateScheduler


class HyperparameterSearch:
    """超参数搜索"""

    def __init__(self, X_train, y_train, X_val, y_val, save_dir='./search_results'):
        """
        初始化超参数搜索

        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            save_dir: 结果保存目录
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.save_dir = save_dir
        self.results = []

        os.makedirs(save_dir, exist_ok=True)

    def grid_search(self, param_grid, epochs=50, batch_size=64):
        """
        网格搜索

        Args:
            param_grid: 参数网格字典
            epochs: 训练轮数
            batch_size: 批次大小

        Returns:
            best_params: 最优参数
            best_acc: 最优准确率
        """
        print("开始网格搜索...")
        print(f"参数网格: {param_grid}")

        # 生成所有参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        print(f"总共 {len(combinations)} 种参数组合")
        print("-" * 60)

        best_acc = 0.0
        best_params = None
        best_idx = 0

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\n[{idx + 1}/{len(combinations)}] 参数: {params}")

            # 训练模型
            val_acc = self._train_and_evaluate(params, epochs, batch_size, idx)

            # 记录结果
            result = {
                'params': params,
                'val_acc': val_acc,
                'idx': idx
            }
            self.results.append(result)

            # 更新最优
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params
                best_idx = idx

            print(f"验证准确率: {val_acc:.4f} | 当前最优: {best_acc:.4f}")

        print("-" * 60)
        print(f"网格搜索完成!")
        print(f"最优参数: {best_params}")
        print(f"最优准确率: {best_acc:.4f}")

        # 保存结果
        self._save_results()

        return best_params, best_acc

    def random_search(self, param_distributions, n_trials=20, epochs=50, batch_size=64):
        """
        随机搜索

        Args:
            param_distributions: 参数分布字典
            n_trials: 搜索次数
            epochs: 训练轮数
            batch_size: 批次大小

        Returns:
            best_params: 最优参数
            best_acc: 最优准确率
        """
        print("开始随机搜索...")
        print(f"搜索次数: {n_trials}")
        print("-" * 60)

        best_acc = 0.0
        best_params = None

        for trial in range(n_trials):
            # 随机采样参数
            params = {}
            for key, value in param_distributions.items():
                if isinstance(value, list):
                    params[key] = np.random.choice(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    params[key] = np.random.uniform(value[0], value[1])
                    if key in ['hidden_size']:
                        params[key] = int(params[key])
                else:
                    params[key] = value

            print(f"\n[{trial + 1}/{n_trials}] 参数: {params}")

            # 训练模型
            val_acc = self._train_and_evaluate(params, epochs, batch_size, trial)

            # 记录结果
            result = {
                'params': params,
                'val_acc': val_acc,
                'trial': trial
            }
            self.results.append(result)

            # 更新最优
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params

            print(f"验证准确率: {val_acc:.4f} | 当前最优: {best_acc:.4f}")

        print("-" * 60)
        print(f"随机搜索完成!")
        print(f"最优参数: {best_params}")
        print(f"最优准确率: {best_acc:.4f}")

        # 保存结果
        self._save_results()

        return best_params, best_acc

    def _train_and_evaluate(self, params, epochs, batch_size, idx):
        """
        训练并评估模型

        Args:
            params: 参数字典
            epochs: 训练轮数
            batch_size: 批次大小
            idx: 实验索引

        Returns:
            val_acc: 验证准确率
        """
        # 创建模型
        model = ThreeLayerMLP(
            input_size=12288,
            hidden_size=params.get('hidden_size', 256),
            output_size=10,
            activation=params.get('activation', 'relu'),
            weight_decay=params.get('weight_decay', 0.0001)
        )

        # 创建优化器
        optimizer = SGDOptimizer(
            learning_rate=params.get('learning_rate', 0.01),
            momentum=params.get('momentum', 0.9),
            weight_decay=params.get('weight_decay', 0.0001)
        )

        # 创建学习率调度器
        lr_scheduler = LearningRateScheduler(
            optimizer,
            decay_type='step',
            decay_rate=0.5,
            decay_steps=20,
            min_lr=1e-6
        )

        # 创建保存目录
        exp_dir = os.path.join(self.save_dir, f'exp_{idx}')
        os.makedirs(exp_dir, exist_ok=True)

        # 训练
        trainer = Trainer(model, optimizer, lr_scheduler, exp_dir)
        history = trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )

        # 返回最优验证准确率
        return max(history['val_acc'])

    def _save_results(self):
        """保存搜索结果"""
        results_file = os.path.join(self.save_dir, 'search_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n搜索结果已保存到 {results_file}")


if __name__ == "__main__":
    # 测试超参数搜索
    from data_loader import DataLoader

    # 加载数据
    loader = DataLoader('../../EuroSAT_RGB')
    images, labels = loader.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(images, labels)

    # 定义参数网格
    param_grid = {
        'hidden_size': [128, 256],
        'learning_rate': [0.01, 0.001],
        'activation': ['relu', 'tanh'],
        'weight_decay': [0.0001, 0.001]
    }

    # 网格搜索
    searcher = HyperparameterSearch(X_train, y_train, X_val, y_val, './search_results')
    best_params, best_acc = searcher.grid_search(param_grid, epochs=10, batch_size=64)
