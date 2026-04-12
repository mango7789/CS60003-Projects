"""
训练模块
包含训练循环、模型保存等功能
"""

import numpy as np
import os
import json
from datetime import datetime
from model import ThreeLayerMLP
from data_loader import get_batches
from optimizers import SGDOptimizer, LearningRateScheduler
from losses import CrossEntropyLoss


class Trainer:
    """训练器类"""

    def __init__(self, model, optimizer, lr_scheduler, save_dir='./checkpoints'):
        """
        初始化训练器

        Args:
            model: 模型实例
            optimizer: 优化器实例
            lr_scheduler: 学习率调度器
            save_dir: 模型保存目录
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # 最优模型记录
        self.best_val_acc = 0.0
        self.best_epoch = 0

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, X_train, y_train, batch_size):
        """
        训练一个 epoch

        Args:
            X_train: 训练数据
            y_train: 训练标签 (类别索引)
            batch_size: 批次大小

        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 获取批次数据
        batches = list(get_batches(X_train, y_train, batch_size, shuffle=True))

        for batch_X, batch_y in batches:
            # One-hot 编码
            batch_size = batch_X.shape[0]
            y_onehot = np.zeros((batch_size, self.model.output_size))
            y_onehot[np.arange(batch_size), batch_y] = 1

            # 前向传播
            output = self.model.forward(batch_X)

            # 计算损失
            loss, ce_loss, reg_loss = self.model.compute_loss(output, y_onehot)

            # 反向传播
            grads = self.model.backward(y_onehot)

            # 参数更新
            self.optimizer.step(self.model.params, grads, self.model.param_names)

            # 统计
            total_loss += loss * batch_size
            predictions = np.argmax(output, axis=1)
            total_correct += np.sum(predictions == batch_y)
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def validate(self, X_val, y_val, batch_size=256):
        """
        验证模型

        Args:
            X_val: 验证数据
            y_val: 验证标签
            batch_size: 批次大小

        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in get_batches(X_val, y_val, batch_size, shuffle=False):
            batch_size = batch_X.shape[0]
            y_onehot = np.zeros((batch_size, self.model.output_size))
            y_onehot[np.arange(batch_size), batch_y] = 1

            # 前向传播
            output = self.model.forward(batch_X)

            # 计算损失
            loss, _, _ = self.model.compute_loss(output, y_onehot)

            # 统计
            total_loss += loss * batch_size
            predictions = np.argmax(output, axis=1)
            total_correct += np.sum(predictions == batch_y)
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=64, verbose=True, early_stopping_patience=None):
        """
        完整训练流程

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
            early_stopping_patience: Early stopping 耐心值 (None 表示不使用)

        Returns:
            history: 训练历史
        """
        print(f"开始训练...")
        print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
        print(f"批次大小: {batch_size}, 训练轮数: {epochs}")
        print(f"模型参数量: {self.model.count_parameters():,}")
        if early_stopping_patience:
            print(f"Early Stopping: patience={early_stopping_patience}")
        print("-" * 60)

        no_improve_count = 0

        for epoch in range(epochs):
            # 训练一个 epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, batch_size)

            # 验证
            val_loss, val_acc = self.validate(X_val, y_val)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.learning_rate)

            # 保存最优模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.model.save_weights(os.path.join(self.save_dir, 'best_model.pkl'))
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 学习率衰减
            self.lr_scheduler.step()

            # 打印进度
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"LR: {self.optimizer.learning_rate:.6f}"
                      f"{' *' if val_acc == self.best_val_acc else ''}")

            # Early stopping
            if early_stopping_patience and no_improve_count >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                break

        print("-" * 60)
        print(f"训练完成! 最优验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch + 1})")

        return self.history

    def save_history(self, filepath):
        """保存训练历史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"训练历史已保存到 {filepath}")


def train_model(config, X_train, y_train, X_val, y_val, save_dir):
    """
    根据配置训练模型

    Args:
        config: 配置字典
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        save_dir: 保存目录

    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    # 创建模型
    model = ThreeLayerMLP(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        activation=config['activation'],
        weight_decay=config['weight_decay']
    )

    # 创建优化器
    optimizer = SGDOptimizer(
        learning_rate=config['learning_rate'],
        momentum=config.get('momentum', 0.9),
        weight_decay=config['weight_decay']
    )

    # 创建学习率调度器
    lr_scheduler = LearningRateScheduler(
        optimizer,
        decay_type=config.get('lr_decay_type', 'step'),
        decay_rate=config.get('lr_decay_rate', 0.1),
        decay_steps=config.get('lr_decay_steps', 30),
        min_lr=1e-6
    )

    # 创建训练器
    trainer = Trainer(model, optimizer, lr_scheduler, save_dir)

    # 训练
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=True
    )

    # 保存训练历史
    trainer.save_history(os.path.join(save_dir, 'history.json'))

    return model, history


if __name__ == "__main__":
    # 测试训练流程
    from data_loader import DataLoader

    config = {
        'input_size': 12288,
        'hidden_size': 256,
        'output_size': 10,
        'activation': 'relu',
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'epochs': 5,
        'batch_size': 64,
        'lr_decay_type': 'step',
        'lr_decay_rate': 0.5,
        'lr_decay_steps': 2
    }

    # 加载数据
    loader = DataLoader('../../EuroSAT_RGB')
    images, labels = loader.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(images, labels)

    # 训练
    model, history = train_model(config, X_train, y_train, X_val, y_val, './test_checkpoints')
