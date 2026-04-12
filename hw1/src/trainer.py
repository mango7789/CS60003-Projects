"""
训练模块
包含训练循环、模型保存等功能
"""

import numpy as np
import os
import json
import logging
from tqdm import tqdm
from model import ThreeLayerMLP
from data_loader import get_batches
from optimizers import SGDOptimizer, LearningRateScheduler
from losses import CrossEntropyLoss


class Trainer:
    """训练器类"""

    def __init__(self, model, optimizer, lr_scheduler, save_dir='./checkpoints', logger=None):
        """
        初始化训练器

        Args:
            model: 模型实例
            optimizer: 优化器实例
            lr_scheduler: 学习率调度器
            save_dir: 模型保存目录
            logger: 日志记录器
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir
        self.logger = logger or logging.getLogger(__name__)

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

    def train_epoch(self, X_train, y_train, batch_size, show_progress=False, epoch_num=0, total_epochs=0):
        """
        训练一个 epoch

        Args:
            X_train: 训练数据
            y_train: 训练标签 (类别索引)
            batch_size: 批次大小
            show_progress: 是否显示进度条
            epoch_num: 当前 epoch 编号
            total_epochs: 总 epoch 数

        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 获取批次数据
        batches = list(get_batches(X_train, y_train, batch_size, shuffle=True))

        if show_progress:
            pbar = tqdm(batches, desc=f'Epoch {epoch_num}/{total_epochs}',
                        leave=False, ncols=100)
        else:
            pbar = batches

        for batch_X, batch_y in pbar:
            # One-hot 编码
            actual_batch_size = batch_X.shape[0]
            y_onehot = np.zeros((actual_batch_size, self.model.output_size))
            y_onehot[np.arange(actual_batch_size), batch_y] = 1

            # 前向传播
            output = self.model.forward(batch_X)

            # 计算损失
            loss, ce_loss, reg_loss = self.model.compute_loss(output, y_onehot)

            # 反向传播
            grads = self.model.backward(y_onehot)

            # 参数更新
            self.optimizer.step(self.model.params, grads, self.model.param_names)

            # 统计
            total_loss += loss * actual_batch_size
            predictions = np.argmax(output, axis=1)
            total_correct += np.sum(predictions == batch_y)
            total_samples += actual_batch_size

            # 更新进度条信息
            if show_progress:
                current_acc = total_correct / total_samples
                current_loss = total_loss / total_samples
                pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})

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
            actual_batch_size = batch_X.shape[0]
            y_onehot = np.zeros((actual_batch_size, self.model.output_size))
            y_onehot[np.arange(actual_batch_size), batch_y] = 1

            # 前向传播
            output = self.model.forward(batch_X)

            # 计算损失
            loss, _, _ = self.model.compute_loss(output, y_onehot)

            # 统计
            total_loss += loss * actual_batch_size
            predictions = np.argmax(output, axis=1)
            total_correct += np.sum(predictions == batch_y)
            total_samples += actual_batch_size

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
            verbose: 是否显示进度条
            early_stopping_patience: Early stopping 耐心值 (None 表示不使用)

        Returns:
            history: 训练历史
        """
        self.logger.info(f"开始训练...")
        self.logger.info(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")
        self.logger.info(f"批次大小: {batch_size}, 训练轮数: {epochs}")
        self.logger.info(f"模型参数量: {self.model.count_parameters():,}")
        if early_stopping_patience:
            self.logger.info(f"Early Stopping: patience={early_stopping_patience}")
        self.logger.info("-" * 60)

        no_improve_count = 0

        for epoch in range(epochs):
            # 训练一个 epoch (带进度条)
            train_loss, train_acc = self.train_epoch(
                X_train, y_train, batch_size,
                show_progress=verbose, epoch_num=epoch + 1, total_epochs=epochs
            )

            # 验证
            val_loss, val_acc = self.validate(X_val, y_val)

            # 记录历史
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc))
            self.history['learning_rates'].append(float(self.optimizer.learning_rate))

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

            # 打印本 epoch 结果
            msg = (f"Epoch {epoch + 1:3d}/{epochs} | "
                   f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                   f"LR: {self.optimizer.learning_rate:.6f}"
                   f"{' *' if val_acc == self.best_val_acc else ''}")
            self.logger.info(msg)

            # Early stopping
            if early_stopping_patience and no_improve_count >= early_stopping_patience:
                self.logger.info(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                break

        self.logger.info("-" * 60)
        self.logger.info(f"训练完成! 最优验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch + 1})")

        return self.history

    def save_history(self, filepath):
        """保存训练历史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"训练历史已保存到 {filepath}")


def train_model(config, X_train, y_train, X_val, y_val, save_dir, logger=None):
    """
    根据配置训练模型

    Args:
        config: 配置字典
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        save_dir: 保存目录
        logger: 日志记录器

    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    if logger is None:
        logger = logging.getLogger(__name__)

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
    trainer = Trainer(model, optimizer, lr_scheduler, save_dir, logger)

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
