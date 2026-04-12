"""
优化器模块
包含 SGD 优化器和学习率衰减策略
"""

import numpy as np


class SGDOptimizer:
    """随机梯度下降优化器"""

    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        """
        Args:
            learning_rate: 学习率
            momentum: 动量系数
            weight_decay: L2 正则化系数
        """
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}  # 存储动量速度

    def step(self, params, grads, param_names):
        """
        执行一步参数更新

        Args:
            params: 参数字典 {name: value}
            grads: 梯度字典 {name: value}
            param_names: 参数名称列表
        """
        for name in param_names:
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(params[name])

            # 添加 L2 正则化梯度
            grad = grads[name]
            if 'W' in name:  # 只对权重进行正则化
                grad = grad + self.weight_decay * params[name]

            # 动量更新
            self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * grad
            params[name] = params[name] + self.velocity[name]

    def update_learning_rate(self, new_lr):
        """更新学习率"""
        self.learning_rate = new_lr


class LearningRateScheduler:
    """学习率调度器"""

    def __init__(self, optimizer, decay_type='step', decay_rate=0.1, decay_steps=30,
                 min_lr=1e-6):
        """
        Args:
            optimizer: 优化器实例
            decay_type: 衰减类型 ('step', 'exponential', 'cosine', 'linear')
            decay_rate: 衰减率
            decay_steps: 衰减步数 (对于 step 衰减)
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.learning_rate
        self.current_step = 0

    def step(self, epoch=None):
        """更新学习率"""
        self.current_step += 1

        if self.decay_type == 'step':
            # 阶梯衰减
            new_lr = self.initial_lr * (self.decay_rate ** (self.current_step // self.decay_steps))
        elif self.decay_type == 'exponential':
            # 指数衰减
            new_lr = self.initial_lr * (self.decay_rate ** self.current_step)
        elif self.decay_type == 'cosine':
            # 余弦退火
            new_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                     (1 + np.cos(np.pi * self.current_step / self.decay_steps))
        elif self.decay_type == 'linear':
            # 线性衰减
            new_lr = self.initial_lr - (self.initial_lr - self.min_lr) * \
                     (self.current_step / self.decay_steps)
        else:
            raise ValueError(f"不支持的衰减类型: {self.decay_type}")

        new_lr = max(new_lr, self.min_lr)
        self.optimizer.update_learning_rate(new_lr)
        return new_lr

    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.learning_rate


if __name__ == "__main__":
    # 测试优化器
    params = {'W1': np.random.randn(3, 4), 'b1': np.zeros(4)}
    grads = {'W1': np.random.randn(3, 4) * 0.1, 'b1': np.random.randn(4) * 0.1}

    optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = LearningRateScheduler(optimizer, decay_type='step', decay_rate=0.5, decay_steps=10)

    print("初始参数:")
    print(params['W1'][:2, :2])

    for step in range(25):
        optimizer.step(params, grads, ['W1', 'b1'])
        lr = scheduler.step()
        if step % 5 == 0:
            print(f"Step {step}, LR: {lr:.6f}")
