"""
损失函数模块
包含交叉熵损失函数
"""

import numpy as np


class CrossEntropyLoss:
    """交叉熵损失函数"""

    def __init__(self):
        self.cache = None

    def forward(self, predictions, targets, epsilon=1e-15):
        """
        计算交叉熵损失

        Args:
            predictions: 模型预测概率 (softmax输出), shape (N, C)
            targets: 真实标签 (one-hot编码), shape (N, C)
            epsilon: 防止 log(0) 的极小值

        Returns:
            loss: 标量损失值
        """
        # 裁剪预测值防止数值问题
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # 计算交叉熵损失
        loss = -np.sum(targets * np.log(predictions)) / predictions.shape[0]

        self.cache = (predictions, targets)
        return loss

    def backward(self):
        """
        计算损失对预测值的梯度

        Returns:
            grad: 损失对 logits 的梯度, shape (N, C)
        """
        predictions, targets = self.cache
        grad = (predictions - targets) / predictions.shape[0]
        return grad


class L2Regularization:
    """L2 正则化"""

    def __init__(self, weight_decay=0.0):
        """
        Args:
            weight_decay: L2 正则化系数
        """
        self.weight_decay = weight_decay

    def forward(self, weights_list):
        """
        计算所有权重矩阵的 L2 正则化损失

        Args:
            weights_list: 权重矩阵列表

        Returns:
            reg_loss: 正则化损失
        """
        reg_loss = 0.0
        for W in weights_list:
            reg_loss += np.sum(W ** 2)
        return 0.5 * self.weight_decay * reg_loss

    def backward(self, W):
        """
        计算权重的正则化梯度

        Args:
            W: 权重矩阵

        Returns:
            grad: 正则化梯度
        """
        return self.weight_decay * W


if __name__ == "__main__":
    # 测试交叉熵损失
    predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    targets = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    ce_loss = CrossEntropyLoss()
    loss = ce_loss.forward(predictions, targets)
    grad = ce_loss.backward()

    print("预测值:")
    print(predictions)
    print("\n真实标签:")
    print(targets)
    print(f"\n交叉熵损失: {loss}")
    print("\n梯度:")
    print(grad)
