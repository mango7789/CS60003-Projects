"""
激活函数模块
包含 ReLU, Sigmoid, Tanh 激活函数及其导数
"""

import numpy as np


class Activation:
    """激活函数基类"""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class ReLU(Activation):
    """ReLU 激活函数: max(0, x)"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """前向传播"""
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        """反向传播"""
        x = self.cache
        grad = grad_output * (x > 0).astype(np.float32)
        return grad


class Sigmoid(Activation):
    """Sigmoid 激活函数: 1 / (1 + exp(-x))"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """前向传播"""
        # 数值稳定性处理
        x = np.clip(x, -500, 500)
        self.cache = 1 / (1 + np.exp(-x))
        return self.cache

    def backward(self, grad_output):
        """反向传播"""
        s = self.cache
        grad = grad_output * s * (1 - s)
        return grad


class Tanh(Activation):
    """Tanh 激活函数: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """前向传播"""
        self.cache = np.tanh(x)
        return self.cache

    def backward(self, grad_output):
        """反向传播"""
        t = self.cache
        grad = grad_output * (1 - t ** 2)
        return grad


class Softmax(Activation):
    """Softmax 激活函数，用于输出层"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """前向传播"""
        # 数值稳定性处理：减去最大值
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.cache = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.cache

    def backward(self, grad_output):
        """反向传播 - 通常与交叉熵损失结合使用"""
        # 当与交叉熵损失结合时，梯度简化为 (softmax_output - y_true)
        # 这里保留通用形式
        return grad_output


def get_activation(name):
    """根据名称获取激活函数"""
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax
    }
    if name.lower() not in activations:
        raise ValueError(f"不支持的激活函数: {name}")
    return activations[name.lower()]()


if __name__ == "__main__":
    # 测试激活函数
    x = np.array([[-1, 0, 1], [2, -2, 0.5]], dtype=np.float32)

    print("输入:")
    print(x)

    print("\nReLU:")
    relu = ReLU()
    print("Forward:", relu.forward(x))
    print("Backward:", relu.backward(np.ones_like(x)))

    print("\nSigmoid:")
    sigmoid = Sigmoid()
    print("Forward:", sigmoid.forward(x))
    print("Backward:", sigmoid.backward(np.ones_like(x)))

    print("\nTanh:")
    tanh = Tanh()
    print("Forward:", tanh.forward(x))
    print("Backward:", tanh.backward(np.ones_like(x)))

    print("\nSoftmax:")
    softmax = Softmax()
    print("Forward:", softmax.forward(x))
