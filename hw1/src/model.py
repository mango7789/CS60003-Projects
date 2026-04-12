"""
三层神经网络模型
从零实现，包含自动微分与反向传播
"""

import numpy as np
import pickle
from activations import ReLU, Sigmoid, Tanh, Softmax, get_activation
from losses import CrossEntropyLoss, L2Regularization


class ThreeLayerMLP:
    """三层全连接神经网络"""

    def __init__(self, input_size, hidden_size, output_size, activation='relu',
                 weight_decay=0.0, seed=None):
        """
        初始化三层神经网络

        Args:
            input_size: 输入维度
            hidden_size: 隐藏层大小
            output_size: 输出维度 (类别数)
            activation: 激活函数 ('relu', 'sigmoid', 'tanh')
            weight_decay: L2 正则化系数
            seed: 随机种子 (None 表示不设置)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation
        self.weight_decay = weight_decay
        self.seed = seed

        # 初始化权重 (He 初始化)
        self.params = {}
        self._init_weights()

        # 激活函数
        self.activation1 = get_activation(activation)  # 第一隐藏层
        self.activation2 = get_activation(activation)  # 第二隐藏层
        self.softmax = Softmax()  # 输出层

        # 损失函数
        self.loss_fn = CrossEntropyLoss()

        # 缓存用于反向传播
        self.cache = {}

    def _init_weights(self):
        """初始化权重参数"""
        if self.seed is not None:
            np.random.seed(self.seed)

        # He 初始化 (适合 ReLU)
        if self.activation_name == 'relu':
            scale1 = np.sqrt(2.0 / self.input_size)
            scale2 = np.sqrt(2.0 / self.hidden_size)
            scale3 = np.sqrt(2.0 / self.hidden_size)
        # Xavier 初始化 (适合 Sigmoid/Tanh)
        else:
            scale1 = np.sqrt(1.0 / self.input_size)
            scale2 = np.sqrt(1.0 / self.hidden_size)
            scale3 = np.sqrt(1.0 / self.hidden_size)

        self.params = {
            'W1': np.random.randn(self.input_size, self.hidden_size) * scale1,
            'b1': np.zeros(self.hidden_size),
            'W2': np.random.randn(self.hidden_size, self.hidden_size) * scale2,
            'b2': np.zeros(self.hidden_size),
            'W3': np.random.randn(self.hidden_size, self.output_size) * scale3,
            'b3': np.zeros(self.output_size)
        }

        # 缓存参数名
        self.param_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']

    def count_parameters(self):
        """计算模型参数总数"""
        total = 0
        for name in self.param_names:
            total += self.params[name].size
        return total

    def get_layer_shapes(self):
        """获取各层参数形状"""
        shapes = {}
        for name in self.param_names:
            shapes[name] = self.params[name].shape
        return shapes

    def forward(self, X):
        """
        前向传播

        Args:
            X: 输入数据, shape (N, input_size)

        Returns:
            output: 预测概率, shape (N, output_size)
        """
        # 第一层
        z1 = X @ self.params['W1'] + self.params['b1']
        a1 = self.activation1.forward(z1)

        # 第二层
        z2 = a1 @ self.params['W2'] + self.params['b2']
        a2 = self.activation2.forward(z2)

        # 第三层 (输出层)
        z3 = a2 @ self.params['W3'] + self.params['b3']
        output = self.softmax.forward(z3)

        # 缓存用于反向传播
        self.cache = {
            'X': X,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'output': output
        }

        return output

    def backward(self, y_true):
        """
        反向传播

        Args:
            y_true: 真实标签 (one-hot), shape (N, output_size)

        Returns:
            grads: 参数梯度字典
        """
        grads = {}
        batch_size = y_true.shape[0]

        # 输出层梯度 (交叉熵 + softmax 的组合梯度)
        dz3 = self.cache['output'] - y_true  # (N, output_size)

        # 第三层权重梯度
        grads['W3'] = self.cache['a2'].T @ dz3  # (hidden_size, output_size)
        grads['b3'] = np.sum(dz3, axis=0)  # (output_size,)

        # 第二层梯度
        da2 = dz3 @ self.params['W3'].T  # (N, hidden_size)
        dz2 = self.activation2.backward(da2)  # 激活函数反向传播

        grads['W2'] = self.cache['a1'].T @ dz2  # (hidden_size, hidden_size)
        grads['b2'] = np.sum(dz2, axis=0)  # (hidden_size,)

        # 第一层梯度
        da1 = dz2 @ self.params['W2'].T  # (N, hidden_size)
        dz1 = self.activation1.backward(da1)  # 激活函数反向传播

        grads['W1'] = self.cache['X'].T @ dz1  # (input_size, hidden_size)
        grads['b1'] = np.sum(dz1, axis=0)  # (hidden_size,)

        # 添加 L2 正则化梯度
        if self.weight_decay > 0:
            grads['W1'] += self.weight_decay * self.params['W1']
            grads['W2'] += self.weight_decay * self.params['W2']
            grads['W3'] += self.weight_decay * self.params['W3']

        return grads

    def compute_loss(self, y_pred, y_true):
        """
        计算总损失 (交叉熵 + L2 正则化)

        Args:
            y_pred: 预测概率
            y_true: 真实标签

        Returns:
            loss: 总损失
            ce_loss: 交叉熵损失
            reg_loss: 正则化损失
        """
        # 交叉熵损失
        ce_loss = self.loss_fn.forward(y_pred, y_true)

        # L2 正则化损失
        reg_loss = 0.0
        if self.weight_decay > 0:
            for key in ['W1', 'W2', 'W3']:
                reg_loss += 0.5 * self.weight_decay * np.sum(self.params[key] ** 2)

        total_loss = ce_loss + reg_loss
        return total_loss, ce_loss, reg_loss

    def predict(self, X):
        """
        预测类别

        Args:
            X: 输入数据

        Returns:
            predictions: 预测类别索引
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """
        预测概率

        Args:
            X: 输入数据

        Returns:
            proba: 预测概率
        """
        return self.forward(X)

    def accuracy(self, X, y):
        """
        计算准确率

        Args:
            X: 输入数据
            y: 真实标签 (类别索引)

        Returns:
            acc: 准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save_weights(self, filepath):
        """保存模型权重"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'output_size': self.output_size,
                    'activation': self.activation_name,
                    'weight_decay': self.weight_decay
                }
            }, f)
        print(f"模型权重已保存到 {filepath}")

    def load_weights(self, filepath):
        """加载模型权重"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.params = data['params']
        print(f"模型权重已从 {filepath} 加载")

    @classmethod
    def from_file(cls, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(
            input_size=data['config']['input_size'],
            hidden_size=data['config']['hidden_size'],
            output_size=data['config']['output_size'],
            activation=data['config']['activation'],
            weight_decay=data['config']['weight_decay']
        )
        model.params = data['params']
        return model


if __name__ == "__main__":
    # 测试模型
    model = ThreeLayerMLP(
        input_size=12288,
        hidden_size=256,
        output_size=10,
        activation='relu'
    )

    # 随机测试数据
    X = np.random.randn(32, 12288).astype(np.float32)
    y = np.random.randint(0, 10, 32)
    y_onehot = np.zeros((32, 10))
    y_onehot[np.arange(32), y] = 1

    # 前向传播
    output = model.forward(X)
    print(f"输出形状: {output.shape}")
    print(f"输出和: {output.sum(axis=1)[:5]}")  # 应该接近 1

    # 计算损失
    loss, ce_loss, reg_loss = model.compute_loss(output, y_onehot)
    print(f"总损失: {loss:.4f}")

    # 反向传播
    grads = model.backward(y_onehot)
    print(f"梯度形状: W1 {grads['W1'].shape}, W2 {grads['W2'].shape}, W3 {grads['W3'].shape}")

    # 准确率
    acc = model.accuracy(X, y)
    print(f"准确率: {acc:.4f}")
