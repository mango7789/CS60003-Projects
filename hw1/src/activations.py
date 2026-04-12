"""
Activation Functions Module
Contains ReLU, Sigmoid, Tanh activation functions and their derivatives
"""

import numpy as np


class Activation:
    """Activation function base class"""

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class ReLU(Activation):
    """ReLU activation function: max(0, x)"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Forward propagation"""
        self.cache = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        """Backward propagation"""
        x = self.cache
        grad = grad_output * (x > 0).astype(np.float32)
        return grad


class Sigmoid(Activation):
    """Sigmoid activation function: 1 / (1 + exp(-x))"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Forward propagation"""
        # Numerical stability
        x = np.clip(x, -500, 500)
        self.cache = 1 / (1 + np.exp(-x))
        return self.cache

    def backward(self, grad_output):
        """Backward propagation"""
        s = self.cache
        grad = grad_output * s * (1 - s)
        return grad


class Tanh(Activation):
    """Tanh activation function: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Forward propagation"""
        self.cache = np.tanh(x)
        return self.cache

    def backward(self, grad_output):
        """Backward propagation"""
        t = self.cache
        grad = grad_output * (1 - t ** 2)
        return grad


class Softmax(Activation):
    """Softmax activation function, used for output layer"""

    def __init__(self):
        self.cache = None

    def forward(self, x):
        """Forward propagation"""
        # Numerical stability: subtract max value
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.cache = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.cache

    def backward(self, grad_output):
        """Backward propagation - usually combined with cross-entropy loss"""
        # When combined with cross-entropy loss, gradient simplifies to (softmax_output - y_true)
        # Keep general form here
        return grad_output


def get_activation(name):
    """Get activation function by name"""
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'softmax': Softmax
    }
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation function: {name}")
    return activations[name.lower()]()


if __name__ == "__main__":
    # Test activation functions
    x = np.array([[-1, 0, 1], [2, -2, 0.5]], dtype=np.float32)

    print("Input:")
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
