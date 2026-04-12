"""
Three-Layer Neural Network Model
Implemented from scratch with automatic differentiation and backpropagation
"""

import numpy as np
import pickle
from activations import ReLU, Sigmoid, Tanh, Softmax, get_activation
from losses import CrossEntropyLoss, L2Regularization


class ThreeLayerMLP:
    """Three-layer fully connected neural network"""

    def __init__(self, input_size, hidden_size, output_size, activation='relu',
                 weight_decay=0.0, seed=None):
        """
        Initialize three-layer neural network

        Args:
            input_size: Input dimension
            hidden_size: Hidden layer size
            output_size: Output dimension (number of classes)
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            weight_decay: L2 regularization coefficient
            seed: Random seed (None means no fixed seed)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation
        self.weight_decay = weight_decay
        self.seed = seed

        # Initialize weights (He initialization)
        self.params = {}
        self._init_weights()

        # Activation functions
        self.activation1 = get_activation(activation)  # First hidden layer
        self.activation2 = get_activation(activation)  # Second hidden layer
        self.softmax = Softmax()  # Output layer

        # Loss function
        self.loss_fn = CrossEntropyLoss()

        # Cache for backpropagation
        self.cache = {}

    def _init_weights(self):
        """Initialize weight parameters"""
        if self.seed is not None:
            np.random.seed(self.seed)

        # Smaller initialization scale to prevent gradient explosion
        # He initialization (for ReLU)
        if self.activation_name == 'relu':
            scale1 = np.sqrt(2.0 / self.input_size) * 0.1
            scale2 = np.sqrt(2.0 / self.hidden_size) * 0.1
            scale3 = np.sqrt(2.0 / self.hidden_size) * 0.1
        # Xavier initialization (for Sigmoid/Tanh)
        else:
            scale1 = np.sqrt(1.0 / self.input_size) * 0.1
            scale2 = np.sqrt(1.0 / self.hidden_size) * 0.1
            scale3 = np.sqrt(1.0 / self.hidden_size) * 0.1

        self.params = {
            'W1': np.random.randn(self.input_size, self.hidden_size) * scale1,
            'b1': np.zeros(self.hidden_size),
            'W2': np.random.randn(self.hidden_size, self.hidden_size) * scale2,
            'b2': np.zeros(self.hidden_size),
            'W3': np.random.randn(self.hidden_size, self.output_size) * scale3,
            'b3': np.zeros(self.output_size)
        }

        # Cache parameter names
        self.param_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']

    def count_parameters(self):
        """Count total model parameters"""
        total = 0
        for name in self.param_names:
            total += self.params[name].size
        return total

    def get_layer_shapes(self):
        """Get layer parameter shapes"""
        shapes = {}
        for name in self.param_names:
            shapes[name] = self.params[name].shape
        return shapes

    def forward(self, X):
        """
        Forward propagation

        Args:
            X: Input data, shape (N, input_size)

        Returns:
            output: Predicted probabilities, shape (N, output_size)
        """
        # First layer
        z1 = X @ self.params['W1'] + self.params['b1']
        a1 = self.activation1.forward(z1)

        # Second layer
        z2 = a1 @ self.params['W2'] + self.params['b2']
        a2 = self.activation2.forward(z2)

        # Third layer (output layer)
        z3 = a2 @ self.params['W3'] + self.params['b3']
        output = self.softmax.forward(z3)

        # Cache for backpropagation
        self.cache = {
            'X': X,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'z3': z3, 'output': output
        }

        return output

    def backward(self, y_true):
        """
        Backward propagation

        Args:
            y_true: True labels (one-hot), shape (N, output_size)

        Returns:
            grads: Parameter gradient dict
        """
        grads = {}
        batch_size = y_true.shape[0]

        # Output layer gradient (cross-entropy + softmax combined gradient)
        dz3 = self.cache['output'] - y_true  # (N, output_size)

        # Third layer weight gradient
        grads['W3'] = self.cache['a2'].T @ dz3  # (hidden_size, output_size)
        grads['b3'] = np.sum(dz3, axis=0)  # (output_size,)

        # Second layer gradient
        da2 = dz3 @ self.params['W3'].T  # (N, hidden_size)
        dz2 = self.activation2.backward(da2)  # Activation backward

        grads['W2'] = self.cache['a1'].T @ dz2  # (hidden_size, hidden_size)
        grads['b2'] = np.sum(dz2, axis=0)  # (hidden_size,)

        # First layer gradient
        da1 = dz2 @ self.params['W2'].T  # (N, hidden_size)
        dz1 = self.activation1.backward(da1)  # Activation backward

        grads['W1'] = self.cache['X'].T @ dz1  # (input_size, hidden_size)
        grads['b1'] = np.sum(dz1, axis=0)  # (hidden_size,)

        # Add L2 regularization gradient
        if self.weight_decay > 0:
            grads['W1'] += self.weight_decay * self.params['W1']
            grads['W2'] += self.weight_decay * self.params['W2']
            grads['W3'] += self.weight_decay * self.params['W3']

        return grads

    def compute_loss(self, y_pred, y_true):
        """
        Compute total loss (cross-entropy + L2 regularization)

        Args:
            y_pred: Predicted probabilities
            y_true: True labels

        Returns:
            loss: Total loss
            ce_loss: Cross-entropy loss
            reg_loss: Regularization loss
        """
        # Cross-entropy loss
        ce_loss = self.loss_fn.forward(y_pred, y_true)

        # L2 regularization loss
        reg_loss = 0.0
        if self.weight_decay > 0:
            for key in ['W1', 'W2', 'W3']:
                reg_loss += 0.5 * self.weight_decay * np.sum(self.params[key] ** 2)

        total_loss = ce_loss + reg_loss
        return total_loss, ce_loss, reg_loss

    def predict(self, X):
        """
        Predict class

        Args:
            X: Input data

        Returns:
            predictions: Predicted class indices
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def predict_proba(self, X):
        """
        Predict probabilities

        Args:
            X: Input data

        Returns:
            proba: Predicted probabilities
        """
        return self.forward(X)

    def accuracy(self, X, y):
        """
        Compute accuracy

        Args:
            X: Input data
            y: True labels (class indices)

        Returns:
            acc: Accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save_weights(self, filepath):
        """Save model weights"""
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
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load model weights"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.params = data['params']
        print(f"Model weights loaded from {filepath}")

    @classmethod
    def from_file(cls, filepath):
        """Load model from file"""
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
    # Test model
    model = ThreeLayerMLP(
        input_size=12288,
        hidden_size=256,
        output_size=10,
        activation='relu'
    )

    # Random test data
    X = np.random.randn(32, 12288).astype(np.float32)
    y = np.random.randint(0, 10, 32)
    y_onehot = np.zeros((32, 10))
    y_onehot[np.arange(32), y] = 1

    # Forward propagation
    output = model.forward(X)
    print(f"Output shape: {output.shape}")
    print(f"Output sum: {output.sum(axis=1)[:5]}")  # Should be close to 1

    # Compute loss
    loss, ce_loss, reg_loss = model.compute_loss(output, y_onehot)
    print(f"Total loss: {loss:.4f}")

    # Backward propagation
    grads = model.backward(y_onehot)
    print(f"Gradient shapes: W1 {grads['W1'].shape}, W2 {grads['W2'].shape}, W3 {grads['W3'].shape}")

    # Accuracy
    acc = model.accuracy(X, y)
    print(f"Accuracy: {acc:.4f}")
