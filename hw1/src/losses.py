"""
Loss Functions Module
Contains cross-entropy loss function
"""

import numpy as np


class CrossEntropyLoss:
    """Cross-entropy loss function"""

    def __init__(self):
        self.cache = None

    def forward(self, predictions, targets, epsilon=1e-15):
        """
        Compute cross-entropy loss

        Args:
            predictions: Model predicted probabilities (softmax output), shape (N, C)
            targets: True labels (one-hot encoded), shape (N, C)
            epsilon: Small value to prevent log(0)

        Returns:
            loss: Scalar loss value
        """
        # Clip predictions to prevent numerical issues
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        # Compute cross-entropy loss
        loss = -np.sum(targets * np.log(predictions)) / predictions.shape[0]

        self.cache = (predictions, targets)
        return loss

    def backward(self):
        """
        Compute gradient of loss w.r.t. predictions

        Returns:
            grad: Gradient of loss w.r.t. logits, shape (N, C)
        """
        predictions, targets = self.cache
        grad = (predictions - targets) / predictions.shape[0]
        return grad


class L2Regularization:
    """L2 regularization"""

    def __init__(self, weight_decay=0.0):
        """
        Args:
            weight_decay: L2 regularization coefficient
        """
        self.weight_decay = weight_decay

    def forward(self, weights_list):
        """
        Compute L2 regularization loss for all weight matrices

        Args:
            weights_list: List of weight matrices

        Returns:
            reg_loss: Regularization loss
        """
        reg_loss = 0.0
        for W in weights_list:
            reg_loss += np.sum(W ** 2)
        return 0.5 * self.weight_decay * reg_loss

    def backward(self, W):
        """
        Compute regularization gradient for weights

        Args:
            W: Weight matrix

        Returns:
            grad: Regularization gradient
        """
        return self.weight_decay * W


if __name__ == "__main__":
    # Test cross-entropy loss
    predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    targets = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    ce_loss = CrossEntropyLoss()
    loss = ce_loss.forward(predictions, targets)
    grad = ce_loss.backward()

    print("Predictions:")
    print(predictions)
    print("\nTrue labels:")
    print(targets)
    print(f"\nCross-entropy loss: {loss}")
    print("\nGradient:")
    print(grad)
