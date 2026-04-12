"""
Optimizer Module
Contains SGD optimizer and learning rate decay strategies
"""

import numpy as np


class SGDOptimizer:
    """Stochastic Gradient Descent Optimizer"""

    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0, grad_clip=None):
        """
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient
            weight_decay: L2 regularization coefficient
            grad_clip: Maximum gradient norm for clipping (None to disable)
        """
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.velocity = {}  # Store momentum velocity

    def _clip_gradients(self, grads, param_names):
        """Clip gradients by global norm"""
        if self.grad_clip is None:
            return grads

        # Calculate global norm
        total_norm = 0.0
        for name in param_names:
            if name in grads:
                total_norm += np.sum(grads[name] ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        if total_norm > self.grad_clip:
            scale = self.grad_clip / (total_norm + 1e-6)
            for name in param_names:
                if name in grads:
                    grads[name] = grads[name] * scale

        return grads

    def step(self, params, grads, param_names):
        """
        Perform one parameter update step

        Args:
            params: Parameter dict {name: value}
            grads: Gradient dict {name: value}
            param_names: Parameter name list
        """
        # Clip gradients
        grads = self._clip_gradients(grads, param_names)

        for name in param_names:
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(params[name])

            grad = grads[name]

            # Momentum update
            self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * grad
            params[name] = params[name] + self.velocity[name]

    def update_learning_rate(self, new_lr):
        """Update learning rate"""
        self.learning_rate = new_lr


class LearningRateScheduler:
    """Learning rate scheduler"""

    def __init__(self, optimizer, decay_type='step', decay_rate=0.1, decay_steps=30,
                 min_lr=1e-6):
        """
        Args:
            optimizer: Optimizer instance
            decay_type: Decay type ('step', 'exponential', 'cosine', 'linear')
            decay_rate: Decay rate
            decay_steps: Decay steps (for step decay)
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.learning_rate
        self.current_step = 0

    def step(self, epoch=None):
        """Update learning rate"""
        self.current_step += 1

        if self.decay_type == 'step':
            # Step decay
            new_lr = self.initial_lr * (self.decay_rate ** (self.current_step // self.decay_steps))
        elif self.decay_type == 'exponential':
            # Exponential decay
            new_lr = self.initial_lr * (self.decay_rate ** self.current_step)
        elif self.decay_type == 'cosine':
            # Cosine annealing
            new_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                     (1 + np.cos(np.pi * self.current_step / self.decay_steps))
        elif self.decay_type == 'linear':
            # Linear decay
            new_lr = self.initial_lr - (self.initial_lr - self.min_lr) * \
                     (self.current_step / self.decay_steps)
        else:
            raise ValueError(f"Unsupported decay type: {self.decay_type}")

        new_lr = max(new_lr, self.min_lr)
        self.optimizer.update_learning_rate(new_lr)
        return new_lr

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.learning_rate


if __name__ == "__main__":
    # Test optimizer
    params = {'W1': np.random.randn(3, 4), 'b1': np.zeros(4)}
    grads = {'W1': np.random.randn(3, 4) * 0.1, 'b1': np.random.randn(4) * 0.1}

    optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9, weight_decay=0.001)
    scheduler = LearningRateScheduler(optimizer, decay_type='step', decay_rate=0.5, decay_steps=10)

    print("Initial parameters:")
    print(params['W1'][:2, :2])

    for step in range(25):
        optimizer.step(params, grads, ['W1', 'b1'])
        lr = scheduler.step()
        if step % 5 == 0:
            print(f"Step {step}, LR: {lr:.6f}")
