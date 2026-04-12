"""
Training Module
Contains training loop and model saving
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
    """Trainer class"""

    def __init__(self, model, optimizer, lr_scheduler, save_dir='./checkpoints', logger=None):
        """
        Initialize trainer

        Args:
            model: Model instance
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler
            save_dir: Model save directory
            logger: Logger
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir
        self.logger = logger or logging.getLogger(__name__)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, X_train, y_train, batch_size, show_progress=False, epoch_num=0, total_epochs=0):
        """
        Train one epoch

        Args:
            X_train: Training data
            y_train: Training labels (class indices)
            batch_size: Batch size
            show_progress: Whether to show progress bar
            epoch_num: Current epoch number
            total_epochs: Total epochs

        Returns:
            avg_loss: Average loss
            avg_acc: Average accuracy
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Get batch data
        batches = list(get_batches(X_train, y_train, batch_size, shuffle=True))

        if show_progress:
            pbar = tqdm(batches, desc=f'Epoch {epoch_num}/{total_epochs}',
                        leave=False, ncols=100)
        else:
            pbar = batches

        for batch_X, batch_y in pbar:
            # One-hot encoding
            actual_batch_size = batch_X.shape[0]
            y_onehot = np.zeros((actual_batch_size, self.model.output_size))
            y_onehot[np.arange(actual_batch_size), batch_y] = 1

            # Forward propagation
            output = self.model.forward(batch_X)

            # Compute loss
            loss, ce_loss, reg_loss = self.model.compute_loss(output, y_onehot)

            # Backward propagation
            grads = self.model.backward(y_onehot)

            # Parameter update
            self.optimizer.step(self.model.params, grads, self.model.param_names)

            # Statistics
            total_loss += loss * actual_batch_size
            predictions = np.argmax(output, axis=1)
            total_correct += np.sum(predictions == batch_y)
            total_samples += actual_batch_size

            # Update progress bar info
            if show_progress:
                current_acc = total_correct / total_samples
                current_loss = total_loss / total_samples
                pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def validate(self, X_val, y_val, batch_size=256):
        """
        Validate model

        Args:
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size

        Returns:
            avg_loss: Average loss
            avg_acc: Average accuracy
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in get_batches(X_val, y_val, batch_size, shuffle=False):
            actual_batch_size = batch_X.shape[0]
            y_onehot = np.zeros((actual_batch_size, self.model.output_size))
            y_onehot[np.arange(actual_batch_size), batch_y] = 1

            # Forward propagation
            output = self.model.forward(batch_X)

            # Compute loss
            loss, _, _ = self.model.compute_loss(output, y_onehot)

            # Statistics
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
        Complete training loop

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to show progress bar
            early_stopping_patience: Early stopping patience (None means disabled)

        Returns:
            history: Training history
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        self.logger.info(f"Batch size: {batch_size}, Epochs: {epochs}")
        self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
        if early_stopping_patience:
            self.logger.info(f"Early Stopping: patience={early_stopping_patience}")
        self.logger.info("-" * 60)

        no_improve_count = 0

        for epoch in range(epochs):
            # Train one epoch (with progress bar)
            train_loss, train_acc = self.train_epoch(
                X_train, y_train, batch_size,
                show_progress=verbose, epoch_num=epoch + 1, total_epochs=epochs
            )

            # Validate
            val_loss, val_acc = self.validate(X_val, y_val)

            # Record history
            self.history['train_loss'].append(float(train_loss))
            self.history['train_acc'].append(float(train_acc))
            self.history['val_loss'].append(float(val_loss))
            self.history['val_acc'].append(float(val_acc))
            self.history['learning_rates'].append(float(self.optimizer.learning_rate))

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.model.save_weights(os.path.join(self.save_dir, 'best_model.pkl'))
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Learning rate decay
            self.lr_scheduler.step()

            # Print epoch result
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
        self.logger.info(f"Training complete! Best val accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch + 1})")

        return self.history

    def save_history(self, filepath):
        """Save training history"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {filepath}")


def train_model(config, X_train, y_train, X_val, y_val, save_dir, logger=None):
    """
    Train model with config

    Args:
        config: Config dict
        X_train, y_train: Training data
        X_val, y_val: Validation data
        save_dir: Save directory
        logger: Logger

    Returns:
        model: Trained model
        history: Training history
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Create model
    model = ThreeLayerMLP(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        activation=config['activation'],
        weight_decay=config['weight_decay']
    )

    # Create optimizer
    optimizer = SGDOptimizer(
        learning_rate=config['learning_rate'],
        momentum=config.get('momentum', 0.9),
        weight_decay=config['weight_decay']
    )

    # Create learning rate scheduler
    lr_scheduler = LearningRateScheduler(
        optimizer,
        decay_type=config.get('lr_decay_type', 'step'),
        decay_rate=config.get('lr_decay_rate', 0.1),
        decay_steps=config.get('lr_decay_steps', 30),
        min_lr=1e-6
    )

    # Create trainer
    trainer = Trainer(model, optimizer, lr_scheduler, save_dir, logger)

    # Train
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=True
    )

    # Save training history
    trainer.save_history(os.path.join(save_dir, 'history.json'))

    return model, history