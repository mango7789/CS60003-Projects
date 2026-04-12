"""
Hyperparameter Search Module
Supports grid search and random search
"""

import numpy as np
import os
import json
import itertools
from datetime import datetime
from model import ThreeLayerMLP
from trainer import Trainer
from optimizers import SGDOptimizer, LearningRateScheduler


class HyperparameterSearch:
    """Hyperparameter search"""

    def __init__(self, X_train, y_train, X_val, y_val, save_dir='./search_results'):
        """
        Initialize hyperparameter search

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            save_dir: Results save directory
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.save_dir = save_dir
        self.results = []

        os.makedirs(save_dir, exist_ok=True)

    def grid_search(self, param_grid, epochs=50, batch_size=64):
        """
        Grid search

        Args:
            param_grid: Parameter grid dict
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            best_params: Best parameters
            best_acc: Best accuracy
        """
        print("Starting grid search...")
        print(f"Parameter grid: {param_grid}")

        # Generate all parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))

        print(f"Total {len(combinations)} parameter combinations")
        print("-" * 60)

        best_acc = 0.0
        best_params = None
        best_idx = 0

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\n[{idx + 1}/{len(combinations)}] Params: {params}")

            # Train model
            val_acc = self._train_and_evaluate(params, epochs, batch_size, idx)

            # Record result
            result = {
                'params': params,
                'val_acc': float(val_acc),
                'idx': idx
            }
            self.results.append(result)

            # Update best
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params
                best_idx = idx

            print(f"Val accuracy: {val_acc:.4f} | Best: {best_acc:.4f}")

        print("-" * 60)
        print("Grid search complete!")
        print(f"Best params: {best_params}")
        print(f"Best accuracy: {best_acc:.4f}")

        # Save results
        self._save_results()

        return best_params, best_acc

    def random_search(self, param_distributions, n_trials=20, epochs=50, batch_size=64):
        """
        Random search

        Args:
            param_distributions: Parameter distribution dict
            n_trials: Number of trials
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            best_params: Best parameters
            best_acc: Best accuracy
        """
        print("Starting random search...")
        print(f"Number of trials: {n_trials}")
        print("-" * 60)

        best_acc = 0.0
        best_params = None

        for trial in range(n_trials):
            # Random sample parameters
            params = {}
            for key, value in param_distributions.items():
                if isinstance(value, list):
                    params[key] = np.random.choice(value)
                elif isinstance(value, tuple) and len(value) == 2:
                    params[key] = np.random.uniform(value[0], value[1])
                    if key in ['hidden_size']:
                        params[key] = int(params[key])
                else:
                    params[key] = value

            print(f"\n[{trial + 1}/{n_trials}] Params: {params}")

            # Train model
            val_acc = self._train_and_evaluate(params, epochs, batch_size, trial)

            # Record result
            result = {
                'params': params,
                'val_acc': float(val_acc),
                'trial': trial
            }
            self.results.append(result)

            # Update best
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params

            print(f"Val accuracy: {val_acc:.4f} | Best: {best_acc:.4f}")

        print("-" * 60)
        print("Random search complete!")
        print(f"Best params: {best_params}")
        print(f"Best accuracy: {best_acc:.4f}")

        # Save results
        self._save_results()

        return best_params, best_acc

    def _train_and_evaluate(self, params, epochs, batch_size, idx):
        """
        Train and evaluate model

        Args:
            params: Parameter dict
            epochs: Training epochs
            batch_size: Batch size
            idx: Experiment index

        Returns:
            val_acc: Validation accuracy
        """
        # Create model
        model = ThreeLayerMLP(
            input_size=12288,
            hidden_size=params.get('hidden_size', 256),
            output_size=10,
            activation=params.get('activation', 'relu'),
            weight_decay=params.get('weight_decay', 0.0001)
        )

        # Create optimizer
        optimizer = SGDOptimizer(
            learning_rate=params.get('learning_rate', 0.01),
            momentum=params.get('momentum', 0.9),
            weight_decay=params.get('weight_decay', 0.0001)
        )

        # Create learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            optimizer,
            decay_type='step',
            decay_rate=0.5,
            decay_steps=20,
            min_lr=1e-6
        )

        # Create save directory
        exp_dir = os.path.join(self.save_dir, f'exp_{idx}')
        os.makedirs(exp_dir, exist_ok=True)

        # Train
        trainer = Trainer(model, optimizer, lr_scheduler, exp_dir)
        history = trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )

        # Return best validation accuracy
        return max(history['val_acc'])

    def _save_results(self):
        """Save search results"""
        results_file = os.path.join(self.save_dir, 'search_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSearch results saved to {results_file}")


if __name__ == "__main__":
    # Test hyperparameter search
    from data_loader import DataLoader

    # Load data
    loader = DataLoader('../../EuroSAT_RGB')
    images, labels = loader.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(images, labels)

    # Define parameter grid
    param_grid = {
        'hidden_size': [128, 256],
        'learning_rate': [0.01, 0.001],
        'activation': ['relu', 'tanh'],
        'weight_decay': [0.0001, 0.001]
    }

    # Grid search
    searcher = HyperparameterSearch(X_train, y_train, X_val, y_val, './search_results')
    best_params, best_acc = searcher.grid_search(param_grid, epochs=10, batch_size=64)
