"""
Main Training Script
Complete training pipeline: data loading, model training, hyperparameter search, evaluation
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from datetime import datetime

from data_loader import DataLoader, get_batches
from model import ThreeLayerMLP
from trainer import Trainer, train_model
from optimizers import SGDOptimizer, LearningRateScheduler
from hyperparameter_search import HyperparameterSearch
from evaluation import Evaluator, evaluate_model
from visualization import plot_training_curves, visualize_first_layer_weights, visualize_weight_patterns


def setup_logging(exp_dir):
    """Setup logging"""
    log_file = os.path.join(exp_dir, 'training.log')

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler - record all info
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Console handler - only show important info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main(args):
    """Main function"""
    # Set random seed
    np.random.seed(args.seed)

    # Create experiment directory
    if args.exp_name:
        exp_dir = os.path.join(args.output_dir, args.exp_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(exp_dir)

    # Output experiment directory path for shell scripts
    print(f"EXP_DIR:{exp_dir}")

    # Save config
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 60)
    logger.info("Three-Layer Neural Network - EuroSAT Image Classification")
    logger.info("=" * 60)
    logger.info(f"Experiment directory: {exp_dir}")
    logger.debug(f"Config: {config}")
    logger.info("=" * 60)

    # ========== Data Loading ==========
    logger.info("\n[1/5] Data Loading...")
    data_loader = DataLoader(args.data_dir, img_size=64)

    processed_data_dir = os.path.join(exp_dir, 'processed_data')
    if os.path.exists(os.path.join(processed_data_dir, 'processed_data.npz')):
        logger.info("Loading preprocessed data...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_processed_data(processed_data_dir)
    else:
        logger.info("Loading raw data and processing...")
        images, labels = data_loader.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(images, labels)
        data_loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, processed_data_dir)

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ========== Hyperparameter Search ==========
    if args.search_hyperparams:
        logger.info("\n[2/5] Hyperparameter Search...")
        search_dir = os.path.join(exp_dir, 'hyperparameter_search')

        if args.search_type == 'grid':
            param_grid = {
                'hidden_size': args.hidden_sizes,
                'learning_rate': args.learning_rates,
                'activation': args.activations,
                'weight_decay': args.weight_decays
            }
            searcher = HyperparameterSearch(X_train, y_train, X_val, y_val, search_dir)
            best_params, best_acc = searcher.grid_search(
                param_grid,
                epochs=args.search_epochs,
                batch_size=args.batch_size
            )
        else:  # random search
            param_distributions = {
                'hidden_size': (128, 512),
                'learning_rate': (0.0001, 0.1),
                'activation': args.activations,
                'weight_decay': (0.00001, 0.01)
            }
            searcher = HyperparameterSearch(X_train, y_train, X_val, y_val, search_dir)
            best_params, best_acc = searcher.random_search(
                param_distributions,
                n_trials=args.n_trials,
                epochs=args.search_epochs,
                batch_size=args.batch_size
            )

        # Update config
        args.hidden_size = best_params['hidden_size']
        args.learning_rate = best_params['learning_rate']
        args.activation = best_params['activation']
        args.weight_decay = best_params['weight_decay']

        logger.info(f"\nBest hyperparameters: {best_params}")
        logger.info(f"Best validation accuracy: {best_acc:.4f}")

    # ========== Model Training ==========
    logger.info("\n[3/5] Model Training...")
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')

    train_config = {
        'input_size': 64 * 64 * 3,  # 12288
        'hidden_size': args.hidden_size,
        'output_size': 10,
        'activation': args.activation,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_decay_type': args.lr_decay_type,
        'lr_decay_rate': args.lr_decay_rate,
        'lr_decay_steps': args.lr_decay_steps,
        'grad_clip': args.grad_clip
    }

    model, history = train_model(train_config, X_train, y_train, X_val, y_val, checkpoint_dir, logger)

    # ========== Visualization ==========
    logger.info("\n[4/5] Visualization...")
    figures_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Training curves
    plot_training_curves(os.path.join(checkpoint_dir, 'history.json'), figures_dir)

    # Weight visualization
    visualize_first_layer_weights(os.path.join(checkpoint_dir, 'best_model.pkl'), figures_dir)
    visualize_weight_patterns(os.path.join(checkpoint_dir, 'best_model.pkl'), figures_dir)

    # ========== Test Evaluation ==========
    logger.info("\n[5/5] Test Evaluation...")
    evaluation_dir = os.path.join(exp_dir, 'evaluation')

    class_names = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]

    results = evaluate_model(
        os.path.join(checkpoint_dir, 'best_model.pkl'),
        X_test, y_test,
        class_names,
        evaluation_dir
    )

    # Save final results
    final_results = {
        'test_accuracy': float(results['accuracy']),
        'best_val_accuracy': float(max(history['val_acc'])),
        'best_epoch': int(history['val_acc'].index(max(history['val_acc'])) + 1),
        'hyperparameters': {
            'hidden_size': args.hidden_size,
            'learning_rate': args.learning_rate,
            'activation': args.activation,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum
        }
    }

    with open(os.path.join(exp_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Three-Layer Neural Network Training Script')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (optional)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default 0.001 for stability)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum coefficient')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='L2 regularization coefficient')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Gradient clipping max norm (default 5.0)')

    # Learning rate decay
    parser.add_argument('--lr_decay_type', type=str, default='step',
                        choices=['step', 'exponential', 'cosine', 'linear'],
                        help='Learning rate decay type')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='Learning rate decay rate')
    parser.add_argument('--lr_decay_steps', type=int, default=30,
                        help='Learning rate decay steps')

    # Hyperparameter search
    parser.add_argument('--search_hyperparams', action='store_true',
                        help='Whether to perform hyperparameter search')
    parser.add_argument('--search_type', type=str, default='grid',
                        choices=['grid', 'random'],
                        help='Search type')
    parser.add_argument('--search_epochs', type=int, default=30,
                        help='Training epochs during search')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials for random search')

    # Grid search parameters
    parser.add_argument('--hidden_sizes', type=int, nargs='+',
                        default=[128, 256, 512],
                        help='Hidden size list')
    parser.add_argument('--learning_rates', type=float, nargs='+',
                        default=[0.001, 0.01, 0.1],
                        help='Learning rate list')
    parser.add_argument('--activations', type=str, nargs='+',
                        default=['relu', 'tanh'],
                        help='Activation function list')
    parser.add_argument('--weight_decays', type=float, nargs='+',
                        default=[0.0001, 0.001, 0.01],
                        help='Weight decay list')

    args = parser.parse_args()
    main(args)
