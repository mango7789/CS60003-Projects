"""
Test Script
Load trained model for test evaluation
"""

import os
import sys
import json
import argparse
import numpy as np

from data_loader import DataLoader
from model import ThreeLayerMLP
from evaluation import Evaluator, evaluate_model
from visualization import visualize_first_layer_weights, visualize_weight_patterns


def main(args):
    """Main function"""
    print("=" * 60)
    print("Model Test Evaluation")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data_loader = DataLoader(args.data_dir, img_size=64)

    processed_data_dir = os.path.join(args.exp_dir, 'processed_data')
    if os.path.exists(os.path.join(processed_data_dir, 'processed_data.npz')):
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_processed_data(processed_data_dir)
    else:
        images, labels = data_loader.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(images, labels)

    print(f"Test set: {X_test.shape}")

    # Class names
    class_names = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]

    # Model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(args.exp_dir, 'checkpoints', 'best_model.pkl')

    # Evaluation directory
    eval_dir = os.path.join(args.exp_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    # Evaluate model
    results = evaluate_model(model_path, X_test, y_test, class_names, eval_dir)

    # Weight visualization
    figures_dir = os.path.join(args.exp_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    visualize_first_layer_weights(model_path, figures_dir)

    print("\n" + "=" * 60)
    print("Test Complete!")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Test Script')

    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB',
                        help='Dataset directory')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment directory')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model weights path (optional, default uses best model in experiment directory)')

    args = parser.parse_args()
    main(args)
