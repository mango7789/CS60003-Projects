"""
Visualization Module
Contains training curves and weight visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


def plot_training_curves(history_path, save_dir='./figures'):
    """
    Plot training curves

    Args:
        history_path: Training history file path
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss and Accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def visualize_first_layer_weights(model_path, save_dir='./figures', n_features=16):
    """
    Visualize first layer weights

    Args:
        model_path: Model weights path
        save_dir: Save directory
        n_features: Number of features to display
    """
    import pickle

    os.makedirs(save_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    W1 = data['params']['W1']  # (input_size, hidden_size) = (12288, hidden_size)

    hidden_size = W1.shape[1]
    n_features = min(n_features, hidden_size)

    # Randomly select features for visualization
    indices = np.random.choice(hidden_size, n_features, replace=False)

    fig, axes = plt.subplots(4, n_features // 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        # Reshape weights to image shape (64, 64, 3)
        weights = W1[:, idx].reshape(64, 64, 3)

        # Normalize to [0, 1]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

        axes[i].imshow(weights)
        axes[i].axis('off')
        axes[i].set_title(f'Feature {idx}', fontsize=10)

    plt.suptitle('First Layer Weight Visualization', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'first_layer_weights.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Weight visualization saved to {save_path}")


def visualize_weight_patterns(model_path, save_dir='./figures'):
    """
    Analyze and visualize weight patterns

    Args:
        model_path: Model weights path
        save_dir: Save directory
    """
    import pickle

    os.makedirs(save_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    W1 = data['params']['W1']  # (12288, hidden_size)

    # Calculate feature statistics
    hidden_size = W1.shape[1]

    # Calculate color channel mean for each feature
    R_mean = np.mean(np.abs(W1[:4096, :]), axis=0)
    G_mean = np.mean(np.abs(W1[4096:8192, :]), axis=0)
    B_mean = np.mean(np.abs(W1[8192:, :]), axis=0)

    # Plot color distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color distribution scatter plot
    axes[0].scatter(R_mean, G_mean, c='red', alpha=0.5, label='R-G')
    axes[0].set_xlabel('Red Channel Mean Weight', fontsize=12)
    axes[0].set_ylabel('Green Channel Mean Weight', fontsize=12)
    axes[0].set_title('Weight Color Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Weight distribution histogram
    axes[1].hist(W1.flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Weight Value', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('First Layer Weight Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'weight_patterns.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Weight patterns saved to {save_path}")


def plot_hyperparameter_comparison(results_path, save_dir='./figures'):
    """
    Plot hyperparameter comparison

    Args:
        results_path: Hyperparameter search results path
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract validation accuracy
    val_accs = [r['val_acc'] for r in results]

    # Plot accuracy distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(val_accs))
    bars = ax.bar(x, val_accs, color='steelblue', alpha=0.7)

    # Mark best
    best_idx = np.argmax(val_accs)
    bars[best_idx].set_color('red')

    ax.set_xlabel('Experiment Index', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Hyperparameter Search Results', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'hyperparameter_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Hyperparameter comparison saved to {save_path}")


if __name__ == "__main__":
    # Test visualization module
    print("Visualization module test")
