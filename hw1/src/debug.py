"""
Debug script to check data and model
Run this to diagnose training issues
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ThreeLayerMLP
from data_loader import DataLoader

def check_data():
    """Check if data is loaded correctly"""
    print("=" * 60)
    print("DATA CHECK")
    print("=" * 60)

    try:
        loader = DataLoader('../EuroSAT_RGB')
        images, labels = loader.load_data()

        print(f"\nImages shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Images dtype: {images.dtype}")
        print(f"Images min: {images.min():.6f}")
        print(f"Images max: {images.max():.6f}")
        print(f"Images mean: {images.mean():.6f}")
        print(f"Images std: {images.std():.6f}")

        # Check for issues
        print(f"\nHas NaN: {np.isnan(images).any()}")
        print(f"Has Inf: {np.isinf(images).any()}")
        print(f"All zeros: {(images == 0).all()}")
        print(f"All same: {(images == images[0]).all()}")

        # Check label distribution
        print(f"\nLabel distribution: {np.bincount(labels)}")

        # Check a sample image
        sample = images[0].reshape(64, 64, 3)
        print(f"\nSample image stats:")
        print(f"  Shape: {sample.shape}")
        print(f"  Min: {sample.min():.6f}, Max: {sample.max():.6f}")
        print(f"  Mean: {sample.mean():.6f}")

        return images, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def check_model(images, labels):
    """Check model forward pass"""
    print("\n" + "=" * 60)
    print("MODEL CHECK")
    print("=" * 60)

    if images is None:
        print("No data to check")
        return

    # Create model
    model = ThreeLayerMLP(
        input_size=12288,
        hidden_size=128,  # Based on your training output
        output_size=10,
        activation='relu',
        weight_decay=0.0001
    )

    print(f"\nModel parameters: {model.count_parameters():,}")

    # Test with a small batch
    batch_size = 4
    X = images[:batch_size]
    y = labels[:batch_size]

    print(f"\nTest batch shape: {X.shape}")
    print(f"Test labels: {y}")

    # Forward pass
    print("\nForward pass...")
    output = model.forward(X)

    print(f"Output shape: {output.shape}")
    print(f"Output sum per sample: {output.sum(axis=1)}")
    print(f"Output min: {output.min():.6f}")
    print(f"Output max: {output.max():.6f}")
    print(f"Output sample:\n{output[0]}")

    # Check for NaN/Inf
    print(f"\nOutput has NaN: {np.isnan(output).any()}")
    print(f"Output has Inf: {np.isinf(output).any()}")

    # One-hot encoding
    y_onehot = np.zeros((batch_size, 10))
    y_onehot[np.arange(batch_size), y] = 1

    # Compute loss
    print("\nLoss computation...")
    loss, ce_loss, reg_loss = model.compute_loss(output, y_onehot)

    print(f"Cross-entropy loss: {ce_loss:.6f}")
    print(f"Regularization loss: {reg_loss:.6f}")
    print(f"Total loss: {loss:.6f}")

    # Expected loss for random guessing
    expected_loss = -np.log(1/10)
    print(f"\nExpected loss for random guessing: {expected_loss:.6f}")

    if loss > 10:
        print("\n⚠️  WARNING: Loss is abnormally high!")
        print("   This suggests there may be an issue with the data or model.")

    # Backward pass
    print("\nBackward pass...")
    grads = model.backward(y_onehot)

    # Check gradients
    print("\nGradient statistics:")
    for name in ['W1', 'W2', 'W3']:
        g = grads[name]
        print(f"  {name}: min={g.min():.6e}, max={g.max():.6e}, mean={g.mean():.6e}")
        print(f"       has_nan={np.isnan(g).any()}, has_inf={np.isinf(g).any()}")


def check_weight_init():
    """Check weight initialization"""
    print("\n" + "=" * 60)
    print("WEIGHT INITIALIZATION CHECK")
    print("=" * 60)

    model = ThreeLayerMLP(
        input_size=12288,
        hidden_size=128,
        output_size=10,
        activation='relu'
    )

    for name in ['W1', 'W2', 'W3']:
        W = model.params[name]
        print(f"\n{name}:")
        print(f"  Shape: {W.shape}")
        print(f"  Mean: {W.mean():.6f}")
        print(f"  Std: {W.std():.6f}")
        print(f"  Min: {W.min():.6f}")
        print(f"  Max: {W.max():.6f}")


if __name__ == "__main__":
    print("DEBUG SCRIPT FOR NEURAL NETWORK TRAINING")

    # Check weight initialization
    check_weight_init()

    # Check data
    images, labels = check_data()

    # Check model
    check_model(images, labels)

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
