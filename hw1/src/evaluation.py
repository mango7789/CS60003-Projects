"""
Evaluation Module
Contains accuracy calculation, confusion matrix, and error analysis
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model import ThreeLayerMLP


class Evaluator:
    """Evaluator class"""

    def __init__(self, model, class_names):
        """
        Initialize evaluator

        Args:
            model: Trained model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names

    def evaluate(self, X_test, y_test, batch_size=256):
        """
        Evaluate model on test set

        Args:
            X_test: Test data
            y_test: Test labels
            batch_size: Batch size

        Returns:
            accuracy: Accuracy
            predictions: Predictions
        """
        predictions = self.model.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        print(f"Test Accuracy: {accuracy:.4f} ({np.sum(predictions == y_test)}/{len(y_test)})")

        return accuracy, predictions

    def confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Calculate and plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Save path

        Returns:
            cm: Confusion matrix
        """
        n_classes = len(self.class_names)
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)

        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("-" * 80)
        print(f"{'True / Pred':<20}", end="")
        for name in self.class_names:
            print(f"{name[:8]:>10}", end="")
        print()

        for i, name in enumerate(self.class_names):
            print(f"{name:<20}", end="")
            for j in range(n_classes):
                print(f"{cm[i, j]:>10}", end="")
            print()

        # Plot confusion matrix heatmap
        plt.figure(figsize=(12, 10))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix (Normalized)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nConfusion matrix saved to {save_path}")

        plt.close()

        return cm

    def per_class_accuracy(self, y_true, y_pred):
        """
        Calculate per-class accuracy

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            class_acc: Per-class accuracy dict
        """
        class_acc = {}
        for i, name in enumerate(self.class_names):
            mask = (y_true == i)
            if mask.sum() > 0:
                acc = np.mean(y_pred[mask] == y_true[mask])
                class_acc[name] = acc
                print(f"{name}: {acc:.4f}")
        return class_acc

    def error_analysis(self, X_test, y_test, y_pred, n_samples=5, save_dir='./error_analysis'):
        """
        Error analysis

        Args:
            X_test: Test data
            y_test: True labels
            y_pred: Predicted labels
            n_samples: Number of samples per error type
            save_dir: Save directory

        Returns:
            error_samples: List of error samples
        """
        os.makedirs(save_dir, exist_ok=True)

        # Find all error samples
        error_mask = (y_pred != y_test)
        error_indices = np.where(error_mask)[0]

        print(f"\nError Analysis:")
        print(f"Total errors: {len(error_indices)}/{len(y_test)}")
        print("-" * 60)

        # Count error types
        error_pairs = {}
        for idx in error_indices:
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            pair = (true_label, pred_label)
            if pair not in error_pairs:
                error_pairs[pair] = []
            error_pairs[pair].append(idx)

        # Sort by error count
        sorted_errors = sorted(error_pairs.items(), key=lambda x: len(x[1]), reverse=True)

        error_samples = []

        # Analyze most common error types
        for (true_label, pred_label), indices in sorted_errors[:5]:
            true_name = self.class_names[true_label]
            pred_name = self.class_names[pred_label]

            print(f"\n{true_name} -> {pred_name}: {len(indices)} errors")

            # Select samples for visualization
            sample_indices = indices[:n_samples]

            fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
            if n_samples == 1:
                axes = [axes]

            for i, sample_idx in enumerate(sample_indices):
                # Reshape image
                img = X_test[sample_idx].reshape(64, 64, 3)
                img = np.clip(img, 0, 1)

                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'#{sample_idx}', fontsize=10)

            plt.suptitle(f'True: {true_name}, Pred: {pred_name}', fontsize=12)
            plt.tight_layout()

            save_path = os.path.join(save_dir, f'error_{true_name}_to_{pred_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            error_samples.append({
                'true_label': true_name,
                'pred_label': pred_name,
                'count': len(indices),
                'indices': [int(x) for x in sample_indices]
            })

        # Save error analysis report
        import json
        report_path = os.path.join(save_dir, 'error_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(error_samples, f, indent=2)
        print(f"\nError analysis report saved to {report_path}")

        return error_samples


def evaluate_model(model_path, X_test, y_test, class_names, save_dir='./evaluation'):
    """
    Complete model evaluation pipeline

    Args:
        model_path: Model weights path
        X_test: Test data
        y_test: Test labels
        class_names: Class names
        save_dir: Save directory

    Returns:
        results: Evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = ThreeLayerMLP.from_file(model_path)
    print(f"Model loaded: {model_path}")

    # Create evaluator
    evaluator = Evaluator(model, class_names)

    # Evaluate accuracy
    accuracy, predictions = evaluator.evaluate(X_test, y_test)

    # Confusion matrix
    cm = evaluator.confusion_matrix(
        y_test, predictions,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    class_acc = evaluator.per_class_accuracy(y_test, predictions)

    # Error analysis
    error_samples = evaluator.error_analysis(
        X_test, y_test, predictions,
        n_samples=5,
        save_dir=os.path.join(save_dir, 'error_analysis')
    )

    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': class_acc,
        'error_samples': error_samples
    }

    return results


if __name__ == "__main__":
    # Test evaluation module
    class_names = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]

    print("Evaluation module test")
