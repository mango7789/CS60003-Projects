"""
Data Loading and Preprocessing Module
Load EuroSAT dataset and perform preprocessing
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle


class DataLoader:
    """Data Loader class"""

    def __init__(self, data_dir, img_size=64, test_ratio=0.15, val_ratio=0.15, random_state=42):
        """
        Initialize data loader

        Args:
            data_dir: Dataset directory path
            img_size: Image size
            test_ratio: Test set ratio
            val_ratio: Validation set ratio
            random_state: Random seed
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

        # Class names
        self.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def load_data(self):
        """Load all image data"""
        images = []
        labels = []

        print("Loading dataset...")
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist")
                continue

            class_idx = self.class_to_idx[class_name]
            files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]

            for file in files:
                img_path = os.path.join(class_dir, file)
                try:
                    img = Image.open(img_path)
                    img = img.resize((self.img_size, self.img_size))
                    img_array = np.array(img, dtype=np.float32)
                    # Normalize to [0, 1]
                    img_array = img_array / 255.0
                    # Flatten to 1D vector (64*64*3 = 12288)
                    img_array = img_array.flatten()
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")

        images = np.array(images)
        labels = np.array(labels)

        print(f"Data loaded: {len(images)} images, {len(self.class_names)} classes")
        print(f"Samples per class: {np.bincount(labels)}")

        return images, labels

    def split_data(self, images, labels):
        """Split into train, validation, and test sets"""
        # First split test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=labels
        )

        # Split validation set from remaining data
        val_ratio_adjusted = self.val_ratio / (1 - self.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )

        print(f"Dataset split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def one_hot_encode(self, labels, num_classes=10):
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, save_dir):
        """Save processed data"""
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_dir, 'processed_data.npz'),
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test
        )
        print(f"Processed data saved to {save_dir}")

    def load_processed_data(self, save_dir):
        """Load processed data"""
        data = np.load(os.path.join(save_dir, 'processed_data.npz'))
        return (data['X_train'], data['X_val'], data['X_test'],
                data['y_train'], data['y_val'], data['y_test'])


def get_batches(X, y, batch_size, shuffle=True):
    """Generate batch data"""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


if __name__ == "__main__":
    # Test data loading
    data_dir = "../EuroSAT_RGB"
    loader = DataLoader(data_dir)
    images, labels = loader.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(images, labels)
    loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, "./processed_data")
