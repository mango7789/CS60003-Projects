"""
Prediction Script
Load trained model for single image prediction
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

from model import ThreeLayerMLP


class Predictor:
    """Predictor class"""

    def __init__(self, model_path):
        """
        Initialize predictor

        Args:
            model_path: Model weights path
        """
        self.model = ThreeLayerMLP.from_file(model_path)
        self.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        print(f"Model loaded: {model_path}")

    def preprocess_image(self, image_path, img_size=64):
        """
        Preprocess image

        Args:
            image_path: Image path
            img_size: Target size

        Returns:
            processed_image: Preprocessed image vector
        """
        img = Image.open(image_path)
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32)
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        # Flatten to 1D vector
        img_array = img_array.flatten()
        # Add batch dimension
        img_array = img_array.reshape(1, -1)
        return img_array

    def predict(self, image_path):
        """
        Predict single image

        Args:
            image_path: Image path

        Returns:
            result: Prediction result dict
        """
        # Preprocess
        img = self.preprocess_image(image_path)

        # Predict
        proba = self.model.predict_proba(img)[0]
        pred_class = np.argmax(proba)
        pred_name = self.class_names[pred_class]
        confidence = proba[pred_class]

        # Get all class probabilities
        class_probabilities = {
            name: float(prob)
            for name, prob in zip(self.class_names, proba)
        }

        result = {
            'image_path': image_path,
            'predicted_class': pred_class,
            'predicted_label': pred_name,
            'confidence': float(confidence),
            'all_probabilities': class_probabilities
        }

        return result

    def predict_batch(self, image_paths):
        """
        Batch predict multiple images

        Args:
            image_paths: Image path list

        Returns:
            results: Prediction result list
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"Prediction failed {path}: {e}")
        return results

    def print_result(self, result):
        """Print prediction result"""
        print("\n" + "=" * 50)
        print(f"Image path: {result['image_path']}")
        print(f"Predicted class: {result['predicted_label']} (ID: {result['predicted_class']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 50)
        print("Class probabilities:")
        for name, prob in sorted(result['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True):
            bar = '#' * int(prob * 30)
            print(f"  {name:<20}: {prob:.4f} {bar}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Predict using trained model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Model weights path')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Single image path')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Image directory (predict all images in directory)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output result file (JSON)')

    args = parser.parse_args()

    # Initialize predictor
    predictor = Predictor(args.model_path)

    # Predict single image
    if args.image_path:
        result = predictor.predict(args.image_path)
        predictor.print_result(result)

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to {args.output}")

    # Predict all images in directory
    elif args.image_dir:
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        for f in os.listdir(args.image_dir):
            if os.path.splitext(f)[1].lower() in image_extensions:
                image_paths.append(os.path.join(args.image_dir, f))

        print(f"Found {len(image_paths)} images")

        results = predictor.predict_batch(image_paths)

        # Print summary
        correct = 0
        total = len(results)
        for result in results:
            predictor.print_result(result)

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    else:
        print("Please specify --image_path or --image_dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
