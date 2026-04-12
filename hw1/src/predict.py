"""
预测脚本
加载训练好的模型进行单张图像预测
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

from model import ThreeLayerMLP


class Predictor:
    """预测器类"""

    def __init__(self, model_path):
        """
        初始化预测器

        Args:
            model_path: 模型权重路径
        """
        self.model = ThreeLayerMLP.from_file(model_path)
        self.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        print(f"模型加载成功: {model_path}")

    def preprocess_image(self, image_path, img_size=64):
        """
        预处理图像

        Args:
            image_path: 图像路径
            img_size: 目标尺寸

        Returns:
            processed_image: 预处理后的图像向量
        """
        img = Image.open(image_path)
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32)
        # 归一化到 [0, 1]
        img_array = img_array / 255.0
        # 展平为一维向量
        img_array = img_array.flatten()
        # 添加 batch 维度
        img_array = img_array.reshape(1, -1)
        return img_array

    def predict(self, image_path):
        """
        预测单张图像

        Args:
            image_path: 图像路径

        Returns:
            result: 预测结果字典
        """
        # 预处理
        img = self.preprocess_image(image_path)

        # 预测
        proba = self.model.predict_proba(img)[0]
        pred_class = np.argmax(proba)
        pred_name = self.class_names[pred_class]
        confidence = proba[pred_class]

        # 获取所有类别的概率
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
        批量预测多张图像

        Args:
            image_paths: 图像路径列表

        Returns:
            results: 预测结果列表
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"预测失败 {path}: {e}")
        return results

    def print_result(self, result):
        """打印预测结果"""
        print("\n" + "=" * 50)
        print(f"图像路径: {result['image_path']}")
        print(f"预测类别: {result['predicted_label']} (ID: {result['predicted_class']})")
        print(f"置信度: {result['confidence']:.4f}")
        print("-" * 50)
        print("各类别概率:")
        for name, prob in sorted(result['all_probabilities'].items(),
                                  key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 30)
            print(f"  {name:<20}: {prob:.4f} {bar}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')

    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--image_path', type=str, default=None,
                        help='单张图像路径')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='图像目录 (预测目录下所有图像)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果文件 (JSON)')

    args = parser.parse_args()

    # 初始化预测器
    predictor = Predictor(args.model_path)

    # 预测单张图像
    if args.image_path:
        result = predictor.predict(args.image_path)
        predictor.print_result(result)

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n结果已保存到 {args.output}")

    # 预测目录下所有图像
    elif args.image_dir:
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        for f in os.listdir(args.image_dir):
            if os.path.splitext(f)[1].lower() in image_extensions:
                image_paths.append(os.path.join(args.image_dir, f))

        print(f"找到 {len(image_paths)} 张图像")

        results = predictor.predict_batch(image_paths)

        # 打印摘要
        correct = 0
        total = len(results)
        for result in results:
            predictor.print_result(result)

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n结果已保存到 {args.output}")

    else:
        print("请指定 --image_path 或 --image_dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
