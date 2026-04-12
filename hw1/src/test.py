"""
测试脚本
加载训练好的模型进行测试评估
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
    """主函数"""
    print("=" * 60)
    print("模型测试评估")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    data_loader = DataLoader(args.data_dir, img_size=64)

    processed_data_dir = os.path.join(args.exp_dir, 'processed_data')
    if os.path.exists(os.path.join(processed_data_dir, 'processed_data.npz')):
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_processed_data(processed_data_dir)
    else:
        images, labels = data_loader.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(images, labels)

    print(f"测试集: {X_test.shape}")

    # 类别名称
    class_names = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]

    # 模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(args.exp_dir, 'checkpoints', 'best_model.pkl')

    # 评估目录
    eval_dir = os.path.join(args.exp_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    # 评估模型
    results = evaluate_model(model_path, X_test, y_test, class_names, eval_dir)

    # 权重可视化
    figures_dir = os.path.join(args.exp_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    visualize_first_layer_weights(model_path, figures_dir)

    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"测试集准确率: {results['accuracy']:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型测试脚本')

    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB',
                        help='数据集目录')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径 (可选，默认使用实验目录下的最优模型)')

    args = parser.parse_args()
    main(args)
