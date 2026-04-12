"""
主训练脚本
完整的训练流程：数据加载、模型训练、超参数搜索、评估
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
    """设置日志"""
    log_file = os.path.join(exp_dir, 'training.log')

    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 文件 handler - 记录所有信息
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 控制台 handler - 只显示重要信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main(args):
    """主函数"""
    # 设置随机种子
    np.random.seed(args.seed)

    # 创建实验目录
    if args.exp_name:
        exp_dir = os.path.join(args.output_dir, args.exp_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.join(args.output_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 设置日志
    logger = setup_logging(exp_dir)

    # 同时输出实验目录路径，方便 shell 脚本捕获
    print(f"EXP_DIR:{exp_dir}")

    # 保存配置
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 60)
    logger.info("三层神经网络 - EuroSAT 图像分类")
    logger.info("=" * 60)
    logger.info(f"实验目录: {exp_dir}")
    logger.debug(f"配置: {config}")
    logger.info("=" * 60)

    # ========== 数据加载 ==========
    logger.info("\n[1/5] 数据加载...")
    data_loader = DataLoader(args.data_dir, img_size=64)

    processed_data_dir = os.path.join(exp_dir, 'processed_data')
    if os.path.exists(os.path.join(processed_data_dir, 'processed_data.npz')):
        logger.info("加载预处理数据...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_processed_data(processed_data_dir)
    else:
        logger.info("加载原始数据并处理...")
        images, labels = data_loader.load_data()
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(images, labels)
        data_loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, processed_data_dir)

    logger.info(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # ========== 超参数搜索 ==========
    if args.search_hyperparams:
        logger.info("\n[2/5] 超参数搜索...")
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

        # 更新配置
        args.hidden_size = best_params['hidden_size']
        args.learning_rate = best_params['learning_rate']
        args.activation = best_params['activation']
        args.weight_decay = best_params['weight_decay']

        logger.info(f"\n最优超参数: {best_params}")
        logger.info(f"最优验证准确率: {best_acc:.4f}")

    # ========== 模型训练 ==========
    logger.info("\n[3/5] 模型训练...")
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
        'lr_decay_steps': args.lr_decay_steps
    }

    model, history = train_model(train_config, X_train, y_train, X_val, y_val, checkpoint_dir, logger)

    # ========== 可视化 ==========
    logger.info("\n[4/5] 可视化...")
    figures_dir = os.path.join(exp_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 训练曲线
    plot_training_curves(os.path.join(checkpoint_dir, 'history.json'), figures_dir)

    # 权重可视化
    visualize_first_layer_weights(os.path.join(checkpoint_dir, 'best_model.pkl'), figures_dir)
    visualize_weight_patterns(os.path.join(checkpoint_dir, 'best_model.pkl'), figures_dir)

    # ========== 测试评估 ==========
    logger.info("\n[5/5] 测试评估...")
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

    # 保存最终结果
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
    logger.info("训练完成!")
    logger.info("=" * 60)
    logger.info(f"测试集准确率: {results['accuracy']:.4f}")
    logger.info(f"最优验证准确率: {max(history['val_acc']):.4f}")
    logger.info(f"实验结果保存在: {exp_dir}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='三层神经网络训练脚本')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB',
                        help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='输出目录')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称 (可选)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='隐藏层大小')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='激活函数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='动量系数')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='L2 正则化系数')

    # 学习率衰减
    parser.add_argument('--lr_decay_type', type=str, default='step',
                        choices=['step', 'exponential', 'cosine', 'linear'],
                        help='学习率衰减类型')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5,
                        help='学习率衰减率')
    parser.add_argument('--lr_decay_steps', type=int, default=30,
                        help='学习率衰减步数')

    # 超参数搜索
    parser.add_argument('--search_hyperparams', action='store_true',
                        help='是否进行超参数搜索')
    parser.add_argument('--search_type', type=str, default='grid',
                        choices=['grid', 'random'],
                        help='搜索类型')
    parser.add_argument('--search_epochs', type=int, default=30,
                        help='搜索时训练轮数')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='随机搜索次数')

    # 网格搜索参数
    parser.add_argument('--hidden_sizes', type=int, nargs='+',
                        default=[128, 256, 512],
                        help='隐藏层大小列表')
    parser.add_argument('--learning_rates', type=float, nargs='+',
                        default=[0.001, 0.01, 0.1],
                        help='学习率列表')
    parser.add_argument('--activations', type=str, nargs='+',
                        default=['relu', 'tanh'],
                        help='激活函数列表')
    parser.add_argument('--weight_decays', type=float, nargs='+',
                        default=[0.0001, 0.001, 0.01],
                        help='权重衰减列表')

    args = parser.parse_args()
    main(args)
