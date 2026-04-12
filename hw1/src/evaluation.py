"""
测试与评估模块
包含准确率计算、混淆矩阵、错例分析等功能
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model import ThreeLayerMLP


class Evaluator:
    """评估器类"""

    def __init__(self, model, class_names):
        """
        初始化评估器

        Args:
            model: 训练好的模型
            class_names: 类别名称列表
        """
        self.model = model
        self.class_names = class_names

    def evaluate(self, X_test, y_test, batch_size=256):
        """
        评估模型在测试集上的性能

        Args:
            X_test: 测试数据
            y_test: 测试标签
            batch_size: 批次大小

        Returns:
            accuracy: 准确率
            predictions: 预测结果
        """
        predictions = self.model.predict(X_test)
        accuracy = np.mean(predictions == y_test)

        print(f"测试集准确率: {accuracy:.4f} ({np.sum(predictions == y_test)}/{len(y_test)})")

        return accuracy, predictions

    def confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        计算并绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径

        Returns:
            cm: 混淆矩阵
        """
        n_classes = len(self.class_names)
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)

        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1

        # 打印混淆矩阵
        print("\n混淆矩阵:")
        print("-" * 80)
        print(f"{'真实\\预测':<20}", end="")
        for name in self.class_names:
            print(f"{name[:8]:>10}", end="")
        print()

        for i, name in enumerate(self.class_names):
            print(f"{name:<20}", end="")
            for j in range(n_classes):
                print(f"{cm[i, j]:>10}", end="")
            print()

        # 绘制混淆矩阵热力图
        plt.figure(figsize=(12, 10))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title('混淆矩阵 (归一化)')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n混淆矩阵已保存到 {save_path}")

        plt.close()

        return cm

    def per_class_accuracy(self, y_true, y_pred):
        """
        计算每个类别的准确率

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            class_acc: 各类别准确率字典
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
        错例分析

        Args:
            X_test: 测试数据
            y_test: 真实标签
            y_pred: 预测标签
            n_samples: 每个错误类型分析的样本数
            save_dir: 保存目录

        Returns:
            error_samples: 错误样本列表
        """
        os.makedirs(save_dir, exist_ok=True)

        # 找出所有错误样本
        error_mask = (y_pred != y_test)
        error_indices = np.where(error_mask)[0]

        print(f"\n错例分析:")
        print(f"总错误数: {len(error_indices)}/{len(y_test)}")
        print("-" * 60)

        # 统计各类错误
        error_pairs = {}
        for idx in error_indices:
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            pair = (true_label, pred_label)
            if pair not in error_pairs:
                error_pairs[pair] = []
            error_pairs[pair].append(idx)

        # 按错误数量排序
        sorted_errors = sorted(error_pairs.items(), key=lambda x: len(x[1]), reverse=True)

        error_samples = []

        # 分析最常见的错误类型
        for (true_label, pred_label), indices in sorted_errors[:5]:
            true_name = self.class_names[true_label]
            pred_name = self.class_names[pred_label]

            print(f"\n{true_name} -> {pred_name}: {len(indices)} 个错误")

            # 选择几个样本进行可视化
            sample_indices = indices[:n_samples]

            fig, axes = plt.subplots(1, n_samples, figsize=(3 * n_samples, 3))
            if n_samples == 1:
                axes = [axes]

            for i, sample_idx in enumerate(sample_indices):
                # 恢复图像
                img = X_test[sample_idx].reshape(64, 64, 3)
                img = np.clip(img, 0, 1)

                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'#{sample_idx}', fontsize=10)

            plt.suptitle(f'真实: {true_name}, 预测: {pred_name}', fontsize=12)
            plt.tight_layout()

            save_path = os.path.join(save_dir, f'error_{true_name}_to_{pred_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            error_samples.append({
                'true_label': true_name,
                'pred_label': pred_name,
                'count': len(indices),
                'indices': sample_indices.tolist()
            })

        # 保存错误分析报告
        import json
        report_path = os.path.join(save_dir, 'error_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(error_samples, f, indent=2, ensure_ascii=False)
        print(f"\n错误分析报告已保存到 {report_path}")

        return error_samples


def evaluate_model(model_path, X_test, y_test, class_names, save_dir='./evaluation'):
    """
    完整的模型评估流程

    Args:
        model_path: 模型权重路径
        X_test: 测试数据
        y_test: 测试标签
        class_names: 类别名称
        save_dir: 保存目录

    Returns:
        results: 评估结果
    """
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = ThreeLayerMLP.from_file(model_path)
    print(f"模型加载成功: {model_path}")

    # 创建评估器
    evaluator = Evaluator(model, class_names)

    # 评估准确率
    accuracy, predictions = evaluator.evaluate(X_test, y_test)

    # 混淆矩阵
    cm = evaluator.confusion_matrix(
        y_test, predictions,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )

    # 各类别准确率
    print("\n各类别准确率:")
    class_acc = evaluator.per_class_accuracy(y_test, predictions)

    # 错例分析
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
    # 测试评估模块
    from data_loader import DataLoader

    class_names = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]

    # 这里需要实际的测试数据
    print("评估模块测试")
