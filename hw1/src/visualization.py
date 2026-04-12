"""
可视化模块
包含训练曲线、权重可视化等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


def plot_training_curves(history_path, save_dir='./figures'):
    """
    绘制训练曲线

    Args:
        history_path: 训练历史文件路径
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # 绘制 Loss 曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练集 Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='验证集 Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证 Loss 曲线', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='训练集 Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='验证集 Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('训练和验证 Accuracy 曲线', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到 {save_path}")


def visualize_first_layer_weights(model_path, save_dir='./figures', n_features=16):
    """
    可视化第一层隐藏层权重

    Args:
        model_path: 模型权重路径
        save_dir: 保存目录
        n_features: 显示的特征数量
    """
    import pickle

    os.makedirs(save_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    W1 = data['params']['W1']  # (input_size, hidden_size) = (12288, hidden_size)

    hidden_size = W1.shape[1]
    n_features = min(n_features, hidden_size)

    # 随机选择一些特征进行可视化
    indices = np.random.choice(hidden_size, n_features, replace=False)

    fig, axes = plt.subplots(4, n_features // 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        # 将权重恢复为图像形状 (64, 64, 3)
        weights = W1[:, idx].reshape(64, 64, 3)

        # 归一化到 [0, 1] 范围
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

        axes[i].imshow(weights)
        axes[i].axis('off')
        axes[i].set_title(f'特征 {idx}', fontsize=10)

    plt.suptitle('第一层隐藏层权重可视化', fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'first_layer_weights.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"权重可视化已保存到 {save_path}")


def visualize_weight_patterns(model_path, save_dir='./figures'):
    """
    分析并可视化权重模式

    Args:
        model_path: 模型权重路径
        save_dir: 保存目录
    """
    import pickle

    os.makedirs(save_dir, exist_ok=True)

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    W1 = data['params']['W1']  # (12288, hidden_size)

    # 计算每个特征的统计信息
    hidden_size = W1.shape[1]

    # 计算每个特征的颜色倾向 (RGB 通道的平均值)
    R_mean = np.mean(np.abs(W1[:4096, :]), axis=0)
    G_mean = np.mean(np.abs(W1[4096:8192, :]), axis=0)
    B_mean = np.mean(np.abs(W1[8192:, :]), axis=0)

    # 绘制颜色倾向分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 颜色倾向散点图
    axes[0].scatter(R_mean, G_mean, c='red', alpha=0.5, label='R-G')
    axes[0].set_xlabel('红色通道平均权重', fontsize=12)
    axes[0].set_ylabel('绿色通道平均权重', fontsize=12)
    axes[0].set_title('特征权重的颜色倾向', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 权重分布直方图
    axes[1].hist(W1.flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('权重值', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].set_title('第一层权重分布', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'weight_patterns.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"权重模式分析已保存到 {save_path}")


def plot_hyperparameter_comparison(results_path, save_dir='./figures'):
    """
    绘制超参数对比图

    Args:
        results_path: 超参数搜索结果路径
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(results_path, 'r') as f:
        results = json.load(f)

    # 提取验证准确率
    val_accs = [r['val_acc'] for r in results]
    params = [r['params'] for r in results]

    # 绘制准确率分布
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(val_accs))
    bars = ax.bar(x, val_accs, color='steelblue', alpha=0.7)

    # 标记最优
    best_idx = np.argmax(val_accs)
    bars[best_idx].set_color('red')

    ax.set_xlabel('实验编号', fontsize=12)
    ax.set_ylabel('验证集准确率', fontsize=12)
    ax.set_title('超参数搜索结果', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'hyperparameter_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"超参数对比图已保存到 {save_path}")


if __name__ == "__main__":
    # 测试可视化模块
    print("可视化模块测试")
