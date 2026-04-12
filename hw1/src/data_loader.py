"""
数据加载与预处理模块
功能：加载EuroSAT数据集，进行预处理和划分
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle


class DataLoader:
    """数据加载器类"""

    def __init__(self, data_dir, img_size=64, test_ratio=0.15, val_ratio=0.15, random_state=42):
        """
        初始化数据加载器

        Args:
            data_dir: 数据集目录路径
            img_size: 图像大小
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            random_state: 随机种子
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

        # 类别名称
        self.class_names = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def load_data(self):
        """加载所有图像数据"""
        images = []
        labels = []

        print("正在加载数据集...")
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 目录 {class_dir} 不存在")
                continue

            class_idx = self.class_to_idx[class_name]
            files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]

            for file in files:
                img_path = os.path.join(class_dir, file)
                try:
                    img = Image.open(img_path)
                    img = img.resize((self.img_size, self.img_size))
                    img_array = np.array(img, dtype=np.float32)
                    # 归一化到 [0, 1]
                    img_array = img_array / 255.0
                    # 展平为一维向量 (64*64*3 = 12288)
                    img_array = img_array.flatten()
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"加载图像 {img_path} 失败: {e}")

        images = np.array(images)
        labels = np.array(labels)

        print(f"数据加载完成: {len(images)} 张图像, {len(self.class_names)} 个类别")
        print(f"每个类别的样本数: {np.bincount(labels)}")

        return images, labels

    def split_data(self, images, labels):
        """划分训练集、验证集、测试集"""
        # 首先划分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=labels
        )

        # 从剩余数据中划分验证集
        val_ratio_adjusted = self.val_ratio / (1 - self.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )

        print(f"数据集划分:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def one_hot_encode(self, labels, num_classes=10):
        """将标签转换为 one-hot 编码"""
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, save_dir):
        """保存处理后的数据"""
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_dir, 'processed_data.npz'),
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test
        )
        print(f"处理后的数据已保存到 {save_dir}")

    def load_processed_data(self, save_dir):
        """加载处理后的数据"""
        data = np.load(os.path.join(save_dir, 'processed_data.npz'))
        return (data['X_train'], data['X_val'], data['X_test'],
                data['y_train'], data['y_val'], data['y_test'])


def get_batches(X, y, batch_size, shuffle=True):
    """生成批次数据"""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


if __name__ == "__main__":
    # 测试数据加载
    data_dir = "../EuroSAT_RGB"
    loader = DataLoader(data_dir)
    images, labels = loader.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(images, labels)
    loader.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test, "./processed_data")
