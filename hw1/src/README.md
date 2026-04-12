# 三层神经网络图像分类器

从零实现三层全连接神经网络 (MLP)，用于 EuroSAT 卫星图像分类。

## 环境依赖

```bash
pip install numpy pillow scikit-learn matplotlib seaborn
```

## 项目结构

```
src/
├── data_loader.py          # 数据加载与预处理
├── activations.py          # 激活函数 (ReLU, Sigmoid, Tanh, Softmax)
├── losses.py               # 损失函数 (交叉熵, L2正则化)
├── optimizers.py           # 优化器 (SGD, 学习率衰减)
├── model.py                # 三层神经网络模型
├── trainer.py              # 训练器
├── hyperparameter_search.py # 超参数搜索 (网格/随机)
├── evaluation.py           # 评估模块 (混淆矩阵, 错例分析)
├── visualization.py        # 可视化模块
├── train.py                # 主训练脚本
└── test.py                 # 测试脚本
```

## 快速开始

### 1. 基础训练

```bash
cd src
python train.py --data_dir ../EuroSAT_RGB --epochs 100 --hidden_size 256 --learning_rate 0.01
```

### 2. 超参数搜索

```bash
# 网格搜索
python train.py --data_dir ../EuroSAT_RGB --search_hyperparams --search_type grid --epochs 100

# 随机搜索
python train.py --data_dir ../EuroSAT_RGB --search_hyperparams --search_type random --n_trials 30
```

### 3. 测试评估

```bash
python test.py --data_dir ../EuroSAT_RGB --exp_dir ./output/experiment_xxx
```

## 参数说明

### 模型参数
- `--hidden_size`: 隐藏层大小 (默认: 256)
- `--activation`: 激活函数，可选 relu/sigmoid/tanh (默认: relu)

### 训练参数
- `--epochs`: 训练轮数 (默认: 100)
- `--batch_size`: 批次大小 (默认: 64)
- `--learning_rate`: 学习率 (默认: 0.01)
- `--momentum`: 动量系数 (默认: 0.9)
- `--weight_decay`: L2正则化系数 (默认: 0.0001)

### 学习率衰减
- `--lr_decay_type`: 衰减类型 step/exponential/cosine/linear (默认: step)
- `--lr_decay_rate`: 衰减率 (默认: 0.5)
- `--lr_decay_steps`: 衰减步数 (默认: 30)

## 输出结果

训练完成后，实验目录包含：
- `checkpoints/best_model.pkl`: 最优模型权重
- `figures/training_curves.png`: 训练曲线
- `figures/first_layer_weights.png`: 第一层权重可视化
- `evaluation/confusion_matrix.png`: 混淆矩阵
- `evaluation/error_analysis/`: 错例分析
- `final_results.json`: 最终结果

## 模型权重下载

[Google Drive 链接]

## 作者

[你的名字]

## 最后更新时间

2026年4月
