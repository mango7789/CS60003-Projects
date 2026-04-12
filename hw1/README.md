# HW1: Three-Layer Neural Network for Land Cover Classification

从零实现三层全连接神经网络 (MLP)，用于 EuroSAT 卫星图像分类。

## 项目结构

```
hw1/
├── EuroSAT_RGB/                    # 数据集目录
├── src/                            # 源代码目录
│   ├── data_loader.py              # 数据加载与预处理
│   ├── activations.py              # 激活函数
│   ├── losses.py                   # 损失函数
│   ├── optimizers.py               # 优化器
│   ├── model.py                    # 三层神经网络模型
│   ├── trainer.py                  # 训练器
│   ├── hyperparameter_search.py    # 超参数搜索
│   ├── evaluation.py               # 评估模块
│   ├── visualization.py            # 可视化模块
│   ├── train.py                    # 主训练脚本
│   ├── test.py                     # 测试脚本
│   └── predict.py                  # 预测脚本
├── scripts/                        # 训练脚本目录
│   ├── train_basic.sh              # 基础训练
│   ├── train_quick.sh              # 快速训练
│   ├── train_grid_search.sh        # 网格搜索
│   ├── train_random_search.sh      # 随机搜索
│   ├── exp_activation.sh           # 实验：激活函数对比
│   ├── exp_hidden_size.sh          # 实验：隐藏层大小对比
│   ├── exp_learning_rate.sh        # 实验：学习率对比
│   ├── exp_weight_decay.sh         # 实验：权重衰减对比
│   ├── exp_lr_decay.sh             # 实验：学习率衰减策略对比
│   ├── run_all_experiments.sh      # 运行所有实验
│   └── test.sh                     # 测试评估
├── report/                         # 报告目录
└── logs/                           # 训练日志目录
```

## 环境依赖

```bash
pip install numpy pillow scikit-learn matplotlib seaborn tqdm
```

---

## 实验方案

### 实验一：激活函数对比

对比 ReLU、Sigmoid、Tanh 三种激活函数的效果。

```bash
./scripts/exp_activation.sh
```

| 固定参数 | 值 |
|---------|-----|
| Hidden Size | 256 |
| Learning Rate | 0.01 |
| Weight Decay | 0.0001 |
| Epochs | 80 |

| 变量 | 取值 |
|-----|------|
| Activation | relu, sigmoid, tanh |

---

### 实验二：隐藏层大小对比

对比不同隐藏层大小对模型性能的影响。

```bash
./scripts/exp_hidden_size.sh
```

| 固定参数 | 值 |
|---------|-----|
| Activation | relu |
| Learning Rate | 0.01 |
| Weight Decay | 0.0001 |
| Epochs | 80 |

| 变量 | 取值 |
|-----|------|
| Hidden Size | 64, 128, 256, 512 |

---

### 实验三：学习率对比

对比不同学习率对训练效果的影响。

```bash
./scripts/exp_learning_rate.sh
```

| 固定参数 | 值 |
|---------|-----|
| Activation | relu |
| Hidden Size | 256 |
| Weight Decay | 0.0001 |
| Epochs | 80 |

| 变量 | 取值 |
|-----|------|
| Learning Rate | 0.001, 0.005, 0.01, 0.05, 0.1 |

---

### 实验四：权重衰减对比

对比不同 L2 正则化强度的影响。

```bash
./scripts/exp_weight_decay.sh
```

| 固定参数 | 值 |
|---------|-----|
| Activation | relu |
| Hidden Size | 256 |
| Learning Rate | 0.01 |
| Epochs | 80 |

| 变量 | 取值 |
|-----|------|
| Weight Decay | 0, 0.00001, 0.0001, 0.001, 0.01 |

---

### 实验五：学习率衰减策略对比

对比不同学习率衰减策略的效果。

```bash
./scripts/exp_lr_decay.sh
```

| 固定参数 | 值 |
|---------|-----|
| Activation | relu |
| Hidden Size | 256 |
| Learning Rate | 0.01 |
| Weight Decay | 0.0001 |
| Epochs | 80 |

| 变量 | 取值 |
|-----|------|
| LR Decay Type | step, exponential, cosine, linear |

---

### 实验六：网格搜索

在多个超参数空间中搜索最优组合。

```bash
./scripts/train_grid_search.sh
```

搜索空间：
- Hidden Size: [128, 256, 512]
- Learning Rate: [0.001, 0.01, 0.05]
- Activation: [relu, tanh]
- Weight Decay: [0.0001, 0.001, 0.01]

---

## 运行所有实验

```bash
# 运行所有对比实验（耗时较长）
./scripts/run_all_experiments.sh

# 或者单独运行某个实验
./scripts/exp_activation.sh      # 激活函数对比
./scripts/exp_hidden_size.sh     # 隐藏层大小对比
./scripts/exp_learning_rate.sh   # 学习率对比
./scripts/exp_weight_decay.sh    # 权重衰减对比
./scripts/exp_lr_decay.sh        # 学习率衰减策略对比
```

---

## 快速开始

### 1. 快速测试（调试用）

```bash
./scripts/train_quick.sh
```

### 2. 基础训练

```bash
./scripts/train_basic.sh
```

### 3. 测试评估

```bash
./scripts/test.sh ./src/output/experiment_YYYYMMDD_HHMMSS
```

---

## 输出结果

每个实验会在 `src/output/` 下生成对应的实验目录：

```
output/experiment_YYYYMMDD_HHMMSS/
├── config.json                 # 训练配置
├── final_results.json          # 最终结果
├── checkpoints/
│   ├── best_model.pkl          # 最优模型权重
│   └── history.json            # 训练历史
├── figures/
│   ├── training_curves.png     # 训练曲线
│   ├── first_layer_weights.png # 权重可视化
│   └── weight_patterns.png     # 权重模式分析
└── evaluation/
    ├── confusion_matrix.png    # 混淆矩阵
    └── error_analysis/         # 错例分析
```

---

## 报告需要填写的实验结果

实验完成后，需要从日志中提取以下数据填入报告：

1. **各激活函数对比结果**
2. **各隐藏层大小对比结果**
3. **各学习率对比结果**
4. **各权重衰减对比结果**
5. **各学习率衰减策略对比结果**
6. **网格搜索最优参数**
7. **最终测试集准确率**
8. **各类别准确率**
9. **混淆矩阵**
10. **错例分析**

---

## 作业要求完成情况

- [x] 自主实现自动微分与反向传播
- [x] 数据加载与预处理模块
- [x] 模型定义（支持自定义隐藏层大小）
- [x] 支持多种激活函数 (ReLU, Sigmoid, Tanh)
- [x] 训练循环
- [x] SGD 优化器
- [x] 学习率衰减策略
- [x] 交叉熵损失函数
- [x] L2 正则化
- [x] 基于验证集保存最优模型
- [x] 超参数搜索（网格搜索/随机搜索）
- [x] 测试集评估
- [x] 混淆矩阵
- [x] 训练曲线可视化
- [x] 权重可视化
- [x] 错例分析