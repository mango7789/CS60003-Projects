# 深度学习作业一实验报告

## 三层全连接神经网络实现与EuroSAT卫星图像分类

**姓名**: [填写姓名]  
**学号**: [填写学号]  
**日期**: 2026年4月

---

## 1. 实验目的

1. 从零实现三层全连接神经网络（MLP），理解神经网络的核心原理
2. 实现前向传播、反向传播、激活函数、损失函数等核心组件
3. 在EuroSAT卫星图像数据集上进行土地覆盖分类任务
4. 通过超参数搜索实验，分析不同超参数对模型性能的影响

---

## 2. 数据集介绍

### 2.1 EuroSAT数据集

EuroSAT是一个基于Sentinel-2卫星图像的土地覆盖分类数据集，包含10个类别：

| 类别编号 | 类别名称 | 说明 |
|---------|---------|------|
| 0 | AnnualCrop | 年生作物 |
| 1 | Forest | 森林 |
| 2 | HerbaceousVegetation | 草本植被 |
| 3 | Highway | 高速公路 |
| 4 | Industrial | 工业区 |
| 5 | Pasture | 牧场 |
| 6 | PermanentCrop | 多年生作物 |
| 7 | Residential | 住宅区 |
| 8 | River | 河流 |
| 9 | SeaLake | 海洋湖泊 |

### 2.2 数据预处理

- **图像尺寸**: 64×64×3 RGB图像
- **输入维度**: 64×64×3 = 12288（展平为一维向量）
- **数据划分**: 训练集70%、验证集15%、测试集15%
- **归一化**: 像素值除以255归一化到[0,1]区间

---

## 3. 模型实现

### 3.1 网络结构

三层全连接神经网络结构：

```
输入层 (12288) → 隐藏层1 (256) → 隐藏层2 (256) → 输出层 (10)
```

数学表达式：
$$
\begin{aligned}
z_1 &= xW_1 + b_1, \quad a_1 = \sigma(z_1) \\
z_2 &= a_1W_2 + b_2, \quad a_2 = \sigma(z_2) \\
z_3 &= a_2W_3 + b_3, \quad \hat{y} = \text{softmax}(z_3)
\end{aligned}
$$

### 3.2 激活函数

实现了三种激活函数：

**ReLU**:
$$\sigma(x) = \max(0, x)$$
$$\sigma'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**Sigmoid**:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Tanh**:
$$\sigma(x) = \tanh(x)$$
$$\sigma'(x) = 1 - \tanh^2(x)$$

### 3.3 损失函数

交叉熵损失函数（带L2正则化）：
$$L = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}y_{ic}\log(\hat{y}_{ic}) + \frac{\lambda}{2}\sum_{k}||W_k||^2$$

### 3.4 反向传播

通过链式法则计算梯度：

$$
\begin{aligned}
\frac{\partial L}{\partial z_3} &= \hat{y} - y \\
\frac{\partial L}{\partial W_3} &= a_2^T \frac{\partial L}{\partial z_3} + \lambda W_3 \\
\frac{\partial L}{\partial z_2} &= \frac{\partial L}{\partial z_3}W_3^T \odot \sigma'(z_2) \\
\frac{\partial L}{\partial W_2} &= a_1^T \frac{\partial L}{\partial z_2} + \lambda W_2 \\
\frac{\partial L}{\partial z_1} &= \frac{\partial L}{\partial z_2}W_2^T \odot \sigma'(z_1) \\
\frac{\partial L}{\partial W_1} &= x^T \frac{\partial L}{\partial z_1} + \lambda W_1
\end{aligned}
$$

### 3.5 优化器

实现SGD优化器（带动量和梯度裁剪）：
$$
\begin{aligned}
v_t &= \gamma v_{t-1} + \eta \nabla L \\
W_t &= W_{t-1} - v_t
\end{aligned}
$$

梯度裁剪（防止梯度爆炸）：
$$\nabla L = \nabla L \cdot \min\left(1, \frac{c}{||\nabla L||}\right)$$

### 3.6 权重初始化

采用He初始化（针对ReLU）和Xavier初始化（针对Sigmoid/Tanh）：
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right) \text{ (He)}$$
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{in}}}\right) \text{ (Xavier)}$$

---

## 4. 实验设计与结果

### 4.1 实验一：激活函数对比

**固定参数**: Hidden Size=256, Learning Rate=0.01, Weight Decay=0.0001, Epochs=80

| 激活函数 | 验证准确率 | 测试准确率 | 最佳Epoch |
|---------|-----------|-----------|----------|
| ReLU | 61.12% | 60.62% | 80 |
| Sigmoid | 62.40% | 62.81% | 72 |
| Tanh | 52.16% | 50.79% | 75 |

**分析**: 
- Sigmoid在本实验中表现最好，可能是因为卫星图像数据特征分布更适合Sigmoid的非线性变换
- Tanh表现较差，可能是因为其输出范围[-1,1]与后续层不匹配
- ReLU表现中等，但训练速度较快

### 4.2 实验二：隐藏层大小对比

**固定参数**: Activation=ReLU, Learning Rate=0.01, Weight Decay=0.0001, Epochs=80

| 隐藏层大小 | 验证准确率 | 测试准确率 | 参数量 |
|-----------|-----------|-----------|-------|
| 64 | 53.59% | 53.04% | ~1.6M |
| 128 | 61.34% | 59.98% | ~3.2M |
| 256 | 61.12% | 60.62% | ~6.4M |
| 512 | 62.87% | 62.44% | ~12.8M |

**分析**:
- 随着隐藏层增大，模型容量增加，性能提升
- 512维隐藏层表现最好，但参数量较大
- 256维是较好的折中选择

### 4.3 实验三：学习率对比

**固定参数**: Activation=ReLU, Hidden Size=256, Weight Decay=0.0001, Epochs=80

| 学习率 | 验证准确率 | 测试准确率 | 最佳Epoch |
|-------|-----------|-----------|----------|
| 0.001 | 62.18% | 61.63% | 77 |
| 0.005 | **65.93%** | **66.40%** | 77 |
| 0.01 | 61.12% | 60.62% | 80 |
| 0.05 | 11.13% | 11.11% | 5 |
| 0.1 | 11.13% | 11.11% | 3 |

**分析**:
- 学习率0.005表现最好，达到66.40%测试准确率
- 学习率过小(0.001)收敛较慢，未充分训练
- 学习率过大(0.05, 0.1)导致梯度爆炸，模型无法学习
- 最佳学习率在0.005附近

### 4.4 实验四：权重衰减对比

**固定参数**: Activation=ReLU, Hidden Size=256, Learning Rate=0.01, Epochs=80

| Weight Decay | 验证准确率 | 测试准确率 |
|-------------|-----------|-----------|
| 0 | 60.82% | 60.89% |
| 0.00001 | 61.24% | 60.77% |
| 0.0001 | 61.12% | 60.62% |
| 0.001 | 62.40% | 61.43% |
| 0.01 | **63.17%** | **62.62%** |

**分析**:
- 适当的L2正则化可以防止过拟合，提升泛化能力
- Weight Decay=0.01时表现最好
- 正则化强度过大可能会限制模型学习能力

### 4.5 最佳超参数组合

综合以上实验，最佳超参数组合为：

| 超参数 | 值 |
|-------|-----|
| Hidden Size | 256 |
| Learning Rate | 0.005 |
| Activation | ReLU |
| Weight Decay | 0.0001 |
| Momentum | 0.9 |
| Batch Size | 64 |
| Epochs | 80 |

**最终结果**: 测试准确率 **66.40%**，验证准确率 **65.93%**

---

## 5. 结果分析

### 5.1 训练曲线分析

![训练曲线](output/experiment_20260412_123429/figures/training_curves.png)

从训练曲线可以看出：
1. 训练损失持续下降，训练准确率持续上升
2. 验证损失在前期快速下降，后期趋于稳定
3. 学习率在第20、40、60个epoch进行衰减，每次衰减后损失有明显下降
4. 训练和验证曲线差距不大，未出现明显过拟合

### 5.2 混淆矩阵分析

主要混淆情况：
- **PermanentCrop → HerbaceousVegetation**: 83例（多年生作物被误判为草本植被）
- **Residential → HerbaceousVegetation**: 81例（住宅区被误判为草本植被）
- **Highway ↔ River**: 63例和57例（高速公路和河流相互混淆）

**分析**:
- 混淆主要发生在视觉特征相似的类别之间
- Highway和River都具有线性特征，容易混淆
- 植被相关类别（PermanentCrop, HerbaceousVegetation, Residential）特征重叠

### 5.3 错例分析

典型错误案例：
1. PermanentCrop误判为HerbaceousVegetation：可能因为作物与自然植被光谱特征相似
2. Highway误判为River：两者都具有细长形状特征
3. Residential误判为HerbaceousVegetation：住宅区中的绿化区域干扰分类

### 5.4 第一层权重可视化

![权重可视化](output/experiment_20260412_123429/figures/first_layer_weights.png)

第一层权重呈现出类似Gabor滤波器的特征，说明网络学习到了边缘、纹理等低层视觉特征。

---

## 6. 讨论与总结

### 6.1 实验结论

1. **激活函数选择**: Sigmoid在本任务中表现略好于ReLU，但ReLU训练效率更高
2. **网络容量**: 适当增加隐藏层大小可以提升性能，但会增加计算成本
3. **学习率**: 0.005是最佳学习率，过大会导致训练失败，过小收敛慢
4. **正则化**: 适当的L2正则化可以提升泛化能力

### 6.2 模型局限性

1. **全连接网络的局限**: 将图像展平丢失了空间结构信息
2. **准确率上限**: 约66%的准确率对于10分类任务仍有提升空间
3. **相似类别混淆**: 视觉相似类别容易混淆

### 6.3 改进方向

1. 使用卷积神经网络（CNN）保留空间结构信息
2. 数据增强（旋转、翻转、缩放等）增加训练数据多样性
3. 使用更深的网络结构
4. 尝试其他优化器（Adam、RMSprop等）

### 6.4 实现要点总结

1. **梯度裁剪**: 对于大输入维度（12288），梯度裁剪是防止梯度爆炸的关键
2. **权重初始化**: 合理的初始化对训练稳定性至关重要
3. **学习率衰减**: 线性衰减策略在本实验中表现良好
4. **数值稳定性**: Softmax计算时减去最大值防止溢出

---

## 7. 代码结构

```
hw1/
├── src/
│   ├── model.py          # 三层神经网络模型
│   ├── activations.py    # 激活函数
│   ├── losses.py         # 损失函数
│   ├── optimizers.py     # 优化器
│   ├── trainer.py        # 训练器
│   ├── data_loader.py    # 数据加载
│   ├── evaluation.py     # 评估模块
│   └── visualization.py  # 可视化
├── scripts/              # 实验脚本
└── output/               # 实验结果
```

---

## 附录：关键代码片段

### A. 前向传播

```python
def forward(self, X):
    # 第一层
    z1 = X @ self.params['W1'] + self.params['b1']
    a1 = self.activation1.forward(z1)
    
    # 第二层
    z2 = a1 @ self.params['W2'] + self.params['b2']
    a2 = self.activation2.forward(z2)
    
    # 输出层
    z3 = a2 @ self.params['W3'] + self.params['b3']
    output = self.softmax.forward(z3)
    
    return output
```

### B. 反向传播

```python
def backward(self, y_true):
    # 输出层梯度
    dz3 = self.cache['output'] - y_true
    grads['W3'] = self.cache['a2'].T @ dz3
    grads['b3'] = np.sum(dz3, axis=0)
    
    # 第二层梯度
    da2 = dz3 @ self.params['W3'].T
    dz2 = self.activation2.backward(da2)
    grads['W2'] = self.cache['a1'].T @ dz2
    grads['b2'] = np.sum(dz2, axis=0)
    
    # 第一层梯度
    da1 = dz2 @ self.params['W2'].T
    dz1 = self.activation1.backward(da1)
    grads['W1'] = self.cache['X'].T @ dz1
    grads['b1'] = np.sum(dz1, axis=0)
    
    # L2正则化
    if self.weight_decay > 0:
        grads['W1'] += self.weight_decay * self.params['W1']
        grads['W2'] += self.weight_decay * self.params['W2']
        grads['W3'] += self.weight_decay * self.params['W3']
    
    return grads
```

---

**实验完成日期**: 2026年4月12日
