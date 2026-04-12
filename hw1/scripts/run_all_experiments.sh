#!/bin/bash
# 运行所有对比实验

echo "========================================"
echo "运行所有对比实验"
echo "开始时间: $(date)"
echo "========================================"

# 获取脚本目录
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# 实验1: 激活函数对比
echo ""
echo ">>> 实验 1/5: 激活函数对比"
"$SCRIPTS_DIR/exp_activation.sh"

# 实验2: 隐藏层大小对比
echo ""
echo ">>> 实验 2/5: 隐藏层大小对比"
"$SCRIPTS_DIR/exp_hidden_size.sh"

# 实验3: 学习率对比
echo ""
echo ">>> 实验 3/5: 学习率对比"
"$SCRIPTS_DIR/exp_learning_rate.sh"

# 实验4: 权重衰减对比
echo ""
echo ">>> 实验 4/5: 权重衰减对比"
"$SCRIPTS_DIR/exp_weight_decay.sh"

# 实验5: 学习率衰减策略对比
echo ""
echo ">>> 实验 5/5: 学习率衰减策略对比"
"$SCRIPTS_DIR/exp_lr_decay.sh"

echo ""
echo "========================================"
echo "所有实验完成!"
echo "结束时间: $(date)"
echo "========================================"
