#!/bin/bash
# 实验脚本 - 不同学习率对比
# 使用方法: ./scripts/exp_learning_rate.sh

SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "实验: 学习率对比"
echo "========================================"
echo "开始时间: $(date)"

cd "$SRC_DIR"

# 固定其他参数，对比不同学习率
for LR in 0.001 0.005 0.01 0.05 0.1; do
    echo ""
    echo "----------------------------------------"
    echo "训练学习率: $LR"
    echo "----------------------------------------"

    LOG_FILE="$LOG_DIR/exp_lr_${LR}_${TIMESTAMP}.log"

    python train.py \
        --data_dir ../EuroSAT_RGB \
        --output_dir ./output \
        --epochs 80 \
        --batch_size 64 \
        --hidden_size 256 \
        --learning_rate $LR \
        --activation relu \
        --momentum 0.9 \
        --weight_decay 0.0001 \
        --lr_decay_type step \
        --lr_decay_rate 0.5 \
        --lr_decay_steps 20 \
        2>&1 | tee "$LOG_FILE"
done

echo ""
echo "========================================"
echo "学习率对比实验完成!"
echo "结束时间: $(date)"
echo "========================================"
