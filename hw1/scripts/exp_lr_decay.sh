#!/bin/bash
# 实验脚本 - 学习率衰减策略对比
# 使用方法: ./scripts/exp_lr_decay.sh

SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "实验: 学习率衰减策略对比"
echo "========================================"
echo "开始时间: $(date)"

cd "$SRC_DIR"

# 固定其他参数，对比不同学习率衰减策略
for DECAY in step exponential cosine linear; do
    echo ""
    echo "----------------------------------------"
    echo "训练学习率衰减: $DECAY"
    echo "----------------------------------------"

    LOG_FILE="$LOG_DIR/exp_decay_${DECAY}_${TIMESTAMP}.log"

    python train.py \
        --data_dir ../EuroSAT_RGB \
        --output_dir ./output \
        --epochs 80 \
        --batch_size 64 \
        --hidden_size 256 \
        --learning_rate 0.01 \
        --activation relu \
        --momentum 0.9 \
        --weight_decay 0.0001 \
        --lr_decay_type $DECAY \
        --lr_decay_rate 0.5 \
        --lr_decay_steps 20 \
        2>&1 | tee "$LOG_FILE"
done

echo ""
echo "========================================"
echo "学习率衰减策略对比实验完成!"
echo "结束时间: $(date)"
echo "========================================"
