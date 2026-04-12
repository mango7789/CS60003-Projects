#!/bin/bash
# 实验脚本 - 不同激活函数对比
# 使用方法: ./scripts/exp_activation.sh

SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
RESULT_DIR="$(cd "$(dirname "$0")/../results" && pwd)"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "实验: 激活函数对比"
echo "========================================"
echo "开始时间: $(date)"

cd "$SRC_DIR"

# 固定其他参数，对比不同激活函数
for ACT in relu sigmoid tanh; do
    echo ""
    echo "----------------------------------------"
    echo "训练激活函数: $ACT"
    echo "----------------------------------------"

    LOG_FILE="$LOG_DIR/exp_activation_${ACT}_${TIMESTAMP}.log"

    python train.py \
        --data_dir ../EuroSAT_RGB \
        --output_dir ./output \
        --epochs 80 \
        --batch_size 64 \
        --hidden_size 256 \
        --learning_rate 0.01 \
        --activation $ACT \
        --momentum 0.9 \
        --weight_decay 0.0001 \
        --lr_decay_type step \
        --lr_decay_rate 0.5 \
        --lr_decay_steps 20 \
        2>&1 | tee "$LOG_FILE"
done

echo ""
echo "========================================"
echo "激活函数对比实验完成!"
echo "结束时间: $(date)"
echo "========================================"
