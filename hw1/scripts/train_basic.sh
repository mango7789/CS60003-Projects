#!/bin/bash
# 训练脚本 - 基础训练
# 使用方法: ./scripts/train_basic.sh

# 设置路径
SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_basic_${TIMESTAMP}.log"

echo "========================================"
echo "基础训练脚本"
echo "========================================"
echo "代码目录: $SRC_DIR"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "========================================"

# 切换到源代码目录
cd "$SRC_DIR"

# 运行训练
python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ./output \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 256 \
    --learning_rate 0.01 \
    --activation relu \
    --momentum 0.9 \
    --weight_decay 0.0001 \
    --lr_decay_type step \
    --lr_decay_rate 0.5 \
    --lr_decay_steps 30 \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "训练完成!"
echo "结束时间: $(date)"
echo "日志保存在: $LOG_FILE"
echo "========================================"
