#!/bin/bash
# 快速训练脚本 - 用于调试和快速验证
# 使用方法: ./scripts/train_quick.sh

# 设置路径
SRC_DIR="src"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_quick_${TIMESTAMP}.log"

echo "========================================"
echo "快速训练脚本 (调试用)"
echo "========================================"
echo "代码目录: $SRC_DIR"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "========================================"

# 切换到源代码目录
cd "$SRC_DIR"

# 运行快速训练 (减少 epochs)
python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ./output \
    --epochs 20 \
    --batch_size 128 \
    --hidden_size 128 \
    --learning_rate 0.01 \
    --activation relu \
    --momentum 0.9 \
    --weight_decay 0.0001 \
    --lr_decay_type step \
    --lr_decay_rate 0.5 \
    --lr_decay_steps 10 \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "快速训练完成!"
echo "结束时间: $(date)"
echo "日志保存在: $LOG_FILE"
echo "========================================"
