#!/bin/bash
# 训练脚本 - 超参数随机搜索
# 使用方法: ./scripts/train_random_search.sh

# 设置路径
SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/random_search_${TIMESTAMP}.log"

echo "========================================"
echo "超参数随机搜索脚本"
echo "========================================"
echo "代码目录: $SRC_DIR"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "========================================"

# 切换到源代码目录
cd "$SRC_DIR"

# 运行随机搜索训练
python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ./output \
    --search_hyperparams \
    --search_type random \
    --n_trials 20 \
    --search_epochs 30 \
    --epochs 100 \
    --batch_size 64 \
    --activations relu tanh \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "随机搜索完成!"
echo "结束时间: $(date)"
echo "日志保存在: $LOG_FILE"
echo "========================================"
