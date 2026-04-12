#!/bin/bash
# 训练脚本 - 超参数网格搜索
# 使用方法: ./scripts/train_grid_search.sh

# 设置路径
SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/grid_search_${TIMESTAMP}.log"

echo "========================================"
echo "超参数网格搜索脚本"
echo "========================================"
echo "代码目录: $SRC_DIR"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "========================================"

# 切换到源代码目录
cd "$SRC_DIR"

# 运行网格搜索训练
python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ./output \
    --search_hyperparams \
    --search_type grid \
    --search_epochs 30 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_sizes 128 256 512 \
    --learning_rates 0.001 0.01 0.05 \
    --activations relu tanh \
    --weight_decays 0.0001 0.001 0.01 \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "网格搜索完成!"
echo "结束时间: $(date)"
echo "日志保存在: $LOG_FILE"
echo "========================================"
