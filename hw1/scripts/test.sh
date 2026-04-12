#!/bin/bash
# 测试脚本 - 评估训练好的模型
# 使用方法: ./scripts/test.sh <experiment_dir>

# 检查参数
if [ -z "$1" ]; then
    echo "使用方法: ./scripts/test.sh <experiment_dir>"
    echo "示例: ./scripts/test.sh ./output/experiment_20260412_120000"
    exit 1
fi

EXP_DIR="$1"

# 设置路径
SRC_DIR="$(cd "$(dirname "$0")/../src" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/test_${TIMESTAMP}.log"

echo "========================================"
echo "模型测试脚本"
echo "========================================"
echo "代码目录: $SRC_DIR"
echo "实验目录: $EXP_DIR"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date)"
echo "========================================"

# 切换到源代码目录
cd "$SRC_DIR"

# 运行测试
python test.py \
    --data_dir ../EuroSAT_RGB \
    --exp_dir "$EXP_DIR" \
    2>&1 | tee "$LOG_FILE"

echo "========================================"
echo "测试完成!"
echo "结束时间: $(date)"
echo "日志保存在: $LOG_FILE"
echo "========================================"
