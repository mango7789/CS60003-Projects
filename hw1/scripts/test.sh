#!/bin/bash
# 测试脚本 - 评估训练好的模型
# 使用方法: ./scripts/test.sh <experiment_dir>

if [ -z "$1" ]; then
    echo "使用方法: ./scripts/test.sh <experiment_dir>"
    echo "示例: ./scripts/test.sh ../output/experiment_20260412_120000"
    exit 1
fi

EXP_DIR="$1"

echo "========================================"
echo "模型测试脚本"
echo "实验目录: $EXP_DIR"
echo "开始时间: $(date)"
echo "========================================"

cd src

python test.py \
    --data_dir ../EuroSAT_RGB \
    --exp_dir "$EXP_DIR"

echo ""
echo "========================================"
echo "测试完成!"
echo "结束时间: $(date)"
echo "========================================"
