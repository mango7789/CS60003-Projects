#!/bin/bash
# 快速训练脚本 - 用于调试和快速验证

echo "========================================"
echo "快速训练脚本 (调试用)"
echo "开始时间: $(date)"
echo "========================================"

cd src

python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ../output \
    --epochs 10 \
    --batch_size 128 \
    --hidden_size 256 \
    --learning_rate 0.001 \
    --activation relu \
    --momentum 0.99 \
    --weight_decay 0.0001 \
    --lr_decay_type step \
    --lr_decay_rate 1 \
    --lr_decay_steps 1000

echo ""
echo "========================================"
echo "快速训练完成!"
echo "结束时间: $(date)"
echo "========================================"
