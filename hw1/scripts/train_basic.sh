#!/bin/bash
# 训练脚本 - 基础训练

echo "========================================"
echo "基础训练脚本"
echo "开始时间: $(date)"
echo "========================================"

cd src

python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ../output \
    --epochs 100 \
    --batch_size 64 \
    --hidden_size 256 \
    --learning_rate 0.01 \
    --activation relu \
    --momentum 0.9 \
    --weight_decay 0.0001 \
    --lr_decay_type step \
    --lr_decay_rate 0.5 \
    --lr_decay_steps 30

echo ""
echo "========================================"
echo "训练完成!"
echo "结束时间: $(date)"
echo "========================================"
