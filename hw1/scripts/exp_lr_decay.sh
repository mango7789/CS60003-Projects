#!/bin/bash
# 实验脚本 - 学习率衰减策略对比

echo "========================================"
echo "实验: 学习率衰减策略对比"
echo "开始时间: $(date)"
echo "========================================"

cd src

for DECAY in step exponential cosine linear; do
    echo ""
    echo "----------------------------------------"
    echo "训练学习率衰减: $DECAY"
    echo "----------------------------------------"

    python train.py \
        --data_dir ../EuroSAT_RGB \
        --output_dir ../output \
        --epochs 80 \
        --batch_size 64 \
        --hidden_size 256 \
        --learning_rate 0.01 \
        --activation relu \
        --momentum 0.9 \
        --weight_decay 0.0001 \
        --lr_decay_type $DECAY \
        --lr_decay_rate 0.5 \
        --lr_decay_steps 20
done

echo ""
echo "========================================"
echo "学习率衰减策略对比实验完成!"
echo "结束时间: $(date)"
echo "========================================"
