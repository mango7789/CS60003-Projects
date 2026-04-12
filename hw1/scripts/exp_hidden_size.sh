#!/bin/bash
# 实验脚本 - 不同隐藏层大小对比

echo "========================================"
echo "实验: 隐藏层大小对比"
echo "开始时间: $(date)"
echo "========================================"

cd src

for HIDDEN in 64 128 256 512; do
    echo ""
    echo "----------------------------------------"
    echo "训练隐藏层大小: $HIDDEN"
    echo "----------------------------------------"

    python train.py \
        --data_dir ../EuroSAT_RGB \
        --output_dir ../output \
        --epochs 80 \
        --batch_size 64 \
        --hidden_size $HIDDEN \
        --learning_rate 0.01 \
        --activation relu \
        --momentum 0.9 \
        --weight_decay 0.0001 \
        --lr_decay_type step \
        --lr_decay_rate 0.5 \
        --lr_decay_steps 20
done

echo ""
echo "========================================"
echo "隐藏层大小对比实验完成!"
echo "结束时间: $(date)"
echo "========================================"
