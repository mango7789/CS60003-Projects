#!/bin/bash
# 超参数随机搜索

echo "========================================"
echo "超参数随机搜索"
echo "开始时间: $(date)"
echo "========================================"

cd src

python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ../output \
    --search_hyperparams \
    --search_type random \
    --n_trials 20 \
    --search_epochs 30 \
    --epochs 100 \
    --batch_size 64 \
    --activations relu tanh

echo ""
echo "========================================"
echo "随机搜索完成!"
echo "结束时间: $(date)"
echo "========================================"
