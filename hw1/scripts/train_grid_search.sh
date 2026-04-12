#!/bin/bash
# 超参数网格搜索

echo "========================================"
echo "超参数网格搜索"
echo "开始时间: $(date)"
echo "========================================"

cd src

python train.py \
    --data_dir ../EuroSAT_RGB \
    --output_dir ../output \
    --search_hyperparams \
    --search_type grid \
    --search_epochs 30 \
    --epochs 100 \
    --batch_size 64 \
    --hidden_sizes 128 256 512 \
    --learning_rates 0.001 0.01 0.05 \
    --activations relu tanh \
    --weight_decays 0.0001 0.001 0.01

echo ""
echo "========================================"
echo "网格搜索完成!"
echo "结束时间: $(date)"
echo "========================================"
