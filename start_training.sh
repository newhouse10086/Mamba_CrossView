#!/bin/bash

# Mamba CrossView 训练启动脚本

echo "=== Mamba CrossView 训练启动脚本 ==="

# 设置数据路径
DATA_DIR="/home/ma-user/work/Mamba_CrossView/data/University-123/train"

# 检查数据路径是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据路径不存在 $DATA_DIR"
    echo "请检查数据路径或修改此脚本中的 DATA_DIR 变量"
    exit 1
fi

echo "数据路径: $DATA_DIR"

# 选择backbone
echo "请选择backbone:"
echo "1) MAMBA-S (Vision Mamba Small)"
echo "2) VIT-S (Vision Transformer Small)"
echo "3) VAN-S (Visual Attention Network Small)"

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        BACKBONE="MAMBA-S"
        echo "选择了 Vision Mamba Small"
        ;;
    2)
        BACKBONE="VIT-S"
        echo "选择了 Vision Transformer Small"
        ;;
    3)
        BACKBONE="VAN-S"
        echo "选择了 Visual Attention Network Small"
        ;;
    *)
        echo "无效选择，使用默认的 MAMBA-S"
        BACKBONE="MAMBA-S"
        ;;
esac

# 预训练模型路径
read -p "请输入预训练模型路径 (直接回车使用随机初始化): " PRETRAIN_PATH

if [ -z "$PRETRAIN_PATH" ]; then
    echo "使用随机初始化"
    PRETRAIN_PATH=""
else
    echo "预训练模型路径: $PRETRAIN_PATH"
fi

# 其他参数
read -p "请输入学习率 (默认 0.01): " LR
LR=${LR:-0.01}

read -p "请输入GPU ID (默认 0): " GPU_ID
GPU_ID=${GPU_ID:-0}

read -p "请输入批量大小 (默认 8): " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-8}

echo ""
echo "=== 训练参数总结 ==="
echo "数据路径: $DATA_DIR"
echo "Backbone: $BACKBONE"
echo "预训练模型: ${PRETRAIN_PATH:-随机初始化}"
echo "学习率: $LR"
echo "GPU ID: $GPU_ID"
echo "批量大小: $BATCH_SIZE"
echo ""

read -p "确认开始训练? (y/N): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo "开始训练..."
    python train.py \
        --backbone $BACKBONE \
        --data_dir "$DATA_DIR" \
        --pretrain_path "$PRETRAIN_PATH" \
        --lr $LR \
        --gpu_ids $GPU_ID \
        --batchsize $BATCH_SIZE \
        --num_epochs 120
else
    echo "训练已取消"
fi 