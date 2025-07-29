# 🎯 官方Vision Mamba (Vim) 使用指南

本指南介绍如何在FSRA框架中使用官方Vision Mamba实现进行跨视角行人重识别训练。

## 📋 特性介绍

### 🚀 官方Vision Mamba (VIM-TINY)
- **基于**: [官方Vision Mamba仓库](https://github.com/hustvl/Vim)
- **架构**: Bidirectional State Space Model (双向状态空间模型)
- **预训练**: 支持官方权重 `vim_t_midclstok_ft_78p3acc.pth`
- **性能**: ImageNet Top-1 准确率 78.3%
- **参数量**: ~7M (轻量高效)
- **特色**: Middle Class Token设计，更好的视觉特征提取

### 🎯 核心优势
1. **线性复杂度**: 相比ViT的O(n²)，Vision Mamba为O(n)
2. **双向建模**: 前向和后向状态空间扫描，捕获更丰富的上下文
3. **官方实现**: 基于论文作者的官方代码，保证准确性
4. **预训练权重**: 支持加载官方ImageNet预训练模型

## 🛠️ 安装依赖

```bash
# 可选：安装mamba_ssm获得最佳性能
pip install mamba-ssm

# 如果无法安装mamba_ssm，代码会自动使用fallback实现
```

## 📖 使用方法

### 1. 快速开始 - VIM-TINY (推荐)

```bash
# 使用官方预训练权重训练
python train.py \
    --backbone VIM-TINY \
    --pretrain_path vim_t_midclstok_ft_78p3acc.pth \
    --lr 0.0005 \
    --optimizer adamw \
    --batchsize 16 \
    --gpu_ids 0 \
    --data_dir data/University-123/train
```

### 2. 高性能配置 - VIM-SMALL

```bash
# 使用更大的模型获得更好性能
python train.py \
    --backbone VIM-SMALL \
    --pretrain_path vim_t_midclstok_ft_78p3acc.pth \
    --lr 0.0003 \
    --optimizer adamw \
    --batchsize 12 \
    --gpu_ids 0 \
    --data_dir data/University-123/train
```

### 3. 快速实验 - 无预训练权重

```bash
# 从随机初始化开始训练（用于实验）
python train.py \
    --backbone VIM-TINY \
    --pretrain_path "" \
    --lr 0.001 \
    --optimizer adamw \
    --batchsize 16 \
    --gpu_ids 0 \
    --data_dir data/University-123/train
```

## ⚙️ 推荐配置

### VIM-TINY (推荐用于生产)
```bash
backbone: VIM-TINY
lr: 0.0005
optimizer: adamw  
batch_size: 16
epochs: 120
预训练: vim_t_midclstok_ft_78p3acc.pth
```

### VIM-SMALL (推荐用于追求最佳性能)
```bash
backbone: VIM-SMALL
lr: 0.0003
optimizer: adamw
batch_size: 12  
epochs: 120
预训练: vim_t_midclstok_ft_78p3acc.pth
```

## 🧪 测试和验证

### 1. 运行测试脚本
```bash
# 测试模型实现是否正确
python test_vim_official.py
```

### 2. 检查权重加载
```bash
# 确保vim_t_midclstok_ft_78p3acc.pth在项目根目录
ls -la vim_t_midclstok_ft_78p3acc.pth
```

### 3. 验证数据路径
```bash
# 确保数据集路径正确
ls data/University-123/train/
# 应该看到: drone/ google/ satellite/ street/
```

## 📊 性能对比

| 模型 | 参数量 | ImageNet Top-1 | 训练速度 | 收敛难度 | 推荐指数 |
|------|--------|----------------|----------|----------|----------|
| **VIM-TINY** | ~7M | **78.3%** | 快 | 简单 | ⭐⭐⭐⭐⭐ |
| VIM-SMALL | ~22M | ~81.0% | 中等 | 简单 | ⭐⭐⭐⭐ |
| ViT-Small | ~22M | 75.2% | 中等 | 简单 | ⭐⭐⭐ |
| MAMBA-V2 | ~22M | 73-77% | 慢 | 中等 | ⭐⭐ |

## 🚨 常见问题

### Q1: 提示"mamba_ssm not available"
**A**: 这是正常的，代码会自动使用fallback实现。如需最佳性能，可安装mamba-ssm。

### Q2: 预训练权重加载失败
**A**: 
1. 确保`vim_t_midclstok_ft_78p3acc.pth`在项目根目录
2. 检查文件是否完整（应该约111MB）
3. 尝试重新下载权重文件

### Q3: 显存不足
**A**: 
1. 减少batch_size（推荐：16→8→4）
2. 使用VIM-TINY替代VIM-SMALL
3. 启用混合精度：`--autocast`

### Q4: 训练不收敛
**A**:
1. 检查学习率是否过大（VIM推荐0.0003-0.0005）
2. 使用warmup：`--warm_epoch 5`
3. 确保使用adamw优化器

## 🔧 高级配置

### 自定义学习率调度
```bash
python train.py \
    --backbone VIM-TINY \
    --lr 0.0005 \
    --steps [40,80] \
    --num_epochs 120
```

### 启用混合精度训练
```bash
python train.py \
    --backbone VIM-TINY \
    --autocast \
    --batchsize 32  # 可以使用更大batch size
```

### 多GPU训练
```bash
python train.py \
    --backbone VIM-TINY \
    --gpu_ids 0,1,2,3 \
    --batchsize 64  # 总batch size跨所有GPU
```

## 📈 训练监控

训练过程中会显示：
- **Satellite_Acc**: 卫星视图准确率
- **Drone_Acc**: 无人机视图准确率  
- **Loss**: 总损失（分类+三元组+KL）
- **lr_backbone**: backbone学习率
- **lr_other**: 分类器学习率

## 🎯 预期结果

使用VIM-TINY + 官方预训练权重，在University-1652数据集上预期结果：
- **训练准确率**: 85-90%
- **收敛轮数**: 80-100轮
- **最佳性能**: 通常在60-80轮达到
- **训练时间**: 约2-4小时（单GPU V100）

## 🤝 技术支持

如遇到问题，请检查：
1. ✅ 运行 `python test_vim_official.py` 通过
2. ✅ 数据路径正确
3. ✅ 预训练权重存在
4. ✅ CUDA和PyTorch版本兼容

---

💡 **提示**: VIM-TINY是当前最推荐的选择，它在性能、速度和稳定性之间达到了最佳平衡。 