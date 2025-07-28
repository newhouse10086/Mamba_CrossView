# 🚀 Mamba CrossView 快速开始指南

## 1. 数据准备

### 数据结构要求
请确保您的数据按以下结构组织：

```
/home/ma-user/work/Mamba_CrossView/data/University-123/
├── train/
│   ├── satellite/
│   │   ├── class_001/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── class_002/
│   │   └── ...
│   ├── street/
│   │   ├── class_001/
│   │   ├── class_002/
│   │   └── ...
│   └── drone/
│       ├── class_001/
│       ├── class_002/
│       └── ...
└── test/
    ├── gallery/
    └── query/
```

### 数据路径检查
运行数据路径检查脚本：

```bash
python test_data_path.py
```

这个脚本会自动检查数据结构是否正确。

## 2. 环境配置

确保已安装所需依赖：

```bash
pip install -r requirement.txt
```

## 3. 训练方法

### 方法一：使用训练脚本（推荐新手）

```bash
bash start_training.sh
```

这个脚本会引导您设置所有参数。

### 方法二：直接命令行

#### 使用Vision Mamba（推荐）

```bash
# 不使用预训练模型（随机初始化）
python train.py \
    --backbone MAMBA-S \
    --data_dir /home/ma-user/work/Mamba_CrossView/data/University-123/train \
    --pretrain_path '' \
    --lr 0.01 \
    --gpu_ids 0 \
    --batchsize 8

# 使用预训练模型
python train.py \
    --backbone MAMBA-S \
    --data_dir /home/ma-user/work/Mamba_CrossView/data/University-123/train \
    --pretrain_path /path/to/vision_mamba_pretrain.pth \
    --lr 0.01 \
    --gpu_ids 0 \
    --batchsize 8
```

#### 使用Vision Transformer

```bash
python train.py \
    --backbone VIT-S \
    --data_dir /home/ma-user/work/Mamba_CrossView/data/University-123/train \
    --pretrain_path /path/to/vit_pretrain.pth \
    --lr 0.01 \
    --gpu_ids 0 \
    --batchsize 8
```

## 4. 常见问题解决

### 问题1：FileNotFoundError: [Errno 2] No such file or directory

**原因**：数据路径不正确

**解决方案**：
1. 检查数据路径是否存在：`ls -la /home/ma-user/work/Mamba_CrossView/data/University-123/train`
2. 运行数据检查脚本：`python test_data_path.py`
3. 使用正确的数据路径：`--data_dir YOUR_ACTUAL_DATA_PATH`

### 问题2：CUDA out of memory

**解决方案**：
- 减小batch size：`--batchsize 4` 或 `--batchsize 2`
- 减小图像尺寸：`--h 224 --w 224`

### 问题3：找不到预训练模型

**解决方案**：
- 检查预训练模型路径是否正确
- 如果没有预训练模型，使用空字符串：`--pretrain_path ''`

### 问题4：数据类别不匹配

**解决方案**：
- 确保satellite、street、drone文件夹中的类别名称完全一致
- 运行数据检查脚本查看详细信息

## 5. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | VIT-S | 主干网络 (MAMBA-S, VIT-S, VAN-S) |
| `--data_dir` | 见脚本 | 训练数据路径 |
| `--pretrain_path` | "" | 预训练模型路径 |
| `--lr` | 0.01 | 学习率 |
| `--gpu_ids` | 0 | GPU设备ID |
| `--batchsize` | 8 | 批量大小 |
| `--num_epochs` | 120 | 训练轮数 |
| `--h` | 256 | 图像高度 |
| `--w` | 256 | 图像宽度 |

## 6. 监控训练进度

训练过程中会显示：
- Loss变化
- 准确率
- 每个epoch的时间

日志会保存在`checkpoints/`目录下。

## 7. 测试模型

训练完成后，使用以下命令测试：

```bash
python test_server.py \
    --backbone MAMBA-S \
    --resume path/to/checkpoint.pth \
    --data_dir /home/ma-user/work/Mamba_CrossView/data/University-123/test \
    --gpu_ids 0
```

## 8. 性能优化建议

1. **内存优化**：
   - 使用较小的batch size
   - 启用混合精度训练：`--autocast`

2. **速度优化**：
   - 使用多个GPU：`--gpu_ids 0,1,2,3`
   - 增加num_worker：`--num_worker 8`

3. **精度优化**：
   - 调整学习率：`--lr 0.0035`
   - 使用数据增强：`--color_jitter --DA`

## 9. 获取帮助

如果遇到问题：
1. 运行 `python test_data_path.py` 检查数据
2. 查看错误信息的详细提示
3. 检查GPU内存使用情况
4. 在GitHub上提交Issue

---

**祝您训练顺利！🎉** 