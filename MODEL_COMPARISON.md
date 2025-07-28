# 🚀 模型性能对比与选择指南

## 📊 各版本对比表

| 模型 | 参数量 | 计算复杂度 | 训练速度 | 收敛难度 | 性能预期 | 推荐场景 |
|------|--------|------------|----------|----------|----------|----------|
| **ViT-S** | ~22M | O(n²) | 中等 | 容易 | 75-80% | 🎯 稳定可靠 |
| **MAMBA-S** | ~22M | O(n) | 慢 | 困难 | 65-70% | ❌ 不推荐 |
| **MAMBA-V2** | ~22M | O(n) | 很慢 | 中等 | 73-77% | 🔬 研究完整版 |
| **MAMBA-LITE** | ~8M | O(n) | 快 | 容易 | 70-75% | 🚀 快速实验 |
| **VAN-S** | ~13M | O(n) | 中等 | 容易 | 72-76% | 💡 轻量选择 |

## 🎯 选择建议

### 🚀 快速训练实验 → **MAMBA-LITE**
```bash
python train.py --backbone MAMBA-LITE --lr 0.001 --batchsize 32 --gpu_ids 0
```
**优势**：
- ✅ 训练速度最快（比MAMBA-V2快4-5倍）
- ✅ 参数量少（8M vs 22M）
- ✅ 容易收敛
- ✅ 内存占用小

**劣势**：
- ❌ 性能略低于完整版

### 🎯 追求稳定性能 → **ViT-S**
```bash
python train.py --backbone VIT-S --lr 0.01 --batchsize 8 --gpu_ids 0
```
**优势**：
- ✅ 训练最稳定
- ✅ 性能可靠
- ✅ 调试容易

### 🔬 研究完整Mamba → **MAMBA-V2**
```bash
python train.py --backbone MAMBA-V2 --lr 0.0001 --batchsize 8 --gpu_ids 0
```
**优势**：
- ✅ 完整的Mamba特性
- ✅ 线性复杂度
- ✅ 理论上性能最佳

**劣势**：
- ❌ 训练很慢
- ❌ 调试困难

## 🔍 详细技术对比

### MAMBA-LITE vs MAMBA-V2 核心差异

| 特性 | MAMBA-LITE | MAMBA-V2 |
|------|------------|-----------|
| **扫描方向** | 2个（水平+垂直） | 4个（上下左右） |
| **状态空间** | 简化cumsum | 完整selective scan |
| **参数维度** | 512 | 768 |
| **深度** | 6层 | 8层 |
| **扩展比例** | 1.5x | 2x |

### 训练时间对比（预估）

| 模型 | 每个epoch时间 | 120 epochs总时间 |
|------|---------------|------------------|
| **ViT-S** | 2-3分钟 | 4-6小时 |
| **MAMBA-S** | 3-4分钟 | 6-8小时 |
| **MAMBA-V2** | 8-12分钟 | 16-24小时 |
| **MAMBA-LITE** | 1-2分钟 | 2-4小时 |

## 🛠️ 优化建议

### 针对MAMBA-LITE的优化
```bash
# 推荐配置：快速训练
python train.py \
    --backbone MAMBA-LITE \
    --lr 0.001 \
    --batchsize 32 \
    --num_epochs 80 \
    --gpu_ids 0
```

### 针对MAMBA-V2的优化
```bash
# 推荐配置：完整训练
python train.py \
    --backbone MAMBA-V2 \
    --lr 0.0001 \
    --batchsize 8 \
    --num_epochs 150 \
    --gpu_ids 0
```

## 📈 性能监控要点

### MAMBA-LITE训练监控
- **Loss应该**：5-10个epoch内开始明显下降
- **收敛速度**：相对较快，类似ViT
- **内存使用**：比ViT节省约30%

### MAMBA-V2训练监控
- **Loss应该**：15-20个epoch内开始下降
- **收敛速度**：较慢，需要耐心
- **内存使用**：与ViT相当

## 🎯 实际使用流程建议

### Step 1: 快速验证
```bash
# 先用MAMBA-LITE快速验证数据和代码
python train.py --backbone MAMBA-LITE --lr 0.001 --num_epochs 20
```

### Step 2: 基线对比
```bash
# 建立ViT基线
python train.py --backbone VIT-S --lr 0.01 --num_epochs 120
```

### Step 3: 深入研究（可选）
```bash
# 如果MAMBA-LITE效果好，再尝试完整版
python train.py --backbone MAMBA-V2 --lr 0.0001 --num_epochs 150
```

## 💡 故障排除

### 如果MAMBA-LITE收敛慢
- 尝试增加学习率到0.002
- 减少batch size到16
- 增加warmup epochs

### 如果MAMBA-V2太慢
- 切换到MAMBA-LITE
- 或者使用更少的epochs先测试

---

**结论**：对于您现在的情况（训练太慢），强烈推荐先使用 **MAMBA-LITE**！🚀 