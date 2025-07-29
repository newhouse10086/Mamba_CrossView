
# Vision Mamba预训练模型整合指南

## 方案1: 使用Mamba® (推荐)

Mamba®是改进版的Vision Mamba，具有更好的收敛性和性能。

### 1. 下载Mamba®预训练模型
```bash
# 访问官方GitHub下载
git clone https://github.com/wangf3014/Mamba-Reg.git
cd Mamba-Reg
# 下载预训练权重（按照项目说明）
```

### 2. 整合到您的项目
```python
# 修改 models/FSRA/backbones/vision_mamba_lite.py
# 或创建新的 vision_mambar.py

class VisionMambaR(nn.Module):
    """Mamba® - Vision Mamba with Registers"""
    def __init__(self, img_size=224, patch_size=16, **kwargs):
        super().__init__()
        # 添加register tokens支持
        self.num_registers = 12  # Mamba®的关键特性
        # ... 其余实现
    
    def load_mambar_pretrained(self, pretrain_path):
        """加载Mamba®预训练权重"""
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        # 处理权重映射
        self.load_state_dict(checkpoint, strict=False)
```

### 3. 训练命令
```bash
python train.py --backbone MAMBA-R --pretrain_path pretrained_models/mambar_small.pth --lr 0.0001 --name mambar_experiment
```

## 方案2: 使用NVIDIA MambaVision

### 1. 安装依赖
```bash
pip install transformers timm
```

### 2. 下载模型
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('nvidia/MambaVision-T-1K')
torch.save(model.state_dict(), 'mambavision_tiny.pth')
```

### 3. 适配到two_view_net
```python
# 修改 models/model.py
class two_view_net(nn.Module):
    def __init__(self, opt, class_num, **kwargs):
        super().__init__()
        if 'MAMBA-VISION' in opt.backbone:
            # 使用NVIDIA MambaVision
            from transformers import AutoModel
            pretrained = AutoModel.from_pretrained('nvidia/MambaVision-T-1K')
            self.model_1 = adapt_mambavision_for_fsra(pretrained, class_num)
        else:
            self.model_1 = make_transformer_model(opt, class_num, **kwargs)
```

## 方案3: 立即解决收敛问题

如果急需解决当前训练问题，建议：

### 1. 临时使用ViT预训练
```bash
# 先用ViT验证数据和代码没问题
python train.py --backbone VIT-S --lr 0.01 --name vit_baseline
```

### 2. 调整Mamba训练参数
```bash
# 使用更大的学习率和更好的优化器
python train.py --backbone MAMBA-LITE --optimizer adamw --lr 0.01 --name mamba_fixed
```

### 3. 添加预热和梯度裁剪
```python
# 在train.py中添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 推荐方案优先级

1. **立即测试**: 使用ViT-S验证代码正确性
2. **短期方案**: 调整MAMBA-LITE训练参数  
3. **长期方案**: 整合Mamba®或NVIDIA MambaVision预训练模型

## 预期改进效果

- ViT-S: ~75-80% 准确率（稳定基线）
- MAMBA-LITE + 调参: ~65-70% 准确率
- Mamba® + 预训练: ~75-83% 准确率
- NVIDIA MambaVision: ~80-85% 准确率
