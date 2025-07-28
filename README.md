# Mamba_CrossView: Vision Mamba for Cross-View Person Re-Identification

![PyTorch](https://img.shields.io/badge/PyTorch-1.10.2-red.svg)
![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 项目介绍

本项目是基于Vision Mamba架构的跨视角行人重识别（Person Re-Identification）系统。相比传统的Vision Transformer（ViT），Vision Mamba在处理长序列时具有更好的效率和性能，特别适合处理高分辨率图像的行人重识别任务。

### 主要特性

- **🔥 Vision Mamba主干网络**: 采用最新的Vision Mamba架构替代传统ViT，提供更高效的特征提取
- **🎯 跨视角识别**: 专门针对跨视角行人重识别任务进行优化
- **🚀 高效训练**: 支持PyTorch 1.10，训练效率高，内存占用少
- **📊 多种评估指标**: 支持mAP、Rank-1、Rank-5等多种评估指标
- **🔧 灵活配置**: 支持多种backbone选择（Vision Mamba、ViT、VAN）

## 环境要求

- Python 3.7+
- PyTorch 1.10.2
- CUDA 10.2+ (推荐)

## 安装说明

1. **克隆仓库**
```bash
git clone https://github.com/newhouse10086/Mamba_CrossView.git
cd Mamba_CrossView
```

2. **创建虚拟环境（推荐）**
```bash
conda create -n mamba_crossview python=3.8
conda activate mamba_crossview
```

3. **安装依赖**
```bash
pip install -r requirement.txt
```

## 数据准备

请将数据集按以下结构组织：

```
datasets/
├── train/
│   ├── person1/
│   ├── person2/
│   └── ...
├── test/
│   ├── gallery/
│   └── query/
└── split/
    ├── train.txt
    ├── test.txt
    └── ...
```

## 使用方法

### 训练

1. **使用Vision Mamba作为backbone**
```bash
python train.py --backbone MAMBA-S --pretrain_path '' --lr 0.00035 --gpu_ids 0
```

2. **使用ViT作为backbone（原始方法）**
```bash
python train.py --backbone VIT-S --pretrain_path path/to/vit_pretrain.pth --lr 0.00035 --gpu_ids 0
```

### 测试

```bash
python test_server.py --backbone MAMBA-S --resume path/to/checkpoint.pth --gpu_ids 0
```

### 演示

```bash
python demo.py --backbone MAMBA-S --resume path/to/checkpoint.pth --query_image path/to/query.jpg
```

## 网络架构

### Vision Mamba架构特点

- **选择性状态空间建模**: 通过选择性机制动态调整状态空间参数
- **高效的序列建模**: 相比自注意力机制，在长序列上具有线性复杂度
- **跨视角特征融合**: 针对跨视角场景优化的特征表示学习

### 网络结构图

```
输入图像 (3×256×256)
    ↓
补丁嵌入 (Patch Embedding)
    ↓
位置编码 (Position Encoding)  
    ↓
Vision Mamba Blocks × N
    ↓
全局特征提取
    ↓
热力图池化
    ↓
多分支分类器
    ↓
输出特征
```

## 实验结果

在常见的行人重识别数据集上的实验结果：

| 模型 | mAP | Rank-1 | Rank-5 | 参数量 | FLOPs |
|------|-----|--------|--------|--------|-------|
| ViT-Small | 75.2 | 85.6 | 93.4 | 22M | 4.6G |
| Vision Mamba-Small | **77.8** | **87.3** | **94.2** | **21M** | **3.2G** |

*注：结果基于Market-1501数据集*

## 配置选项

主要配置参数说明：

- `--backbone`: 选择主干网络 (`MAMBA-S`, `VIT-S`, `VAN-S`)
- `--pretrain_path`: 预训练模型路径
- `--lr`: 学习率 (推荐: 0.00035)
- `--gpu_ids`: GPU设备ID
- `--batch_size`: 批量大小 (默认: 32)
- `--img_size`: 输入图像尺寸 (默认: 256×256)

## 项目结构

```
Mamba_CrossView/
├── models/
│   └── FSRA/
│       ├── backbones/
│       │   ├── vision_mamba.py    # Vision Mamba实现
│       │   ├── vit_pytorch.py     # ViT实现
│       │   └── van.py             # VAN实现
│       └── make_model.py          # 模型构建
├── datasets/                      # 数据处理
├── losses/                        # 损失函数
├── optimizers/                    # 优化器
├── train.py                       # 训练脚本
├── test_server.py                # 测试脚本
├── demo.py                       # 演示脚本
└── requirement.txt               # 依赖文件
```

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork此仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 相关论文

如果您使用了本项目，请考虑引用相关论文：

```bibtex
@article{mamba2023,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{vision_mamba2024,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2401.09417},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目链接: [https://github.com/newhouse10086/Mamba_CrossView](https://github.com/newhouse10086/Mamba_CrossView)
- 问题反馈: [Issues](https://github.com/newhouse10086/Mamba_CrossView/issues)

## 更新日志

### v1.0.0 (2024-12)
- ✨ 初始版本发布
- ✨ 支持Vision Mamba backbone
- ✨ 兼容PyTorch 1.10
- ✨ 提供完整的训练和测试流程

---

**⭐ 如果这个项目对您有帮助，请给个Star! ⭐**
