
# NVIDIA MambaVision下载脚本
from transformers import AutoImageProcessor, AutoModel
import torch

# 可选的模型大小
models = {
    'tiny': 'nvidia/MambaVision-T-1K',      # ~31M parameters
    'small': 'nvidia/MambaVision-S-1K',     # ~50M parameters  
    'base': 'nvidia/MambaVision-B-1K',      # ~93M parameters
}

# 下载Tiny模型（推荐用于测试）
model_name = models['tiny']
print(f"正在下载: {model_name}")

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 保存到本地
torch.save(model.state_dict(), 'mambavision_tiny_1k.pth')
print("✅ NVIDIA MambaVision Tiny 下载完成")
