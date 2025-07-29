"""
针对Vision Mamba优化的优化器配置
"""
import torch.optim as optim
from torch.optim import lr_scheduler
import torch


def make_optimizer_adamw(model, opt):
    """
    AdamW优化器 - 适合Vision Mamba等Transformer架构
    """
    ignored_params = []
    if opt.views == 3:
        for i in [model.model_1, model.model_2]:
            ignored_params += list(map(id, i.transformer.parameters()))
    else:
        for i in [model.model_1]:
            ignored_params += list(map(id, i.transformer.parameters()))
    
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    
    # AdamW优化器配置
    optimizer_ft = optim.AdamW([
        {'params': base_params, 'lr': opt.lr, 'weight_decay': 0.01},      # Transformer: 标准weight_decay
        {'params': extra_params, 'lr': opt.lr * 2, 'weight_decay': 0.05}  # 分类器: 更大学习率和weight_decay
    ], betas=(0.9, 0.999), eps=1e-8)
    
    # Cosine退火学习率调度
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer_ft, 
        T_max=opt.num_epochs,
        eta_min=opt.lr * 0.01  # 最小学习率为初始学习率的1%
    )
    
    print(f"🚀 使用AdamW优化器 + Cosine退火调度")
    print(f"   Transformer学习率: {opt.lr}")
    print(f"   分类器学习率: {opt.lr * 2}")
    print(f"   Weight decay: Transformer=0.01, 分类器=0.05")
    
    return optimizer_ft, exp_lr_scheduler


def make_optimizer_sgd_improved(model, opt):
    """
    改进的SGD优化器配置 - 对Vision Mamba友好
    """
    ignored_params = []
    if opt.views == 3:
        for i in [model.model_1, model.model_2]:
            ignored_params += list(map(id, i.transformer.parameters()))
    else:
        for i in [model.model_1]:
            ignored_params += list(map(id, i.transformer.parameters()))
    
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    
    # 改进的SGD配置
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.5 * opt.lr, 'weight_decay': 1e-4},   # Transformer: 更温和的学习率
        {'params': extra_params, 'lr': opt.lr, 'weight_decay': 5e-4}          # 分类器: 标准配置
    ], momentum=0.9, nesterov=True)
    
    # 更温和的学习率调度
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, 
        milestones=[int(opt.num_epochs * 0.6), int(opt.num_epochs * 0.8)],  # 60%和80%时降低
        gamma=0.3  # 降为30%而不是10%
    )
    
    print(f"🔧 使用改进SGD优化器")
    print(f"   Transformer学习率: {0.5 * opt.lr}")
    print(f"   分类器学习率: {opt.lr}")
    print(f"   调度: 在epoch {int(opt.num_epochs * 0.6)}, {int(opt.num_epochs * 0.8)}时降低到30%")
    
    return optimizer_ft, exp_lr_scheduler


def make_optimizer_lion(model, opt):
    """
    Lion优化器 - 新兴的高效优化器，适合大模型
    需要安装: pip install lion-pytorch
    """
    try:
        from lion_pytorch import Lion
        
        ignored_params = []
        if opt.views == 3:
            for i in [model.model_1, model.model_2]:
                ignored_params += list(map(id, i.transformer.parameters()))
        else:
            for i in [model.model_1]:
                ignored_params += list(map(id, i.transformer.parameters()))
        
        extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
        
        # Lion优化器通常需要更小的学习率
        optimizer_ft = Lion([
            {'params': base_params, 'lr': opt.lr * 0.1, 'weight_decay': 0.01},
            {'params': extra_params, 'lr': opt.lr * 0.2, 'weight_decay': 0.02}
        ])
        
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer_ft, 
            T_max=opt.num_epochs,
            eta_min=opt.lr * 0.001
        )
        
        print(f"🦁 使用Lion优化器 (内存高效)")
        print(f"   Transformer学习率: {opt.lr * 0.1}")
        print(f"   分类器学习率: {opt.lr * 0.2}")
        
        return optimizer_ft, exp_lr_scheduler
        
    except ImportError:
        print("❌ Lion优化器未安装，回退到AdamW")
        return make_optimizer_adamw(model, opt)


def make_optimizer_auto(model, opt):
    """
    自动选择最适合的优化器
    """
    backbone = getattr(opt, 'backbone', 'VIT-S')
    
    if 'MAMBA' in backbone:
        print(f"🎯 检测到Vision Mamba架构 ({backbone})，推荐AdamW优化器")
        return make_optimizer_adamw(model, opt)
    else:
        print(f"🎯 检测到其他架构 ({backbone})，使用改进SGD优化器")
        return make_optimizer_sgd_improved(model, opt)


# 为了方便切换，提供统一接口
def make_optimizer(model, opt, optimizer_type='auto'):
    """
    统一的优化器创建接口
    
    Args:
        model: 模型
        opt: 配置选项
        optimizer_type: 优化器类型
            - 'auto': 自动选择
            - 'adamw': AdamW优化器
            - 'sgd': 改进的SGD优化器
            - 'sgd_original': 原始SGD配置
            - 'lion': Lion优化器
    """
    if optimizer_type == 'auto':
        return make_optimizer_auto(model, opt)
    elif optimizer_type == 'adamw':
        return make_optimizer_adamw(model, opt)
    elif optimizer_type == 'sgd':
        return make_optimizer_sgd_improved(model, opt)
    elif optimizer_type == 'lion':
        return make_optimizer_lion(model, opt)
    elif optimizer_type == 'sgd_original':
        # 回到原始配置
        from .make_optimizer import make_optimizer as original_make_optimizer
        return original_make_optimizer(model, opt)
    else:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")


# 推荐的训练配置
RECOMMENDED_CONFIGS = {
    'VIM-TINY': {
        'optimizer': 'adamw',
        'lr': 0.0005,
        'batch_size': 16,
        'num_epochs': 120,
        'description': '官方Vision Mamba Tiny，双向状态空间，78.3% ImageNet准确率'
    },
    'VIM-SMALL': {
        'optimizer': 'adamw',
        'lr': 0.0003,
        'batch_size': 12,
        'num_epochs': 120,
        'description': '官方Vision Mamba Small，更大容量，更好性能'
    },
    'MAMBA-LITE': {
        'optimizer': 'adamw',
        'lr': 0.001,
        'batch_size': 32,
        'num_epochs': 80,
        'description': '轻量级Vision Mamba，AdamW + Cosine调度'
    },
    'MAMBA-V2': {
        'optimizer': 'adamw', 
        'lr': 0.0005,
        'batch_size': 16,
        'num_epochs': 120,
        'description': '完整Vision Mamba，小学习率AdamW'
    },
    'VIT-S': {
        'optimizer': 'sgd',
        'lr': 0.01,
        'batch_size': 8,
        'num_epochs': 120,
        'description': 'Vision Transformer，SGD + MultiStep调度'
    }
}

def print_recommended_config(backbone):
    """打印推荐配置"""
    if backbone in RECOMMENDED_CONFIGS:
        config = RECOMMENDED_CONFIGS[backbone]
        print(f"\n💡 {backbone} 推荐配置:")
        print(f"   优化器: {config['optimizer']}")
        print(f"   学习率: {config['lr']}")
        print(f"   批大小: {config['batch_size']}")
        print(f"   训练轮数: {config['num_epochs']}")
        print(f"   说明: {config['description']}")
    else:
        print(f"\n💡 {backbone} 建议使用自动配置 (auto)") 