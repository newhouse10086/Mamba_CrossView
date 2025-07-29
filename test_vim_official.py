#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试官方Vision Mamba (Vim)实现
验证模型创建、权重加载和前向传播

用法:
python test_vim_official.py
"""

import torch
import torch.nn as nn
import os
import sys
sys.path.append('.')

from models.FSRA.backbones.vim_official import (
    vim_tiny_patch16_224_FSRA, 
    vim_small_patch16_224_FSRA, 
    VisionMambaOfficial
)

def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("🔧 测试模型创建...")
    
    try:
        # 测试Vim-Tiny
        print("\n📦 创建Vim-Tiny模型...")
        model_tiny = vim_tiny_patch16_224_FSRA(
            img_size=(256, 256), 
            stride_size=16, 
            drop_rate=0.0, 
            local_feature=False
        )
        print(f"✅ Vim-Tiny创建成功")
        print(f"   - 参数量: {sum(p.numel() for p in model_tiny.parameters()):,}")
        print(f"   - 嵌入维度: {model_tiny.embed_dim}")
        
        # 测试Vim-Small
        print("\n📦 创建Vim-Small模型...")
        model_small = vim_small_patch16_224_FSRA(
            img_size=(256, 256), 
            stride_size=16, 
            drop_rate=0.0, 
            local_feature=False
        )
        print(f"✅ Vim-Small创建成功")
        print(f"   - 参数量: {sum(p.numel() for p in model_small.parameters()):,}")
        print(f"   - 嵌入维度: {model_small.embed_dim}")
        
        return model_tiny, model_small
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, model_name):
    """测试前向传播"""
    print(f"\n🚀 测试{model_name}前向传播...")
    
    try:
        # 创建输入数据
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        print(f"   输入形状: {input_tensor.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            
        print(f"✅ {model_name}前向传播成功")
        print(f"   输出形状: {output.shape}")
        
        # 检查输出维度
        expected_patches = (256 // 16) * (256 // 16)  # 16x16 patches
        expected_seq_len = expected_patches + 1  # +1 for cls token
        
        if output.shape[1] == expected_seq_len:
            print(f"   ✅ 输出序列长度正确: {expected_seq_len}")
        else:
            print(f"   ⚠️  输出序列长度: {output.shape[1]}, 预期: {expected_seq_len}")
            
        return True
        
    except Exception as e:
        print(f"❌ {model_name}前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pretrained_loading():
    """测试预训练权重加载"""
    print("\n🔄 测试预训练权重加载...")
    
    pretrained_path = "vim_t_midclstok_ft_78p3acc.pth"
    
    if not os.path.exists(pretrained_path):
        print(f"⚠️  预训练权重文件不存在: {pretrained_path}")
        print("   请确保vim_t_midclstok_ft_78p3acc.pth在项目根目录")
        return False
        
    try:
        # 创建模型
        model = vim_tiny_patch16_224_FSRA(
            img_size=(224, 224),  # 使用标准ImageNet尺寸测试
            stride_size=16,
            drop_rate=0.0,
            local_feature=False
        )
        
        # 加载预训练权重
        print(f"📥 正在加载预训练权重: {pretrained_path}")
        model.load_param(pretrained_path)
        
        print("✅ 预训练权重加载测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 预训练权重加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fsra_compatibility():
    """测试FSRA框架兼容性"""
    print("\n🔗 测试FSRA框架兼容性...")
    
    try:
        from models.FSRA.make_model import build_transformer
        
        # 创建opt对象模拟
        class MockOpt:
            def __init__(self):
                self.backbone = "VIM-TINY"
                self.pretrain_path = "vim_t_midclstok_ft_78p3acc.pth"
        
        opt = MockOpt()
        num_classes = 701  # University-1652数据集类别数
        
        # 测试build_transformer
        print("🏗️  测试build_transformer...")
        transformer_model = build_transformer(opt, num_classes, block=4, return_f=False)
        
        print("✅ FSRA框架兼容性测试通过")
        print(f"   - 输入维度: {transformer_model.in_planes}")
        print(f"   - 分类器数量: {transformer_model.block}")
        
        # 测试前向传播
        input_tensor = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = transformer_model(input_tensor)
            
        print(f"   - 输出形状: {output.shape if not isinstance(output, list) else [o.shape for o in output]}")
        
        return True
        
    except Exception as e:
        print(f"❌ FSRA框架兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🎯 官方Vision Mamba (Vim)实现测试")
    print("基于: https://github.com/hustvl/Vim")
    print("=" * 60)
    
    all_tests_passed = True
    
    # 1. 测试模型创建
    model_tiny, model_small = test_model_creation()
    if model_tiny is None or model_small is None:
        all_tests_passed = False
    
    # 2. 测试前向传播
    if model_tiny is not None:
        if not test_forward_pass(model_tiny, "Vim-Tiny"):
            all_tests_passed = False
            
    if model_small is not None:
        if not test_forward_pass(model_small, "Vim-Small"):
            all_tests_passed = False
    
    # 3. 测试预训练权重加载
    if not test_pretrained_loading():
        all_tests_passed = False
        
    # 4. 测试FSRA兼容性
    if not test_fsra_compatibility():
        all_tests_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 所有测试通过！")
        print("\n💡 使用建议:")
        print("   1. 推荐使用VIM-TINY作为主要backbone")
        print("   2. 学习率建议: 0.0003-0.0005")
        print("   3. 预训练权重: vim_t_midclstok_ft_78p3acc.pth")
        print("\n🚀 训练命令示例:")
        print("   python train.py --backbone VIM-TINY --pretrain_path vim_t_midclstok_ft_78p3acc.pth --lr 0.0005")
    else:
        print("❌ 部分测试失败，请检查错误信息")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main()) 