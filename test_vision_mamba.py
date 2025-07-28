#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Mamba 测试脚本
用于验证模型能否正常前向传播
"""

import torch
import torch.nn as nn
from models.FSRA.backbones.vision_mamba import vision_mamba_small_patch16_224_FSRA

def test_vision_mamba():
    """测试Vision Mamba模型的前向传播"""
    print("=== Vision Mamba 测试 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    print("创建Vision Mamba模型...")
    model = vision_mamba_small_patch16_224_FSRA(
        img_size=(256, 256), 
        stride_size=16, 
        drop_rate=0.0, 
        local_feature=False
    )
    model = model.to(device)
    
    # 创建测试输入
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    
    print(f"创建测试输入: [{batch_size}, {channels}, {height}, {width}]")
    test_input = torch.randn(batch_size, channels, height, width).to(device)
    
    # 测试前向传播
    print("开始前向传播测试...")
    try:
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ 前向传播成功!")
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        
        # 检查输出维度
        expected_patches = (256 // 16) * (256 // 16) + 1  # +1 for cls token
        print(f"预期序列长度: {expected_patches}")
        print(f"实际序列长度: {output.shape[1]}")
        
        if output.shape[1] == expected_patches:
            print("✅ 输出维度正确!")
        else:
            print("⚠️ 输出维度可能不符合预期")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_sizes():
    """测试不同输入尺寸"""
    print("\n=== 测试不同输入尺寸 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vision_mamba_small_patch16_224_FSRA(
        img_size=(256, 256), 
        stride_size=16, 
        drop_rate=0.0, 
        local_feature=False
    ).to(device)
    
    test_sizes = [
        (1, 256, 256),  # batch=1
        (4, 256, 256),  # batch=4
        (2, 224, 224),  # different image size
    ]
    
    for batch, h, w in test_sizes:
        try:
            print(f"测试尺寸: [{batch}, 3, {h}, {w}]")
            test_input = torch.randn(batch, 3, h, w).to(device)
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  ✅ 成功! 输出形状: {output.shape}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")

def test_gradient_flow():
    """测试梯度流"""
    print("\n=== 测试梯度流 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vision_mamba_small_patch16_224_FSRA(
        img_size=(256, 256), 
        stride_size=16, 
        drop_rate=0.0, 
        local_feature=False
    ).to(device)
    
    test_input = torch.randn(2, 3, 256, 256).to(device)
    test_input.requires_grad_(True)
    
    try:
        # 前向传播
        output = model(test_input)
        
        # 创建一个简单的损失
        loss = output.mean()
        
        # 反向传播
        loss.backward()
        
        print("✅ 梯度流测试成功!")
        print(f"损失值: {loss.item():.6f}")
        
        # 检查是否有梯度
        has_grad = test_input.grad is not None and test_input.grad.abs().sum() > 0
        print(f"输入梯度存在: {has_grad}")
        
        return True
        
    except Exception as e:
        print(f"❌ 梯度流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Vision Mamba 综合测试")
    print("=" * 50)
    
    # 基本前向传播测试
    success1 = test_vision_mamba()
    
    # 不同尺寸测试
    test_with_different_sizes()
    
    # 梯度流测试
    success2 = test_gradient_flow()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 所有测试通过! Vision Mamba可以正常使用")
        print("\n现在您可以运行:")
        print("python train.py --backbone MAMBA-S --data_dir YOUR_DATA_PATH --pretrain_path '' --lr 0.01 --gpu_ids 0")
    else:
        print("❌ 部分测试失败，请检查模型实现")

if __name__ == "__main__":
    main() 