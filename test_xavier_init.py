"""
测试VisionMambaLite的Xavier初始化
"""
import torch
import numpy as np
from models.FSRA.backbones.vision_mamba_lite import vision_mamba_lite_small_patch16_224_FSRA

def test_xavier_initialization():
    print("🔍 测试VisionMambaLite的Xavier初始化...")
    
    # 创建模型
    model = vision_mamba_lite_small_patch16_224_FSRA(
        img_size=(256, 256), 
        stride_size=16, 
        drop_rate=0.0, 
        local_feature=False
    )
    
    print(f"✅ 模型创建成功")
    print(f"📊 总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 检查各层的初始化统计
    print("\n📈 权重初始化统计:")
    print("=" * 60)
    
    layer_stats = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) >= 2:
            weight_data = param.data.cpu().numpy()
            mean = np.mean(weight_data)
            std = np.std(weight_data)
            min_val = np.min(weight_data)
            max_val = np.max(weight_data)
            
            layer_stats.append({
                'name': name,
                'shape': tuple(param.shape),
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val
            })
    
    # 打印统计信息
    for stat in layer_stats[:15]:  # 只显示前15层
        print(f"🔹 {stat['name']:<40} | Shape: {str(stat['shape']):<15} | Mean: {stat['mean']:.4f} | Std: {stat['std']:.4f}")
    
    if len(layer_stats) > 15:
        print(f"... 还有 {len(layer_stats) - 15} 层未显示")
    
    # 检查Xavier初始化的理论标准差
    print("\n🎯 Xavier初始化验证:")
    print("=" * 60)
    
    for stat in layer_stats[:5]:  # 检查前5层
        shape = stat['shape']
        if len(shape) == 2:  # Linear层
            fan_in, fan_out = shape[1], shape[0]
            xavier_std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier uniform的理论std
            print(f"📐 {stat['name']:<40}")
            print(f"   理论Xavier std: {xavier_std:.4f} | 实际std: {stat['std']:.4f} | 比率: {stat['std']/xavier_std:.2f}")
        elif len(shape) == 4:  # Conv2d层
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
            xavier_std = np.sqrt(2.0 / (fan_in + fan_out))
            print(f"📐 {stat['name']:<40}")
            print(f"   理论Xavier std: {xavier_std:.4f} | 实际std: {stat['std']:.4f} | 比率: {stat['std']/xavier_std:.2f}")
    
    # 测试前向传播
    print("\n🚀 测试前向传播:")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        # 创建测试输入
        test_input = torch.randn(2, 3, 256, 256)
        print(f"📥 输入形状: {test_input.shape}")
        
        try:
            output = model(test_input)
            print(f"📤 输出形状: {output.shape}")
            
            # 检查输出统计
            output_stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item()
            }
            
            print(f"📊 输出统计: Mean={output_stats['mean']:.4f}, Std={output_stats['std']:.4f}")
            print(f"            Min={output_stats['min']:.4f}, Max={output_stats['max']:.4f}")
            
            # 检查是否有异常值
            if abs(output_stats['mean']) > 1.0:
                print("⚠️  警告: 输出均值较大，可能存在初始化问题")
            elif abs(output_stats['mean']) < 0.1 and output_stats['std'] > 0.1:
                print("✅ 输出统计正常，Xavier初始化工作良好")
            
            print("✅ 前向传播成功!")
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False
    
    print("\n🎉 Xavier初始化测试完成!")
    return True

if __name__ == "__main__":
    success = test_xavier_initialization()
    if success:
        print("\n🚀 可以开始训练了!")
        print("推荐命令:")
        print("python train.py --backbone MAMBA-LITE --lr 0.001 --batchsize 32 --gpu_ids 0")
    else:
        print("\n❌ 测试失败，请检查初始化代码") 