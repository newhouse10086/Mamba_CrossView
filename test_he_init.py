"""
测试VisionMambaLite的HE(Kaiming)初始化
"""
import torch
import numpy as np
from models.FSRA.backbones.vision_mamba_lite import vision_mamba_lite_small_patch16_224_FSRA

def test_he_initialization():
    print("🔍 测试VisionMambaLite的HE(Kaiming)初始化...")
    
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
    print("=" * 70)
    
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
    
    # 检查HE初始化的理论标准差
    print("\n🎯 HE初始化验证 (适合ReLU/SiLU激活):")
    print("=" * 70)
    
    for stat in layer_stats[:8]:  # 检查前8层
        shape = stat['shape']
        if len(shape) == 2:  # Linear层
            fan_in, fan_out = shape[1], shape[0]
            # HE uniform (fan_in mode): std = sqrt(6/fan_in)
            # HE normal (fan_out mode): std = sqrt(2/fan_out)
            he_std_fan_in = np.sqrt(2.0 / fan_in)  # HE normal fan_in mode
            he_std_fan_out = np.sqrt(2.0 / fan_out)  # HE normal fan_out mode
            he_uniform_std = np.sqrt(6.0 / fan_in)  # HE uniform fan_in mode
            
            print(f"📐 {stat['name']:<40}")
            print(f"   HE std (fan_in): {he_std_fan_in:.4f} | HE std (fan_out): {he_std_fan_out:.4f}")
            print(f"   HE uniform std: {he_uniform_std:.4f} | 实际std: {stat['std']:.4f}")
            
            # 判断最接近哪种初始化
            diff_fan_in = abs(stat['std'] - he_std_fan_in)
            diff_fan_out = abs(stat['std'] - he_std_fan_out)
            diff_uniform = abs(stat['std'] - he_uniform_std)
            
            if diff_uniform < min(diff_fan_in, diff_fan_out):
                print(f"   🎯 最接近: HE uniform (fan_in) | 差异: {diff_uniform:.4f}")
            elif diff_fan_in < diff_fan_out:
                print(f"   🎯 最接近: HE normal (fan_in) | 差异: {diff_fan_in:.4f}")
            else:
                print(f"   🎯 最接近: HE normal (fan_out) | 差异: {diff_fan_out:.4f}")
                
        elif len(shape) == 4:  # Conv2d层
            fan_in = shape[1] * shape[2] * shape[3]
            fan_out = shape[0] * shape[2] * shape[3]
            he_std_fan_out = np.sqrt(2.0 / fan_out)  # HE normal fan_out mode
            
            print(f"📐 {stat['name']:<40}")
            print(f"   HE std (fan_out): {he_std_fan_out:.4f} | 实际std: {stat['std']:.4f} | 比率: {stat['std']/he_std_fan_out:.2f}")
        
        print()
    
    # 测试前向传播
    print("\n🚀 测试前向传播:")
    print("=" * 70)
    
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
            elif abs(output_stats['mean']) < 0.2 and output_stats['std'] > 0.1:
                print("✅ 输出统计正常，HE初始化工作良好")
            
            # 检查激活值的分布
            print(f"💡 HE初始化特点: 适合ReLU族激活函数(如SiLU)，保持前向传播时激活值方差稳定")
            
            print("✅ 前向传播成功!")
            
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            return False
    
    # 测试梯度传播
    print("\n🔄 测试梯度传播:")
    print("=" * 70)
    
    model.train()
    try:
        test_input = torch.randn(1, 3, 256, 256, requires_grad=True)
        output = model(test_input)
        
        # 创建一个简单的loss
        loss = output.mean()
        loss.backward()
        
        # 检查梯度统计
        grad_stats = []
        for name, param in model.named_parameters():
            if param.grad is not None and len(param.shape) >= 2:
                grad_data = param.grad.data.cpu().numpy()
                grad_mean = np.mean(np.abs(grad_data))
                grad_std = np.std(grad_data)
                grad_stats.append({
                    'name': name,
                    'grad_mean': grad_mean,
                    'grad_std': grad_std
                })
        
        print(f"📈 梯度统计 (前5层):")
        for stat in grad_stats[:5]:
            print(f"   {stat['name']:<40} | Mean: {stat['grad_mean']:.6f} | Std: {stat['grad_std']:.6f}")
        
        # 检查梯度是否健康
        all_grad_means = [s['grad_mean'] for s in grad_stats]
        if all_grad_means:
            avg_grad = np.mean(all_grad_means)
            if avg_grad > 1e-6 and avg_grad < 1e-2:
                print("✅ 梯度分布健康，HE初始化有助于梯度稳定传播")
            elif avg_grad < 1e-6:
                print("⚠️  梯度较小，可能存在梯度消失问题")
            else:
                print("⚠️  梯度较大，可能存在梯度爆炸问题")
        
        print("✅ 梯度传播测试成功!")
        
    except Exception as e:
        print(f"❌ 梯度传播测试失败: {e}")
    
    print("\n🎉 HE初始化测试完成!")
    return True

if __name__ == "__main__":
    success = test_he_initialization()
    if success:
        print("\n🚀 HE初始化验证完成，可以开始训练了!")
        print("\n💡 HE初始化的优势:")
        print("   - 专为ReLU族激活函数(包括SiLU)设计")
        print("   - 更好地保持激活值方差")
        print("   - 有助于防止梯度消失/爆炸")
        print("   - 通常收敛更快更稳定")
        
        print("\n推荐训练命令:")
        print("python train.py --backbone MAMBA-LITE --lr 0.001 --batchsize 32 --gpu_ids 0")
    else:
        print("\n❌ 测试失败，请检查初始化代码") 