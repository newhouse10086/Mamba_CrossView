"""
优化器对比测试脚本 - 比较不同优化器的效果
"""
import torch
import torch.nn as nn
import numpy as np
from models.FSRA.backbones.vision_mamba_lite import vision_mamba_lite_small_patch16_224_FSRA
from optimizers.make_optimizer_mamba import make_optimizer, RECOMMENDED_CONFIGS
import matplotlib.pyplot as plt
import time

class OptArgs:
    """模拟命令行参数"""
    def __init__(self, backbone='MAMBA-LITE', lr=0.001, num_epochs=50, views=1):
        self.backbone = backbone
        self.lr = lr
        self.num_epochs = num_epochs
        self.views = views

def create_dummy_model():
    """创建测试模型"""
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_1 = type('', (), {})()  # 创建空对象
            self.model_1.transformer = vision_mamba_lite_small_patch16_224_FSRA(
                img_size=(256, 256), 
                stride_size=16, 
                drop_rate=0.0, 
                local_feature=False
            )
            # 添加一个简单的分类器
            self.classifier = nn.Linear(512, 100)
            
        def forward(self, x):
            feat = self.model_1.transformer(x)
            return self.classifier(feat[:, 0])  # 使用cls_token
    
    return DummyModel()

def test_optimizer_performance():
    """测试不同优化器的性能"""
    print("🧪 优化器性能对比测试")
    print("=" * 60)
    
    # 测试配置
    optimizers_to_test = ['sgd_original', 'sgd', 'adamw']  # 不测试lion避免依赖问题
    batch_size = 4
    num_batches = 20  # 测试20个batch
    input_size = (batch_size, 3, 256, 256)
    num_classes = 100
    
    results = {}
    
    for opt_type in optimizers_to_test:
        print(f"\n🔍 测试优化器: {opt_type}")
        print("-" * 30)
        
        try:
            # 创建模型和优化器
            model = create_dummy_model()
            opt_args = OptArgs(lr=0.001 if opt_type == 'adamw' else 0.01)
            
            optimizer, scheduler = make_optimizer(model, opt_args, optimizer_type=opt_type)
            
            # 创建loss函数
            criterion = nn.CrossEntropyLoss()
            
            # 测试训练性能
            model.train()
            losses = []
            times = []
            
            for batch_idx in range(num_batches):
                # 生成随机数据
                inputs = torch.randn(input_size)
                targets = torch.randint(0, num_classes, (batch_size,))
                
                start_time = time.time()
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                end_time = time.time()
                
                losses.append(loss.item())
                times.append(end_time - start_time)
                
                if batch_idx % 5 == 0:
                    print(f"   Batch {batch_idx:2d}: Loss={loss.item():.4f}, Time={end_time-start_time:.3f}s")
            
            # 统计结果
            avg_loss = np.mean(losses)
            final_loss = losses[-1]
            avg_time = np.mean(times)
            loss_std = np.std(losses)
            
            results[opt_type] = {
                'avg_loss': avg_loss,
                'final_loss': final_loss,
                'avg_time': avg_time,
                'loss_std': loss_std,
                'losses': losses,
                'convergence_rate': (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
            }
            
            print(f"   📊 结果: 平均Loss={avg_loss:.4f}, 最终Loss={final_loss:.4f}")
            print(f"           平均时间={avg_time:.3f}s, 收敛率={results[opt_type]['convergence_rate']:.2%}")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            results[opt_type] = None
    
    # 打印对比结果
    print("\n📈 优化器对比结果:")
    print("=" * 80)
    print(f"{'优化器':<15} {'平均Loss':<10} {'最终Loss':<10} {'平均时间':<10} {'收敛率':<10} {'稳定性':<10}")
    print("-" * 80)
    
    for opt_type, result in results.items():
        if result is not None:
            stability = "好" if result['loss_std'] < result['avg_loss'] * 0.1 else "中等" if result['loss_std'] < result['avg_loss'] * 0.3 else "差"
            print(f"{opt_type:<15} {result['avg_loss']:<10.4f} {result['final_loss']:<10.4f} {result['avg_time']:<10.3f} {result['convergence_rate']:<10.2%} {stability:<10}")
        else:
            print(f"{opt_type:<15} {'失败':<50}")
    
    return results

def test_optimizer_params():
    """测试不同优化器的参数设置"""
    print("\n🔧 优化器参数设置测试")
    print("=" * 60)
    
    model = create_dummy_model()
    opt_args = OptArgs()
    
    optimizers_info = {
        'sgd_original': '原版SGD',
        'sgd': '改进SGD', 
        'adamw': 'AdamW',
        'auto': '自动选择'
    }
    
    for opt_type, description in optimizers_info.items():
        print(f"\n🔍 {description} ({opt_type}):")
        try:
            optimizer, scheduler = make_optimizer(model, opt_args, optimizer_type=opt_type)
            
            # 分析参数组
            param_groups = optimizer.param_groups
            print(f"   参数组数量: {len(param_groups)}")
            
            for i, group in enumerate(param_groups):
                param_count = sum(p.numel() for p in group['params'])
                print(f"   组 {i+1}: {param_count:,} 参数")
                print(f"        学习率: {group['lr']}")
                print(f"        Weight decay: {group.get('weight_decay', 'N/A')}")
                if 'momentum' in group:
                    print(f"        Momentum: {group['momentum']}")
                if 'betas' in group:
                    print(f"        Betas: {group['betas']}")
            
            print(f"   调度器类型: {type(scheduler).__name__}")
            
        except Exception as e:
            print(f"   ❌ 创建失败: {e}")

def show_recommendations():
    """显示推荐配置"""
    print("\n💡 不同backbone的推荐配置:")
    print("=" * 60)
    
    for backbone, config in RECOMMENDED_CONFIGS.items():
        print(f"\n🎯 {backbone}:")
        print(f"   优化器: {config['optimizer']}")
        print(f"   学习率: {config['lr']}")
        print(f"   批大小: {config['batch_size']}")
        print(f"   训练轮数: {config['num_epochs']}")
        print(f"   说明: {config['description']}")

def main():
    """主测试函数"""
    print("🚀 Vision Mamba 优化器全面测试")
    print("=" * 60)
    
    # 1. 显示推荐配置
    show_recommendations()
    
    # 2. 测试优化器参数设置
    test_optimizer_params()
    
    # 3. 性能对比测试
    print("\n" + "="*60)
    print("开始性能测试 (这可能需要几分钟)...")
    results = test_optimizer_performance()
    
    # 4. 总结建议
    print("\n🎯 使用建议:")
    print("=" * 60)
    print("1. Vision Mamba模型推荐使用 AdamW 优化器")
    print("   命令: python train.py --backbone MAMBA-LITE --optimizer adamw --lr 0.001")
    print()
    print("2. 如果内存有限，可以使用改进的 SGD")
    print("   命令: python train.py --backbone MAMBA-LITE --optimizer sgd --lr 0.001")
    print()
    print("3. 使用自动选择让系统为您推荐")
    print("   命令: python train.py --backbone MAMBA-LITE --optimizer auto")
    print()
    print("4. 保持原版配置")
    print("   命令: python train.py --backbone MAMBA-LITE --optimizer sgd_original --lr 0.01")

if __name__ == "__main__":
    main() 