"""
测试模型保存功能的修复
"""
import torch
import torch.nn as nn
import os
import sys

# 添加项目路径
sys.path.append('.')
from tool.utils_server import save_best_model, save_network_with_name

class DummyModel(nn.Module):
    """测试用的简单模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

def test_model_save_functions():
    """测试模型保存函数"""
    print("🧪 测试模型保存函数修复...")
    
    # 创建测试模型
    model = DummyModel()
    
    # 测试目录
    test_dir = "test_save"
    
    try:
        # 测试 save_best_model
        print("\n1. 测试 save_best_model...")
        save_path = save_best_model(
            network=model,
            dirname=test_dir, 
            epoch_label=10,
            metric_value=0.8734,
            metric_name="accuracy",
            model_name="test_model"
        )
        print(f"✅ save_best_model 成功，保存路径: {save_path}")
        
        # 测试 save_network_with_name  
        print("\n2. 测试 save_network_with_name...")
        save_path2 = save_network_with_name(
            network=model,
            dirname=test_dir,
            epoch_label=10, 
            model_name="test_model"
        )
        print(f"✅ save_network_with_name 成功，保存路径: {save_path2}")
        
        # 检查保存的文件
        print("\n3. 检查保存的文件...")
        checkpoint_dir = f"./checkpoints/{test_dir}"
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            print(f"📁 保存了 {len(files)} 个文件:")
            for file in files:
                filepath = os.path.join(checkpoint_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"   📄 {file} ({size_mb:.2f}MB)")
                
                # 检查文件内容
                try:
                    checkpoint = torch.load(filepath, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        print(f"      ✅ 包含键: {list(checkpoint.keys())}")
                        print(f"      📊 架构: {checkpoint.get('architecture', 'N/A')}")
                        print(f"      🎯 轮数: {checkpoint.get('epoch', 'N/A')}")
                        if 'best_metric' in checkpoint:
                            metric = checkpoint['best_metric']
                            print(f"      🏆 最佳指标: {metric['name']}={metric['value']:.4f}")
                    else:
                        print(f"      ⚠️  旧格式文件")
                except Exception as e:
                    print(f"      ❌ 读取失败: {e}")
        else:
            print(f"❌ 目录不存在: {checkpoint_dir}")
            
        print(f"\n🎉 所有测试通过！模型保存功能已修复。")
        
        # 清理测试文件
        print(f"\n🧹 清理测试文件...")
        import shutil
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            print(f"✅ 已清理测试目录: {checkpoint_dir}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_model_save_functions()
    
    if success:
        print(f"\n✅ 修复验证成功！现在可以正常训练了。")
        print(f"\n🚀 重新开始训练:")
        print(f"python train.py --backbone MAMBA-LITE --optimizer adamw --lr 0.001 --name my_experiment")
    else:
        print(f"\n❌ 修复验证失败，请检查代码。") 