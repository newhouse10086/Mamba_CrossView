#!/usr/bin/env python3
"""
快速测试Vision Mamba前向传播
"""
import torch
import sys
sys.path.append('.')

def test_vim_forward():
    print("🧪 测试VIM模型前向传播...")
    
    try:
        # 模拟opt配置
        class MockOpt:
            def __init__(self):
                self.backbone = "VIM-TINY"
                self.pretrain_path = ""  # 不加载预训练权重，快速测试
        
        opt = MockOpt()
        
        # 创建模型
        from models.FSRA.make_model import build_transformer
        print("📦 创建build_transformer模型...")
        model = build_transformer(opt, num_classes=701, block=4, return_f=False)
        
        # 设置为评估模式
        model.eval()
        
        # 创建测试输入
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        print(f"📥 输入形状: {input_tensor.shape}")
        
        # 前向传播
        print("🚀 执行前向传播...")
        with torch.no_grad():
            output = model(input_tensor)
            
        print("✅ 前向传播成功!")
        if isinstance(output, list):
            print(f"📤 输出: {len(output)}个tensor，形状分别为:")
            for i, o in enumerate(output):
                print(f"   输出{i}: {o.shape}")
        else:
            print(f"📤 输出形状: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vim_forward()
    if success:
        print("\n🎉 测试通过！模型可以正常进行前向传播。")
        print("💡 现在可以开始训练了。")
    else:
        print("\n❌ 测试失败，请检查错误信息。")
        exit(1) 