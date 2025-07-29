"""
检查预训练模型文件内容和格式
"""
import torch
import os

def check_pretrain_model(model_path):
    """检查预训练模型文件"""
    print(f"🔍 检查预训练模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        return False
    
    try:
        # 检查文件大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"📁 文件大小: {file_size:.2f} MB")
        
        # 加载模型文件
        print(f"📤 正在加载模型文件...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"✅ 模型文件加载成功")
        print(f"📊 数据类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"🔑 包含的键:")
            for key in checkpoint.keys():
                if key == 'model_state_dict':
                    state_dict = checkpoint[key]
                    print(f"   📋 {key}: {len(state_dict)} 个参数")
                    
                    # 显示前几个参数名
                    param_names = list(state_dict.keys())[:10]
                    print(f"      前10个参数: {param_names}")
                    
                elif key == 'best_metric':
                    metric = checkpoint[key]
                    print(f"   🏆 {key}: {metric}")
                else:
                    print(f"   📄 {key}: {checkpoint[key]}")
        else:
            # 旧格式，直接是state_dict
            print(f"📋 旧格式state_dict，包含 {len(checkpoint)} 个参数")
            param_names = list(checkpoint.keys())[:10]
            print(f"   前10个参数: {param_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_current_model_structure():
    """检查当前模型结构"""
    print(f"\n🔍 检查当前模型结构...")
    
    try:
        from models.FSRA.backbones.vision_mamba_lite import vision_mamba_lite_small_patch16_224_FSRA
        
        model = vision_mamba_lite_small_patch16_224_FSRA(
            img_size=(256, 256), 
            stride_size=16, 
            drop_rate=0.0, 
            local_feature=False
        )
        
        state_dict = model.state_dict()
        print(f"📋 当前模型包含 {len(state_dict)} 个参数")
        
        param_names = list(state_dict.keys())[:10]
        print(f"   前10个参数: {param_names}")
        
        return state_dict
        
    except Exception as e:
        print(f"❌ 创建当前模型失败: {e}")
        return None

def compare_model_compatibility(pretrain_path):
    """比较预训练模型和当前模型的兼容性"""
    print(f"\n🔄 比较模型兼容性...")
    
    # 检查预训练模型
    if not check_pretrain_model(pretrain_path):
        return
    
    # 检查当前模型
    current_state_dict = check_current_model_structure()
    if current_state_dict is None:
        return
    
    # 加载预训练模型
    try:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            pretrain_state_dict = checkpoint['model_state_dict']
        else:
            pretrain_state_dict = checkpoint
        
        # 比较参数
        current_keys = set(current_state_dict.keys())
        pretrain_keys = set(pretrain_state_dict.keys())
        
        common_keys = current_keys & pretrain_keys
        only_current = current_keys - pretrain_keys
        only_pretrain = pretrain_keys - current_keys
        
        print(f"\n📊 兼容性分析:")
        print(f"   🟢 共同参数: {len(common_keys)}")
        print(f"   🔴 仅当前模型: {len(only_current)}")
        print(f"   🟡 仅预训练模型: {len(only_pretrain)}")
        
        if only_current:
            print(f"\n❗ 仅当前模型有的参数 (前10个):")
            for key in list(only_current)[:10]:
                print(f"      {key}")
        
        if only_pretrain:
            print(f"\n❗ 仅预训练模型有的参数 (前10个):")
            for key in list(only_pretrain)[:10]:
                print(f"      {key}")
        
        # 检查形状兼容性
        shape_mismatches = 0
        for key in common_keys:
            if current_state_dict[key].shape != pretrain_state_dict[key].shape:
                shape_mismatches += 1
                if shape_mismatches <= 5:  # 只显示前5个
                    print(f"   ⚠️  形状不匹配 {key}: {current_state_dict[key].shape} vs {pretrain_state_dict[key].shape}")
        
        if shape_mismatches > 5:
            print(f"   ⚠️  还有 {shape_mismatches - 5} 个参数形状不匹配...")
        
        compatibility_rate = len(common_keys) / len(current_keys) * 100
        print(f"\n🎯 兼容性评分: {compatibility_rate:.1f}%")
        
        if compatibility_rate < 50:
            print(f"❌ 兼容性较低，建议重新训练或检查模型版本")
        elif compatibility_rate < 80:
            print(f"⚠️  兼容性中等，可能需要调整")
        else:
            print(f"✅ 兼容性良好")
            
    except Exception as e:
        print(f"❌ 比较失败: {e}")

if __name__ == "__main__":
    pretrain_path = "checkpoints/my_experiment/vision_mamba_lite_small_patch16_224_FSRA_best_accuracy_0.0029.pth"
    
    print(f"🚀 预训练模型兼容性检查")
    print(f"=" * 60)
    
    compare_model_compatibility(pretrain_path) 