"""
模型管理使用示例
演示如何使用新的最佳模型保存功能
"""

print("🚀 Vision Mamba 最佳模型保存功能使用指南")
print("=" * 60)

print("\n1️⃣ 训练时自动保存最佳模型:")
print("python train.py --backbone MAMBA-LITE --optimizer adamw --lr 0.001 --name my_experiment")
print("   ✅ 只保存性能最佳的模型")
print("   ✅ 模型名称: vision_mamba_lite_small_patch16_224_FSRA_best_accuracy_X.XXX.pth")
print("   ✅ 自动更新最新副本: vision_mamba_lite_small_patch16_224_FSRA_latest.pth")

print("\n2️⃣ 自定义模型名称:")
print("python train.py --backbone MAMBA-LITE --custom_model_name my_custom_mamba --name my_experiment")
print("   ✅ 模型将保存为: my_custom_mamba_best_accuracy_X.XXX.pth")

print("\n3️⃣ 控制checkpoint保存频率:")
print("python train.py --backbone MAMBA-LITE --save_checkpoint_freq 20 --name my_experiment")
print("   ✅ 每20轮保存一次训练checkpoint")

print("\n4️⃣ 使用示例:")
example_commands = [
    {
        "backbone": "MAMBA-LITE",
        "desc": "轻量级Vision Mamba",
        "lr": "0.001",
        "batch": "32"
    },
    {
        "backbone": "MAMBA-V2", 
        "desc": "完整Vision Mamba",
        "lr": "0.0005", 
        "batch": "16"
    },
    {
        "backbone": "VIT-S",
        "desc": "Vision Transformer",
        "lr": "0.01",
        "batch": "8"
    }
]

for i, cmd in enumerate(example_commands, 1):
    print(f"\n   示例{i} - {cmd['desc']}:")
    print(f"   python train.py --backbone {cmd['backbone']} --lr {cmd['lr']} --batchsize {cmd['batch']} --name exp_{cmd['backbone'].lower()}")

print("\n💡 训练过程中的输出示例:")
print("   🏆 首次设定最佳准确率: 0.6234")
print("   💾 最佳准确率模型已更新并保存")
print("   🏆 发现更好的准确率! 0.6587 > 0.6234 (之前最佳)")
print("   💾 最佳准确率模型已更新并保存")
print("   📊 第25轮: 准确率 0.6412 (最佳: 0.6587)")

print("\n📁 保存的文件结构:")
print("./checkpoints/my_experiment/")
print("├── vision_mamba_lite_small_patch16_224_FSRA_best_accuracy_0.6587.pth  # 最佳模型")
print("├── vision_mamba_lite_small_patch16_224_FSRA_latest.pth                # 最新副本")
print("├── net_009.pth  # checkpoint (第10轮)")
print("├── net_019.pth  # checkpoint (第20轮)")
print("└── ...")

print("\n🔄 如何加载最佳模型:")
print("```python")
print("from tool.utils_server import load_network_with_name, list_saved_models")
print("from models.FSRA.backbones.vision_mamba_lite import vision_mamba_lite_small_patch16_224_FSRA")
print("")
print("# 创建模型")
print("model = vision_mamba_lite_small_patch16_224_FSRA()")
print("")
print("# 加载最佳模型")
print("model_info = load_network_with_name(model, './checkpoints/my_experiment/vision_mamba_lite_small_patch16_224_FSRA_best_accuracy_0.6587.pth')")
print("")
print("# 查看所有保存的模型")
print("list_saved_models('my_experiment')")
print("```")

print("\n🎯 优势总结:")
print("   ✅ 不再保存120个模型文件，只保存最佳的1个")
print("   ✅ 自动跟踪最佳性能，无需手动比较")  
print("   ✅ 支持自定义模型名称")
print("   ✅ 保留checkpoint机制用于调试")
print("   ✅ 完整的模型信息保存（epoch、性能指标等）")
print("   ✅ 训练过程实时显示性能改进")

print("\n🚀 开始训练吧!")
print("推荐命令: python train.py --backbone MAMBA-LITE --optimizer adamw --lr 0.001 --name my_best_model") 