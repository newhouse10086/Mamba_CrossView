#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据路径测试脚本
用于检查数据集是否正确放置和组织
"""

import os
import glob

def check_data_structure(data_root):
    """检查数据结构是否符合要求"""
    print(f"检查数据根目录: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"❌ 数据根目录不存在: {data_root}")
        return False
    
    print(f"✅ 数据根目录存在: {data_root}")
    
    # 检查子目录
    expected_dirs = ['satellite', 'street', 'drone']
    found_dirs = []
    
    for subdir in expected_dirs:
        subdir_path = os.path.join(data_root, subdir)
        if os.path.exists(subdir_path):
            found_dirs.append(subdir)
            print(f"✅ 找到子目录: {subdir}")
            
            # 检查类别数量
            classes = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
            print(f"   - 类别数量: {len(classes)}")
            
            # 检查前几个类别的图片数量
            for i, cls in enumerate(classes[:3]):
                cls_path = os.path.join(subdir_path, cls)
                images = glob.glob(os.path.join(cls_path, '*.*'))
                images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                print(f"   - 类别 {cls}: {len(images)} 张图片")
                
        else:
            print(f"⚠️  未找到子目录: {subdir}")
    
    if len(found_dirs) == 0:
        print("❌ 没有找到任何预期的子目录")
        return False
    elif len(found_dirs) < len(expected_dirs):
        print(f"⚠️  只找到部分子目录: {found_dirs}")
        print("   程序会自动适配现有的目录")
        return True
    else:
        print("✅ 所有预期的子目录都存在")
        return True

def main():
    print("=== 数据路径检查工具 ===\n")
    
    # 默认路径
    default_paths = [
        "/home/ma-user/work/Mamba_CrossView/data/University-123/train",
        "/home/ma-user/work/Mamba_CrossView/data/University-123/test",
        "./data/University-123/train",
        "./data/University-123/test"
    ]
    
    print("检查默认路径：")
    for path in default_paths:
        print(f"\n{'='*50}")
        check_data_structure(path)
    
    # 允许用户输入自定义路径
    print(f"\n{'='*50}")
    custom_path = input("请输入自定义数据路径 (直接回车跳过): ").strip()
    if custom_path:
        check_data_structure(custom_path)
    
    print("\n=== 检查完成 ===")
    print("\n如果数据路径检查通过，您可以使用以下命令训练：")
    print("python train.py --backbone MAMBA-S --data_dir YOUR_DATA_PATH --lr 0.01 --gpu_ids 0")

if __name__ == "__main__":
    main() 