#!/usr/bin/env python3
"""
UDD预测结果可视化示例脚本

使用方法：
python visualize_udd_example.py --checkpoint [检查点路径] --num_samples [要可视化的样本数量]

示例：
python visualize_udd_example.py --checkpoint QAT_checkpoint/udd/SpikingLETNet_shallowbs32gpu1_trainval20251112-113356/model_best.pth --num_samples 3
"""

import os
import sys
import subprocess

def main():
    # 检查是否有可用的检查点
    checkpoint_path = "QAT_checkpoint/udd/SpikingLETNet_shallowbs32gpu1_trainval20251112-113356/model_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        print("请确保检查点路径正确，或者使用其他检查点")
        return
    
    print(f"使用检查点: {checkpoint_path}")
    
    # 运行可视化脚本
    cmd = [
        "python", "visualize_udd.py",
        "--checkpoint", checkpoint_path,
        "--model", "SpikingLETNet_shallow",
        "--dataset", "udd",
        "--num_samples", "5",
        "--save_dir", "./udd_visualization_results",
        "--batch_size", "1",
        "--num_workers", "2"
    ]
    
    print("运行命令:", " ".join(cmd))
    print("开始可视化UDD预测结果...")
    
    try:
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)
        print("标准输出:")
        print(result.stdout)
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ 可视化完成！")
            print("结果保存在: ./udd_visualization_results/visualizations/")
            print("图例保存在: udd_legend.png")
        else:
            print(f"\n❌ 可视化失败，返回码: {result.returncode}")
            
    except Exception as e:
        print(f"运行可视化脚本时出错: {e}")
        print("\n您也可以直接运行:")
        print(f"python visualize_udd.py --checkpoint {checkpoint_path} --num_samples 5")

if __name__ == "__main__":
    main()
