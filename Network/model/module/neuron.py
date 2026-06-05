import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import seaborn
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate, layer, functional


# (这段代码需要 torch, torch.nn as nn)

class SparsityTracker:
    def __init__(self, model: nn.Module, layer_types, args):
        self.results = {}
        self.handles = []
        self.layer_types = layer_types
        self.register_hooks(model)
        self.args = args

    def _hook_fn(self, module, input, output, layer_name):
        if layer_name not in self.results:
            # 如果是，则初始化该层的累积数据结构
            self.results[layer_name] = {
                'total_zeros': 0,
                'total_elems': 0,
            }
        sparsity, zeros, elems = calculate_sparsity_after_transform(output)

        self.results[layer_name]['total_zeros'] += zeros
        self.results[layer_name]['total_elems'] += elems


    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                hook_with_name = lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
                
                handle = module.register_forward_hook(hook_with_name)
                self.handles.append(handle)
        print(f"成功在 {len(self.handles)} 个层上注册了hooks。")

    def report(self):
        """打印所有收集到的结果和整体平均值。"""
        if not self.results:
            print("没有收集到任何结果。请先运行模型的前向传播。")
            return

        report_lines = []
        report_lines.append("\n" + "="*50)
        report_lines.append("稀疏度分析报告")
        report_lines.append("="*50)

        overall_zeros = 0
        overall_elems = 0
        for name, result in self.results.items():
            total_zeros = result['total_zeros']
            total_elems = result['total_elems']
            overall_zeros += total_zeros
            overall_elems += total_elems
            report_lines.append(f"\n[层: {name}]")
            report_lines.append(f"  - 0的个数: {total_zeros}")
            report_lines.append(f"  - 元素个数: {total_elems}")
            report_lines.append(f"  - 本层平均稀疏度: {total_zeros/total_elems:.4f} ({total_zeros/total_elems:.2%})")
        
        overall_avg = overall_zeros / overall_elems
        report_lines.append("\n" + "-"*50)
        report_lines.append(f"模型整体平均稀疏度 (所有被追踪层): {overall_avg:.4f} ({overall_avg:.2%})")
        report_lines.append("="*50)

        filepath = self.args.model + '_sparsity.txt'
        if self.args.config is not None:
            filepath = self.args.model + '_' + os.path.basename(self.args.config) + '_sparsity.txt'
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已成功写入文件: {filepath}")
        except IOError as e:
            print(f"\n错误：无法将报告写入文件 {filepath}。原因: {e}")

    def remove_hooks(self):
        """移除所有已注册的hooks，清理资源。"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print(f"\n所有hooks已被移除。")
        
    def clear_results(self):
        """清空已保存的结果，以便进行新的分析。"""
        self.results = {}
        print("结果已清空。")

class SumTracker:
    def __init__(self, model: nn.Module, layer_types, args):
        self.results = {}
        self.handles = []
        self.layer_types = layer_types
        self.register_hooks(model)
        self.args = args

    def _hook_fn(self, module, input, output, layer_name):
        if layer_name not in self.results:
            # 如果是，则初始化该层的累积数据结构
            self.results[layer_name] = {
                'sum_elems': 0,
                'total_elems': 0,
            }
            self.results[layer_name]['sum_elems'] += output.sum()
            self.results[layer_name]['total_elems'] += output.numel() * 4


    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                hook_with_name = lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
                
                handle = module.register_forward_hook(hook_with_name)
                self.handles.append(handle)
        print(f"成功在 {len(self.handles)} 个层上注册了hooks。")

    def report(self):
        """打印所有收集到的结果和整体平均值。"""
        if not self.results:
            print("没有收集到任何结果。请先运行模型的前向传播。")
            return

        report_lines = []
        report_lines.append("\n" + "="*50)
        report_lines.append("元素和分析报告")
        report_lines.append("="*50)

        for name, result in self.results.items():
            report_lines.append(f"\n[层: {name}]")
            report_lines.append(f"  - 元素和: {result['sum_elems']}")
            report_lines.append(f"  - 总元素数: {result['total_elems']}")
            report_lines.append(f"  - 本层稀疏率: {result['sum_elems']/result['total_elems']:.4f} ({result['sum_elems']/result['total_elems']:.2%})")
    
        filepath = self.args.model + '_sum.txt'
        if self.args.config is not None:
            filepath = self.args.model + '_' + os.path.basename(self.args.config) + '_sum.txt'
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已成功写入文件: {filepath}")
        except IOError as e:
            print(f"\n错误：无法将报告写入文件 {filepath}。原因: {e}")

    def remove_hooks(self):
        """移除所有已注册的hooks，清理资源。"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print(f"\n所有hooks已被移除。")
        
    def clear_results(self):
        """清空已保存的结果，以便进行新的分析。"""
        self.results = {}
        print("结果已清空。")

class ZeroDistributionTracker:
    def __init__(self, model: nn.Module, layer_types=(nn.Conv2d,), args=None):
        self.results = {}
        self.handles = []
        self.layer_types = layer_types
        self.register_hooks(model)
        print("ZeroDistributionTracker 已初始化。")

    def _hook_fn(self, module, input, output, layer_name):
        """
        Hook函数：累积每个空间位置的零点计数。
        """
        # output shape: [B, C, H, W]
        output = output.squeeze(0)
        B, C, H, W = output.shape
        
        # 首次遇到该层时，初始化其数据结构
        if layer_name not in self.results:
            self.results[layer_name] = {
                # 使用 long 类型以避免计数溢出
                'zero_counts': torch.zeros(C, H, W, dtype=torch.long, device=output.device),
                'total_samples': 0
            }

        # 找到当前批次输出中的零点 (考虑浮点数精度)
        epsilon = 1e-8
        is_zero = torch.abs(output) < epsilon

        # 沿着批次维度(dim=0)求和，得到 [C, H, W] 的批次零点计数
        batch_zero_counts = is_zero.sum(dim=0)

        # 累积到总计数中
        self.results[layer_name]['zero_counts'] += batch_zero_counts
        self.results[layer_name]['total_samples'] += B

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                hook_with_name = lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
                handle = module.register_forward_hook(hook_with_name)
                self.handles.append(handle)
        print(f"成功在 {len(self.handles)} 个层上注册了零点分布追踪hooks。")
    
    def report(self, output_dir="zero_distribution_report"):
        """
        生成报告：将每个通道的零点频率热力图保存为图片，并生成一个总结文件。
        :param output_dir: 保存报告文件的目录。
        """
        if not self.results:
            print("没有收集到任何结果，无法生成报告。")
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        summary_lines = ["="*50, "零点分布分析报告", "="*50]

        for name, data in self.results.items():
            total_samples = data['total_samples']
            if total_samples == 0:
                continue

            # 计算频率: [C, H, W]
            zero_frequency_map = data['zero_counts'].float() / total_samples
            
            # 为该层创建一个子目录
            layer_dir = os.path.join(output_dir, name)
            os.makedirs(layer_dir, exist_ok=True)
            
            # 保存热力图
            # 我们将每个通道的热力图保存为一张图片网格
            # torchvision.utils.save_image 会自动将 [0, 1] 的值映射到 [0, 255] 的像素值
            filepath = os.path.join(layer_dir, f"{name}_zero_heatmap.png")
            # torchvision.utils.save_image(
            #     zero_frequency_map.unsqueeze(1), # 增加一个维度变为 [C, 1, H, W] 以便 save_image 处理
            #     filepath,
            #     normalize=False, # 我们的值已经在[0,1]范围内，不需要再标准化
            #     nrow=int(zero_frequency_map.shape[0]**0.5) # 尝试以方形排列网格
            # )
            plt.figure(figsize=(8, 6))
            for i in range(zero_frequency_map.shape[0]):
                seaborn.heatmap(zero_frequency_map[i].cpu().numpy(), vmin=0, vmax=1, cmap='viridis')
                plt.title(f"Layer: {name} Channel: {i} Zero Frequency Heatmap")
                plt.savefig(os.path.join(layer_dir, f"{name}_channel_{i}_zero_heatmap.png"), dpi=400)
                plt.clf()  # 清除当前图形，准备绘制下一个


            # 同时保存原始频率张量，以供后续精确分析
            torch.save(zero_frequency_map, os.path.join(layer_dir, f"{name}_zero_frequency.pt"))

            summary_lines.append(f"\n[层: {name}]")
            summary_lines.append(f"  - 特征图尺寸 (C, H, W): {tuple(zero_frequency_map.shape)}")
            summary_lines.append(f"  - 总计样本数: {total_samples}")
            summary_lines.append(f"  - 结果已保存至目录: {layer_dir}")

        # 写入总结文件
        try:
            with open(os.path.join(output_dir, "summary_report.txt"), 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))
            print(f"\n报告已成功生成至目录: {output_dir}")
        except IOError as e:
            print(f"\n错误：无法写入总结报告。原因: {e}")

    def clear_results(self): self.results = {}
    def remove_hooks(self): 
        for handle in self.handles: handle.remove()
        print(f"\n所有hooks已被移除。")


class TFlipTracker:
    def __init__(self, model: nn.Module, layer_types=(nn.Conv2d,), args=None):
        self.results = {}
        self.handles = []
        self.args = args
        self.layer_types = layer_types
        self.register_hooks(model)
        print("TFlipTracker 已初始化。")

    def _hook_fn(self, module, input, output, layer_name):
        """
        Hook函数：累积每个空间位置的零点计数。
        """
        # output shape: [B, C, H, W]
        output = output.squeeze(0)
        B, C, H, W = output.shape
        
        # 首次遇到该层时，初始化其数据结构
        if layer_name not in self.results:
            self.results[layer_name] = {
                # 使用 long 类型以避免计数溢出
                'flip_counts': torch.zeros(C, dtype=torch.long, device=output.device),
                'total_samples': 0
            }

        for c in range(C):
            for i, value in enumerate([1, 2, 3]):
                # 统计当前值 'value' 出现的次数
                count = torch.sum(output[:, c, :, :] == value).long()
                # 累加到对应的计数器中
                self.results[layer_name]['flip_counts'][c] += count

        # 累积到总计数中
        self.results[layer_name]['total_samples'] += H * W * B

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                hook_with_name = lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
                handle = module.register_forward_hook(hook_with_name)
                self.handles.append(handle)
        print(f"成功在 {len(self.handles)} 个层上注册了hooks。")
    
    def report(self):
        if not self.results:
            print("没有收集到任何结果。请先运行模型的前向传播。")
            return

        report_lines = []
        report_lines.append("\n" + "="*50)
        report_lines.append("TFlip分析报告")
        report_lines.append("="*50)

        for name, result in self.results.items():
            report_lines.append(f"\n[层: {name}]")
            report_lines.append(f"  - 翻转数: {result['flip_counts'].cpu().numpy().tolist()}")
            report_lines.append(f"  - 总元素数: {result['total_samples']}")
            report_lines.append(f"  - 本层翻转率: {(result['flip_counts']/result['total_samples']).cpu().numpy().tolist()}")

        filepath = self.args.model + '_tflip.txt'
        if self.args.config is not None:
            filepath = self.args.model + '_' + os.path.basename(self.args.config) + '_tflip.txt'
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已成功写入文件: {filepath}")
        except IOError as e:
            print(f"\n错误：无法将报告写入文件 {filepath}。原因: {e}")

    def clear_results(self): self.results = {}
    def remove_hooks(self): 
        for handle in self.handles: handle.remove()
        print(f"\n所有hooks已被移除。")


class CFlipTracker:
    def __init__(self, model: nn.Module, layer_types=(nn.Conv2d,), args=None, max_value=4):
        self.results = {}
        self.handles = []
        self.args = args
        self.layer_types = layer_types
        self.max_value = max_value
        self.register_hooks(model)
        print("CFlipTracker 已初始化。")

    def _hook_fn(self, module, input, output, layer_name):

        output = expand_tensor_cumulative(output, max_value=self.max_value).long()
        T, B, C, H, W = output.shape    
        # 首次遇到该层时，初始化其数据结构
        if layer_name not in self.results:
            self.results[layer_name] = {
                # 使用 long 类型以避免计数溢出
                'flip_counts': torch.zeros(T, dtype=torch.long, device=output.device),
                'total_samples': 0
            }

        for t in range(T):
            for c in range(C - 1):
                # 统计当前时间步和通道的翻转数
                count = torch.sum(output[t, :, c, :, :] != output[t, :, c+1, :, :]).long()
                self.results[layer_name]['flip_counts'][t] += count

        # 累积到总计数中
        self.results[layer_name]['total_samples'] += H * W * B * (C - 1)

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                hook_with_name = lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
                handle = module.register_forward_hook(hook_with_name)
                self.handles.append(handle)
        print(f"成功在 {len(self.handles)} 个层上注册了hooks。")
    
    def report(self):
        if not self.results:
            print("没有收集到任何结果。请先运行模型的前向传播。")
            return

        report_lines = []
        report_lines.append("\n" + "="*50)
        report_lines.append("CFlip分析报告")
        report_lines.append("="*50)

        for name, result in self.results.items():
            report_lines.append(f"\n[层: {name}]")
            report_lines.append(f"  - 翻转数: {result['flip_counts'].cpu().numpy().tolist()}")
            report_lines.append(f"  - 总元素数: {result['total_samples']}")
            report_lines.append(f"  - 本层翻转率: {(result['flip_counts']/result['total_samples']).cpu().numpy().tolist()}")

        filepath = self.args.model + '_cflip.txt'
        if self.args.config is not None:
            filepath = self.args.model + '_' + os.path.basename(self.args.config) + '_cflip.txt'
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已成功写入文件: {filepath}")
        except IOError as e:
            print(f"\n错误：无法将报告写入文件 {filepath}。原因: {e}")

    def clear_results(self): self.results = {}
    def remove_hooks(self): 
        for handle in self.handles: handle.remove()
        print(f"\n所有hooks已被移除。")


class FirstStepFlipTracker:
    def __init__(self, model: nn.Module, layer_types=(nn.Conv2d,), args=None, max_value=4):
        self.results = {}
        self.handles = []
        self.args = args
        self.layer_types = layer_types
        self.max_value = max_value
        self.register_hooks(model)
        print("FirstStepFlipTracker 已初始化。")

    def _hook_fn(self, module, input, output, layer_name):

        output = expand_tensor_cumulative(output, max_value=self.max_value).long()
        T, B, C, H, W = output.shape    
        # 首次遇到该层时，初始化其数据结构
        if layer_name not in self.results:
            self.results[layer_name] = {
                # 使用 long 类型以避免计数溢出
                'flip_counts': torch.zeros(C, dtype=torch.long, device=output.device),
                'total_samples': 0
            }

        base = output[0]
        kernel = torch.ones((C, 1, 3, 3), device=output.device)
        for t in range(1, T):
            cur = output[t]
            same = (cur == base).float()  # [B, C, H, W]
            self.results[layer_name]['flip_counts'] += (B * H * W - same.sum(dim=(0, 2, 3))).long()
            self.results[layer_name]['total_samples'] += B * H * W


    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                hook_with_name = lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
                handle = module.register_forward_hook(hook_with_name)
                self.handles.append(handle)
        print(f"成功在 {len(self.handles)} 个层上注册了hooks。")
    
    def report(self):
        if not self.results:
            print("没有收集到任何结果。请先运行模型的前向传播。")
            return

        report_lines = []
        report_lines.append("\n" + "="*50)
        report_lines.append("FirstStepFlipTracker分析报告")
        report_lines.append("="*50)

        for name, result in self.results.items():
            report_lines.append(f"\n[层: {name}]")
            report_lines.append(f"  - 翻转数: {result['flip_counts'].cpu().numpy().tolist()}")
            report_lines.append(f"  - 总元素数: {result['total_samples']}")
            report_lines.append(f"  - 本层翻转率: {(result['flip_counts']/result['total_samples']).cpu().numpy().tolist()}")

        filepath = self.args.model + '_firststepflip.txt'
        if self.args.config is not None:
            filepath = self.args.model + '_' + os.path.basename(self.args.config) + '_firststepflip.txt'
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已成功写入文件: {filepath}")
        except IOError as e:
            print(f"\n错误：无法将报告写入文件 {filepath}。原因: {e}")

    def clear_results(self): self.results = {}
    def remove_hooks(self): 
        for handle in self.handles: handle.remove()
        print(f"\n所有hooks已被移除。")


class XORTracker:
    def __init__(self, model: nn.Module, layer_types=(nn.Conv2d,), args=None, max_value=4):
        self.results = {}
        self.handles = []
        self.args = args
        self.layer_types = layer_types
        self.max_value = max_value
        self.register_hooks(model)
        print("XORTracker 已初始化。")

    def _hook_fn(self, module, input, output, layer_name):

        output = expand_tensor_cumulative(output, max_value=self.max_value).long()
        T, B, C, H, W = output.shape    
        # 首次遇到该层时，初始化其数据结构
        if layer_name not in self.results:
            self.results[layer_name] = {
                # 使用 long 类型以避免计数溢出
                'zero_counts': torch.zeros(C, dtype=torch.long, device=output.device),
                'total_samples': 0
            }

        base = output[0]
        kernel = torch.ones((C, 1, 3, 3), device=output.device)
        for t in range(1, T):
            cur = output[t]
            xor = (cur != base).float()  # [B, C, H, W]
            zero_count = F.conv2d(xor, kernel, groups=C)
            self.results[layer_name]['zero_counts'] += (zero_count == 0).sum(dim=(0, 2, 3)).long()
            self.results[layer_name]['total_samples'] += zero_count.numel() / C


    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, self.layer_types):
                hook_with_name = lambda m, i, o, name=name: self._hook_fn(m, i, o, name)
                handle = module.register_forward_hook(hook_with_name)
                self.handles.append(handle)
        print(f"成功在 {len(self.handles)} 个层上注册了hooks。")
    
    def report(self):
        if not self.results:
            print("没有收集到任何结果。请先运行模型的前向传播。")
            return

        report_lines = []
        report_lines.append("\n" + "="*50)
        report_lines.append("XORTracker分析报告")
        report_lines.append("="*50)

        for name, result in self.results.items():
            report_lines.append(f"\n[层: {name}]")
            report_lines.append(f"  - 全0数: {result['zero_counts'].cpu().numpy().tolist()}")
            report_lines.append(f"  - 总元素数: {result['total_samples']}")
            report_lines.append(f"  - 本层全0率: {(result['zero_counts']/result['total_samples']).cpu().numpy().tolist()}")

        filepath = self.args.model + '_XORTracker.txt'
        if self.args.config is not None:
            filepath = self.args.model + '_' + os.path.basename(self.args.config) + '_XORTracker.txt'
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            print(f"\n报告已成功写入文件: {filepath}")
        except IOError as e:
            print(f"\n错误：无法将报告写入文件 {filepath}。原因: {e}")

    def clear_results(self): self.results = {}
    def remove_hooks(self): 
        for handle in self.handles: handle.remove()
        print(f"\n所有hooks已被移除。")


def calculate_sparsity_after_transform(x: torch.Tensor) -> float:
    """
    对输入的5D张量执行分块、矩阵变换，并计算最终结果的稀疏率。

    参数:
        x (torch.Tensor): 输入张量，形状为 [T, B, C, H, W]。

    返回:
        float: 计算出的稀疏率 (0到1之间的浮点数)。
    """
    # 检查输入维度是否合法
    if x.dim() != 5:
        raise ValueError(f"输入张量必须是5维，但得到了 {x.dim()} 维")

    T, B, C, H, W = x.shape
    
    if H < 4 or W < 4:
        raise ValueError(f"H或W小于4，无法提取4x4的块。")

    # --- 步骤 1: 定义 B^T 和 B 矩阵 ---
    # B^T 来自您提供的图片
    BT = torch.tensor([
        [1,  0, -1,  0],
        [0,  1,  1,  0],
        [0, -1,  1,  0],
        [0,  1,  0, -1]
    ], dtype=x.dtype, device=x.device)

    # B 是 B^T 的转置
    B_mat = BT.T

    # --- 步骤 2: 维度转换 ---
    # 将 T, B, C 三个维度合并
    # 形状从 [T, B, C, H, W] -> [T*B*C, H, W]
    x_reshaped = x.view(-1, H, W)
    N = x_reshaped.shape[0] # N = T*B*C

    # --- 步骤 3: 滑窗分块 ---
    # 使用 unfold 高效提取所有 4x4 的块，步长为 2
    # 首先对 H 维度进行分块
    # [N, H, W] -> [N, num_patches_h, W, 4]
    patches_h = x_reshaped.unfold(dimension=1, size=4, step=2)
    
    # 然后对 W 维度进行分块
    # [N, num_patches_h, W, 4] -> [N, num_patches_h, num_patches_w, 4, 4]
    patches_hw = patches_h.unfold(dimension=2, size=4, step=2)
    
    # 为了方便进行批处理矩阵乘法，我们将所有块 reshape 成一个大的批次
    # [N, num_patches_h, num_patches_w, 4, 4] -> [Total_Patches, 4, 4]
    # contiguous() 确保张量在内存中是连续的，这是 view 操作有时所必需的
    d = patches_hw.contiguous().view(-1, 4, 4)
    
    if d.shape[0] == 0:
        print("警告: 没有成功的提取出任何4x4的块。返回稀疏率为0。")
        return 0.0

    # --- 步骤 4: 矩阵乘法 ---
    # 执行 B^T @ d @ B
    # PyTorch 的广播机制会自动处理批处理矩阵乘法
    # (4, 4) @ (N_patches, 4, 4) @ (4, 4) -> (N_patches, 4, 4)
    transformed_d = BT @ d @ B_mat

    # --- 步骤 5: 计算稀疏率 ---
    # 计算结果中等于0的元素总数
    # 为了处理浮点数精度问题，我们可以检查绝对值是否小于一个很小的数（epsilon）
    epsilon = 1e-8
    num_zeros = torch.sum(torch.abs(transformed_d) < epsilon).item()
    
    # 计算总元素数量
    total_elements = transformed_d.numel()
    
    # 计算稀疏率
    sparsity_rate = num_zeros / total_elements if total_elements > 0 else 0.0
    
    return sparsity_rate, num_zeros, total_elements

def expand_tensor_cumulative(tensor, max_value=4):

    T, B, C, H, W = tensor.shape
    # 创建一个 shape 为 [max_value, 1, 1, 1, 1, 1] 的比较向量
    steps = torch.arange(max_value, device=tensor.device).view(max_value, 1, 1, 1, 1, 1)

    # 扩展原始张量维度，便于比较 → [1, T, B, C, H, W]
    tensor_expanded = tensor.unsqueeze(0)

    # 比较：每个位置 v，生成 v 个 1，其余为 0
    binary = (steps < tensor_expanded).float()  # → shape [max_value, T, B, C, H, W]

    # 重新 reshape → [max_value * T, B, C, H, W]
    binary = binary.permute(1, 0, 2, 3, 4, 5).reshape(T * max_value, B, C, H, W)

    return binary

class quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=4): #1111
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class Quant(surrogate.SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)
        self.alpha = alpha
        self.spiking = spiking
    
    @staticmethod
    def spiking_function(x, alpha):
         return quant.apply(x, 0, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()

    def forward(self, x: torch.Tensor, alpha):
        if self.spiking:
            return self.spiking_function(x, alpha)
        else:
            return self.primitive_function(x, alpha)



class QIFNode(neuron.BaseNode):
    def __init__(self, v_threshold = 1.0, v_reset = None, T=4, surrogate_function = Quant(), detach_reset = False, step_mode='s', backend='torch', store_v_seq = False, bin=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.bin = bin
        self.T = T

    def neuronal_fire(self):
        assert isinstance(self.surrogate_function, Quant)
        if self.bin:
            return expand_tensor_cumulative(self.surrogate_function(self.v, alpha=self.T))
        else:
            return self.surrogate_function(self.v, alpha=self.T)

    def neuronal_charge(self, x: torch.Tensor):
        if self.bin:
            self.v = self.v + x.sum(0, keepdim=True)
        else:
            self.v = self.v + x

    def neuronal_reset(self, spike):
        if self.bin:
            self.v = self.v - spike.sum(0) * self.v_threshold
        else:
            self.v = self.v - spike * self.v_threshold

    def multi_step_forward(self, x_seq: torch.Tensor):
        y = self.single_step_forward(x_seq)
        return y
    
    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.mean(0, keepdim=True).data, v_init)


class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = True

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        # shape correction
        if self._spiking is not True and len(x.shape) == 5:
            x = x.mean([0])
        return x

def spike_activation(x, ste=False, temp=1.0):
    out_s = torch.gt(x, 0.5) # torch.gt:逐元素对比x>0.5?
    if ste:
        out_bp = torch.clamp(x, 0, 1) # 将x中的元素限制[0,1]范围内
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    return (out_s.float() - out_bp).detach() + out_bp


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def mem_update(x_in, mem, V_th, decay, grad_scale=1., temp=1.0):
    mem = mem * decay + x_in
    #if mem.shape[1]==256:
    #    embed()
    #V_th = gradient_scale(V_th, grad_scale)
    spike = spike_activation(mem / V_th, temp=temp)
    mem = mem * (1 - spike)
    #mem = 0
    #spike = spike * Fire_ratio
    return mem, spike


class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, step = 1, vth = 1.0):
        super(LIFAct, self).__init__()
        self.step = step
        #self.V_th = nn.Parameter(torch.tensor(1.))
        self.V_th = vth
        # self.tau = nn.Parameter(torch.tensor(-1.1))
        self.temp = 3.0
        #self.temp = nn.Parameter(torch.tensor(1.))
        self.grad_scale = 0.1

    def forward(self, x):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th,
                                  grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out
    

if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 3, 4, 5)
    a = nn.Sequential(
        layer.Conv2d(3, 3, 1),
        QIFNode(),
        layer.Conv2d(3, 3, 1),
        QIFNode()
    )
    functional.set_step_mode(a, 'm')
    tracker = SparsityTracker(a, layer_types=(QIFNode,))
    o = a(input_tensor)
    tracker.report()