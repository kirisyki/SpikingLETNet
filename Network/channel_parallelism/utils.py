import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import seaborn
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from spikingjelly.activation_based import neuron, surrogate, layer, functional

# 这个函数应该替换成仿真实现的 API 接口 
def multi_kernel_compute(input_list, kernel_list, stride_list, dilation_list, T_list, scale_m, mode_list, kernel_num=1):

    """
    input_list & kernel_list: 按顺序传入的激活图和卷积核

    stride_list & dilation_list: 必要计算超参 应该是元组 写成 a*b 

    T_list: 时间步 eg: when T=4, 3 -> 0111 

    scale_m: 中间值量化缩放因子

    mode_list: 'snn' or 'ann' or 'gemm'

    kernel_num: 使用多少个核并, 这里以 1 举例 

    """

    # 应用并行的仿真运算实现 并且按顺序加入一个 output_list 列表 (待替换)
    conv_layer = nn.Conv2d(1, 1, kernel_size=(kernel_list[0].shape[2], kernel_list[0].shape[3]), padding=0, stride=stride_list[0], dilation=dilation_list[0], bias=False)
    output_list = []
    with torch.no_grad():
        for i in range(len(input_list)):
            single_input = input_list[i]
            single_kernel = kernel_list[i]
            conv_layer.weight.data = single_kernel
            single_output = conv_layer(single_input)
            output_list.append(single_output)
    
    # 返回 output_list
    return output_list



def channel_parallel_compute(input:torch.tensor, layer:nn.Conv2d, dilation=1, T=4):
    # 定义可能用到的参数
    conv_weights = layer.weight.data
    c_in = conv_weights.shape[1]
    c_out = conv_weights.shape[0]
    k = conv_weights.shape[2]
    p = layer.padding
    s = layer.stride
    d = dilation
    t = T
    scale_m = 2
    H = input.shape[2]
    W = input.shape[3]

    # 将输入和卷积核通道进行拆分 并且分组调用硬件仿真 API, 最后完成通道合并累加
    outputs = []
    for j in tqdm(range(c_out)):
        # 创建分组
        kernel_list = []
        input_list = []
        stride_list = []
        dilation_list = []
        mode_list = []
        T_list = []
        for i in range(c_in):
            single_conv_weights = conv_weights[j, i, :, :].view(1,1,k,k)
            kernel_list.append(single_conv_weights)
            single_input = input[:, i, :, :].view(1,1,H,W)
            input_list.append(single_input)
            stride_list.append(s)
            dilation_list.append(d)
            mode_list.append('snn')
            T_list.append(t)
        
        # 将分组传入仿真 API 计算 (待替换)
        unaccumulated_outputs = multi_kernel_compute(input_list, kernel_list, stride_list, dilation_list, T_list, scale_m, mode_list)

        # 通道累加
        accumulated_outputs = torch.stack(unaccumulated_outputs, dim=0)
        accumulated_outputs = torch.sum(accumulated_outputs, dim=0, keepdim=False)
        outputs.append(accumulated_outputs)
    
    # 结果合并
    output = torch.cat(outputs, dim=1)

    # 返回一致结果
    return output

if __name__ == '__main__':
    input = torch.randn(1, 10, 32, 32) # B C H W
    conv_layer = nn.Conv2d(10, 32, kernel_size=3, padding=0, stride=1, bias=False)
    output = channel_parallel_compute(input, conv_layer)
    original_output = conv_layer(input)
    loss = nn.MSELoss()(output, original_output)
    print(f"loss between different calculation method: {loss}")

    # print(f"output: {output}")
    # print(f"original_output: {original_output}")
    # print(f"computational results are identical: {output==original_output}")
    # conv_weight = conv_layer.weight.data
    # print(f"conv_weight.shape: {conv_weight.shape}")
