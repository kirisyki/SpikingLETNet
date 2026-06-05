import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from contextlib import contextmanager
MAX_KERNEL_NUM = 10004

def manual_get_shift(number):
    shift = 0
    tmp_num = number
    while tmp_num < -128 or tmp_num > 127:
        tmp_num >>= 1
        shift += 1
    return shift

def manual_padding(input:torch.tensor, p:tuple):
    p_h, p_w = p
    padded_input = F.pad(input, (p_w, p_w, p_h, p_h))
    return padded_input  

def binary_to_float_tensor(binary_list_2d, scale=1.0, bit_width=16):
    """
    将二进制字符串的二维列表转为浮点数 torch.tensor
    :param binary_list_2d: 嵌套列表，内部是二进制字符串，如 [['0001', '1110'], ...]
    :param scale: 量化时使用的缩放因子 (scale_x)
    :param bit_width: 二进制的位数
    :return: torch.FloatTensor
    """
    def bin_to_int(s):
        # 处理补码：如果首位是 1，表示负数
        val = int(s, 2)
        if s[0] == '1':
            val -= (1 << len(s))
        return val

    # 1. 将所有字符串转为整数
    int_data = [[bin_to_int(s) for s in row] for row in binary_list_2d]
    
    # 2. 转为 torch tensor
    tensor_int = torch.tensor(int_data, dtype=torch.float32)


    return tensor_int

def FC_simulation(input:torch.tensor, layer:nn.Linear):
    input_torch = input.clone().detach()
    c_in = input_torch.shape[0]
    c_out = input_torch.shape[0]
    layer_weight = layer.weight.transpose(-2, -1).detach()
    p = 0
    k_h = 3
    k_w = 3
    H = input_torch.shape[1]
    W = input_torch.shape[2]
    s = 1
    T = 4
    output_h = input_torch.shape[1]
    output_w = layer_weight.shape[1]
    d_h = 1
    d_w = 1
    depth = 1
    kernel_num = c_in
    bias = layer.bias
    zhanwei_kernel = torch.round(torch.rand(3,3)).to(torch.int64).detach().cpu().numpy()


    input_list = [input_torch[i,:,:].view(H,W) for i in range(kernel_num)]
    A2_list = [layer_weight for i in range(kernel_num)]

    outputs = []
    for i in range(kernel_num):
        outputs.append(torch.matmul(input_list[i], A2_list[i]))

    outputs = torch.stack(outputs, dim=0)

    if bias is not None:
        outputs += bias

    return outputs


def conv_simulation(input:torch.tensor, layer:nn.Conv2d, T=4):
    input_torch = input.clone().detach()
    # 确认 input 的时间步维度
    t_dimention=None
    if len(input_torch.shape) == 5:
        # 消除时间步维度
        input_torch = input_torch.squeeze(0)
        t_dimention = 1
    
    # 确认 input 的有/无符号情况
    mode = 'snn'
    bias_output = None
    if torch.all(input_torch >= 0) and torch.all(input_torch < T):
        # 无符号
        mode = 'snn'
    else:
        # 有符号, 加 8 转为无符号数
        input_torch += 8
        mode = 'cnn'
        bias_input = torch.zeros_like(input_torch) + 8
        if t_dimention is not None:
            bias_input = bias_input.unsqueeze(0)
        bias_output = layer(bias_input)

    # 手动对激活矩阵进行 padding 填充
    input_torch = manual_padding(input_torch, layer.padding)
    
    # 定义可能用到的参数
    groups = layer.groups
    conv_weights = layer.weight.data
    c_in = input_torch.shape[1]
    c_out = conv_weights.shape[0]
    p=0
    k_h = conv_weights.shape[2]
    k_w = conv_weights.shape[3]
    H = input_torch.shape[2]
    W = input_torch.shape[3]
    s, _ = layer.stride
    dilation = layer.dilation
    d_h, d_w = dilation
    T = T
    output_h = int((H+2*p-d_h*(k_h-1)-1)/s) + 1
    output_w = int((W+2*p-d_w*(k_w-1)-1)/s) + 1
    total_kernel_num = c_in
    kernel_num_queue = []
    bias = layer.bias
    while total_kernel_num > MAX_KERNEL_NUM:
        kernel_num_queue.append(MAX_KERNEL_NUM)
        total_kernel_num -= MAX_KERNEL_NUM

    kernel_num_queue.append(total_kernel_num)

    # assert kernel_num <= MAX_KERNEL_NUM, f"kernel num is toot big: {kernel_num} > MAX_KERNEL_NUM={MAX_KERNEL_NUM}"
    simulation_conv = nn.Conv2d(1, 1, kernel_size=(k_h, k_w), padding=0, dilation=(d_h, d_w), stride=(s,s), bias=False)
    shift_mat = np.zeros(shape=(c_in, c_out), dtype=np.int16)
    # 将输入和卷积核通道进行拆分 并且分组调用硬件仿真 API, 最后完成通道合并累加
    if groups == 1:
        outputs = []
        for j in range(c_out):
            unaccumulated_outputs = []
            for t in range(len(kernel_num_queue)):
                kernel_num = kernel_num_queue[t]
                kernel_list = [conv_weights[j, i, :, :].view(k_h,k_w) for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                A2_list = [conv_weights[j,i,:,:].view(k_h,k_w) for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]

                kernel_size_list = [f'{k_h}*{k_w}' for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                input_list = [input_torch[0,i,:,:].view(H,W) for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                stride = s
                depth = 1
                padding_one_dilated = 0
                if (k_h, k_w)==(3, 1):
                    depadding_num = output_w
                else:
                    depadding_num = output_h
                
                if (k_h, k_w) == (3, 1) and dilation == (2, 1):
                    depth = int((output_h+1)/2)
                    padding_one_dilated = output_h % 2
                elif (k_h, k_w) == (1, 3) and dilation == (1, 2):
                    depth = int((output_w+1)/2)
                    padding_one_dilated = output_w % 2


                # print(f"depadding_num: {depadding_num}")
                # print(f"depth: {depth}")
                # print(f"padding_one_dilated: {padding_one_dilated}")
                # 将分组传入仿真 API 计算 (待替换)
                for i in range(kernel_num):
                    simulation_conv.weight.data = kernel_list[i].unsqueeze(0).unsqueeze(0)
                    single_channel_output = simulation_conv(input_list[i].unsqueeze(0).unsqueeze(0))
                    scale_m = torch.max(torch.abs(single_channel_output))
                    shift = manual_get_shift(int(scale_m))
                    shift_mat[i, j] = shift
                    unaccumulated_outputs.append(single_channel_output)
                shift = np.max(shift_mat[:, j])
                for i in range(len(unaccumulated_outputs)):
                    single_channel_output = unaccumulated_outputs[i]
                    single_channel_output = single_channel_output.to(torch.int)
                    single_channel_output >>= shift
                    # print(f"pytorch_mid_output: {single_channel_output}")
                    single_channel_output <<= shift
                    single_channel_output = single_channel_output.to(torch.float)
                    unaccumulated_outputs[i] = single_channel_output
            # 通道累加
            accumulated_outputs = torch.stack(unaccumulated_outputs, dim=0)
            accumulated_outputs = torch.sum(accumulated_outputs, dim=0, keepdim=False)
            outputs.append(accumulated_outputs)
        
        # 结果合并
        output = torch.cat(outputs, dim=1)
        if t_dimention is not None:
            output = output.unsqueeze(0)
        
        if bias_output is not None:
            output -= bias_output

        if bias is not None:
            bias = bias.view(1, -1, 1, 1)
            if t_dimention is not None:
                bias = bias.unsqueeze(0)
            output += bias
        # 返回一致结果
        return output, shift_mat
    
    elif groups == input_torch.shape[1] and groups == conv_weights.shape[0]:
        # 针对网络中的 groups 等于通道数进行特殊优化
        outputs = []
        for t in range(len(kernel_num_queue)):
            kernel_num = kernel_num_queue[t]
            kernel_list = [conv_weights[j, 0, :, :].view(k_h,k_w) for j in range(c_out)]
            A2_list = [conv_weights[j,0,:,:].view(k_h,k_w) for j in range(c_out)]

            kernel_size_list = [f'{k_h}*{k_w}' for j in range(c_out)]
            input_list = [input_torch[0,j,:,:].view(H,W) for j in range(c_out)]
            stride = s
            depth = 1
            padding_one_dilated = 0
            if (k_h, k_w)==(3, 1):
                depadding_num = output_w
            else:
                depadding_num = output_h
            
            if (k_h, k_w) == (3, 1) and dilation == (2, 1):
                depth = int((output_h+1)/2)
                padding_one_dilated = output_h % 2
            elif (k_h, k_w) == (1, 3) and dilation == (1, 2):
                depth = int((output_w+1)/2)
                padding_one_dilated = output_w % 2


            # print(f"depadding_num: {depadding_num}")
            # print(f"depth: {depth}")
            # print(f"padding_one_dilated: {padding_one_dilated}")
            # 将分组传入仿真 API 计算 (待替换)
            for L in range(kernel_num):
                simulation_conv.weight.data = kernel_list[L].unsqueeze(0).unsqueeze(0)
                single_channel_output = simulation_conv(input_list[L].unsqueeze(0).unsqueeze(0))
                scale_m = torch.max(torch.abs(single_channel_output))
                shift = manual_get_shift(int(scale_m))
                shift_mat[L, L] = shift
                outputs.append(single_channel_output) 
            shift = np.max(shift_mat)
            for i in range(len(outputs)):
                single_channel_output = outputs[i]
                single_channel_output = single_channel_output.to(torch.int)
                single_channel_output >>= shift
                single_channel_output <<= shift
                single_channel_output = single_channel_output.to(torch.float)
                outputs[i] = single_channel_output
        # 结果合并
        output = torch.cat(outputs, dim=1)
        if t_dimention is not None:
            output = output.unsqueeze(0)

        if bias_output is not None:
            output -= bias_output

        if bias is not None:
            bias = bias.view(1, -1, 1, 1)
            if t_dimention is not None:
                bias = bias.unsqueeze(0)
            output += bias

        # 返回一致结果
        return output, shift_mat

def dconv_simulation(input:torch.tensor, layer:nn.ConvTranspose2d, T=4):
    # 1. 获取参数
    t_dimension = None
    if len(input.shape) == 5:
        # 消除时间步维度
        t_dimension = 1
        input = input.squeeze(0)

    weight = layer.weight.data
    stride = layer.stride[0]
    padding = layer.padding[0]
    output_padding = layer.output_padding[0]
    bias = layer.bias

    batch_size, in_channels, iH, iW = input.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    
    # 2. 对输入进行插空补零 (Stride Handling)
    if stride > 1:
        new_h, new_w = iH + (iH - 1) * (stride - 1), iW + (iW - 1) * (stride - 1)
        spaced_input = torch.zeros((batch_size, in_channels, new_h, new_w), device=input.device)
        spaced_input[:, :, ::stride, ::stride] = input
    else:
        spaced_input = input

    # 3. 计算实际的 Padding 边界
    # PyTorch 的逻辑是：先在四周补 (kH - 1 - padding) 的零
    # 然后在右侧和下方额外增加 output_padding 数量的零，让卷积核能扫过去
    pad_top = kH - 1 - padding
    pad_left = kW - 1 - padding
    pad_bottom = kH - 1 - padding + output_padding
    pad_right = kW - 1 - padding + output_padding
    
    # 使用不对应的 padding (左, 右, 上, 下)
    padded_input = F.pad(spaced_input, (pad_left, pad_right, pad_top, pad_bottom))

    # 4. 权重处理
    flipped_weight = torch.flip(weight, dims=[2, 3]).transpose(0, 1)

    # 5. 调用卷积算子 (此时无需再手动补 output_padding)
    # 注意：in_channels 和 out_channels 在反卷积权重中是转置的
    conv_layer = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=(kH, kW), 
        stride=1, 
        padding=0, 
        bias=False
    )

    conv_layer.weight.data = flipped_weight
    output, shift_mat = conv_simulation(padded_input, conv_layer, T)

    if t_dimension is not None:
        output = output.unsqueeze(0)

    # 6. 加上偏置
    if bias is not None:
        bias = bias.view(1, -1, 1, 1)
        if t_dimension is not None:
            bias = bias.unsqueeze(0)
        output += bias

    return output, shift_mat



