import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import copy
from contextlib import contextmanager

# gen_inst_dir = "/data1/user22/verification_8_8/verification_logic/sim"
# if gen_inst_dir not in sys.path:
#     sys.path.append(gen_inst_dir)
from quantization.gen_inst import calculate_using_hardware, get_serial_port_list, open_serial_port

@contextmanager
def cd_to_tools():
    old_cwd = os.getcwd()  # 保存当前所在的目录
    os.chdir(gen_inst_dir)    # 切换到 gen_inst.py 的目录
    try:
        yield
    finally:
        os.chdir(old_cwd)  # 无论执行成功或报错，都跳回原目录

MAX_KERNEL_NUM = 64
Using_MODE = "vcs"
SER = None


def make_hardware_config():
    if Using_MODE == "uart" :
        # 获取可用的串口的列表
        port_list = get_serial_port_list()

        # 打开列表中的某个串口，进行数据收发
        if port_list:
            # 选择串口
            while True:
                portx_in_port_list = False
                portx = 'COM3'  #input("请输入要打开的串口的名称(例如：COM5)：")
                

                for my_port in port_list:
                    if portx == my_port[:4]:
                        portx_in_port_list = True
                        break
                if portx_in_port_list:
                    break
            # 设置其余参数
            
            bps = 115200
            timeout = 1
            stopbits = 1
            bytesize = 8
            parity = 'Odd'
            # 打开串口
            ser, successful = open_serial_port(portx, bps, timeout, stopbits, bytesize, parity)
            if successful:
                print(f"串口 {portx} 打开成功！")
            else:
                print(f"串口 {portx} 打开失败！")
                raise AssertionError("串口打开失败，程序终止。")
            
            return ser


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

def FC_hardware_compute_layerwise(input:torch.tensor, layer:nn.Linear):
    input_hardware = input.clone().detach()
    c_in = input_hardware.shape[0]
    c_out = input_hardware.shape[0]
    layer_weight = layer.weight.transpose(-2, -1).detach()
    p = 0
    k_h = 3
    k_w = 3
    H = input_hardware.shape[1]
    W = input_hardware.shape[2]
    s = 1
    T = 4
    output_h = input_hardware.shape[1]
    output_w = layer_weight.shape[1]
    d_h = 1
    d_w = 1
    depth = 1
    kernel_num = 1
    bias = layer.bias
    zhanwei_kernel = torch.round(torch.rand(3,3)).to(torch.int64).detach().cpu().numpy()

    kernel_size_list = [f'{k_h}*{k_w}' for i in range(kernel_num)]
    kernel_list = [zhanwei_kernel for i in range(kernel_num)]
    input_list = [np.vectorize(lambda v: format(int(v) & 0xF, "04b"))(input_hardware[i,:,:].view(H,W)) for i in range(kernel_num)]
    A2_list = [np.array(layer_weight, dtype=int) for i in range(kernel_num)]
    dilation_list = [f'{d_h}*{d_w}' for i in range(kernel_num)]
    step_mode_list = [f"{T}" for i in range(kernel_num)]
    scale_list = [0x0000 for i in range(kernel_num)]
    pe_mode_list = ['gemm' for i in range(kernel_num)]
    a1_height = H
    stride = s
    depadding_num = output_h
    padding_one_dilated = 0

    with cd_to_tools():
        outputs = calculate_using_hardware(input_list, A2_list, kernel_list, kernel_size_list, dilation_list, step_mode_list, scale_list, pe_mode_list, kernel_num, a1_height, stride, depadding_num, depth, padding_one_dilated, ser=SER, using_mode=Using_MODE)
    for i in range(len(outputs)):
        outputs[i] = binary_to_float_tensor(outputs[i])

    outputs = torch.stack(outputs, dim=0)

    if bias is not None:
        outputs += bias

    return outputs


def channel_parallel_compute_layerwise(input:torch.tensor, layer:nn.Conv2d, shift_mat, T=4, force_ann=False):
    input_hardware = input.clone().detach()
    # 确认 input 的时间步维度
    t_dimention=None
    if len(input_hardware.shape) == 5:
        # 消除时间步维度
        input_hardware = input_hardware.squeeze(0)
        t_dimention = 1
    
    # 确认 input 的有/无符号情况
    mode = 'snn'
    bias_output = None
    if torch.all(input_hardware >= 0) and torch.all(input_hardware < T):
        # 无符号
        mode = 'snn'
    elif torch.any(input_hardware < 0):
        # 有符号整体平移8
        mode = 'cnn'
        input_hardware += 8
        bias_input = torch.zeros_like(input_hardware) + 8
        if t_dimention is not None:
            bias_input = bias_input.unsqueeze(0)
        bias_output = layer(bias_input)
    else:
        mode = 'cnn'
    
    # force_ann 可强制使用 cnn 计算模式
    if force_ann:
        mode = 'cnn'

    
    # 手动对激活矩阵进行 padding 填充
    input_hardware = manual_padding(input_hardware, layer.padding)
    
    # 定义可能用到的参数
    groups = layer.groups
    conv_weights = layer.weight.data
    c_in = input_hardware.shape[1]
    c_out = conv_weights.shape[0]
    p=0
    k_h = conv_weights.shape[2]
    k_w = conv_weights.shape[3]
    H = input_hardware.shape[2]
    W = input_hardware.shape[3]
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


    # 将输入和卷积核通道进行拆分 并且分组调用硬件仿真 API, 最后完成通道合并累加
    if groups == 1:
        outputs = []
        for j in range(1):
            print(f"Computing output channel: {j}/{c_out}")
            shift = np.max(shift_mat[:, j])
            unaccumulated_outputs = []
            for t in range(len(kernel_num_queue)):
                kernel_num = kernel_num_queue[t]
                if (k_h, k_w) == (3, 3):
                    kernel_list = [conv_weights[j, i, :, :].view(k_h,k_w).to(torch.int64).detach().cpu().numpy() for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                    A2_list = [conv_weights[j,i,:,:].view(k_h,k_w).to(torch.int64).detach().cpu().numpy() for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                elif (k_h, k_w) == (1, 3):
                    kernel_list = []
                    A2_list = []
                    for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num):
                        single_channel_weight = conv_weights[j, i, :, :].view(k_h,k_w)
                        single_channel_weight_expanded = torch.cat([single_channel_weight for _ in range(3)], dim=0).view(3,3).to(torch.int64).detach().cpu().numpy()
                        kernel_list.append(single_channel_weight_expanded)
                        A2_list.append(single_channel_weight_expanded)
                elif (k_h, k_w) == (3, 1):
                    kernel_list = []
                    A2_list = []
                    for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num):
                        single_channel_weight = conv_weights[j, i, :, :].view(k_h,k_w).transpose(-2, -1)
                        single_channel_weight_expanded = torch.cat([single_channel_weight for _ in range(3)], dim=0).view(3,3).to(torch.int64).detach().cpu().numpy()
                        kernel_list.append(single_channel_weight_expanded)
                        A2_list.append(single_channel_weight_expanded)
                else:
                    raise AssertionError("kernel_size doesn't match.")

                kernel_size_list = [f'{k_h}*{k_w}' for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                input_list = [np.vectorize(lambda v: format(int(v) & 0xF, "04b"))(input_hardware[0,i,:,:].view(H,W)) for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                dilation_list = [f'{d_h}*{d_w}' for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                step_mode_list = [f"{T}" for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                scale_list = [shift for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                pe_mode_list = [mode for i in range(MAX_KERNEL_NUM*(t), MAX_KERNEL_NUM*(t)+kernel_num)]
                a1_height = H
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


                print(f"depadding_num: {depadding_num}")
                print(f"depth: {depth}")
                print(f"padding_one_dilated: {padding_one_dilated}")
                print(f"scale_list: {scale_list}")
                # 将分组传入仿真 API 计算 (待替换)
                with cd_to_tools():
                    unaccumulated_outputs_binary = calculate_using_hardware(input_list, A2_list, kernel_list, kernel_size_list, dilation_list, step_mode_list, scale_list, pe_mode_list, kernel_num, a1_height, stride, depadding_num, depth, padding_one_dilated, ser=SER, using_mode=Using_MODE)
                for binary_item in unaccumulated_outputs_binary:
                    unaccumulated_outputs.append(binary_to_float_tensor(binary_item))
            for kk in range(len(unaccumulated_outputs)):
                # print(f"hardware_mid_output: {unaccumulated_outputs[kk]}")
                unaccumulated_outputs[kk] = unaccumulated_outputs[kk].to(torch.int)
                unaccumulated_outputs[kk] <<= shift
                unaccumulated_outputs[kk] = unaccumulated_outputs[kk].to(torch.float)
            # 通道累加
            accumulated_outputs = torch.stack(unaccumulated_outputs, dim=0)
            accumulated_outputs = torch.sum(accumulated_outputs, dim=0, keepdim=False)
            outputs.append(accumulated_outputs)
        
        # 结果合并
        output = torch.stack(outputs, dim=0).unsqueeze(0)
        if t_dimention is not None:
            output = output.unsqueeze(0)
        
        if bias_output is not None:
            # 只减单通道
            if t_dimention is not None:
                output -= bias_output[:,:,0,:,:]
            else:
                output -= bias_output[:,0,:,:]
        
        if bias is not None:
            bias = bias.view(1, -1, 1, 1)
            if t_dimention is not None:
                bias = bias.unsqueeze(0)
                output += bias[:,:,0,:,:]
            else:
                output += bias[:,0,:,:]
        # 返回一致结果
        return output
    
    elif groups == input_hardware.shape[1] and groups == conv_weights.shape[0]:
        # 针对网络中的 groups 等于通道数进行特殊优化
        outputs = []
        shift = np.max(shift_mat)
        for t in range(len(kernel_num_queue)):
            kernel_num = 1
            if (k_h, k_w) == (3, 3):
                kernel_list = [conv_weights[j, 0, :, :].view(k_h,k_w).to(torch.int64).detach().cpu().numpy() for j in range(1)]
                A2_list = [conv_weights[j,0,:,:].view(k_h,k_w).to(torch.int64).detach().cpu().numpy() for j in range(1)]
            elif (k_h, k_w) == (1, 3):
                kernel_list = []
                A2_list = []
                for j in range(1):
                    single_channel_weight = conv_weights[j, 0, :, :].view(k_h,k_w)
                    single_channel_weight_expanded = torch.cat([single_channel_weight for _ in range(3)], dim=0).view(3,3).to(torch.int64).detach().cpu().numpy()
                    kernel_list.append(single_channel_weight_expanded)
                    A2_list.append(single_channel_weight_expanded)
            elif (k_h, k_w) == (3, 1):
                kernel_list = []
                A2_list = []
                for j in range(1):
                    single_channel_weight = conv_weights[j, 0, :, :].view(k_h,k_w).transpose(-2, -1)
                    single_channel_weight_expanded = torch.cat([single_channel_weight for _ in range(3)], dim=0).view(3,3).to(torch.int64).detach().cpu().numpy()
                    kernel_list.append(single_channel_weight_expanded)
                    A2_list.append(single_channel_weight_expanded)
            else:
                raise AssertionError("kernel_size doesn't match.")

            kernel_size_list = [f'{k_h}*{k_w}' for j in range(1)]
            input_list = [np.vectorize(lambda v: format(int(v) & 0xF, "04b"))(input_hardware[0,j,:,:].view(H,W)) for j in range(1)]
            dilation_list = [f'{d_h}*{d_w}' for j in range(1)]
            step_mode_list = [f"{T}" for i in range(1)]
            scale_list = [shift for j in range(1)]
            pe_mode_list = [mode for j in range(1)]
            a1_height = H
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


            print(f"depadding_num: {depadding_num}")
            print(f"depth: {depth}")
            print(f"padding_one_dilated: {padding_one_dilated}")
            # 将分组传入仿真 API 计算 (待替换)
            with cd_to_tools():
                single_channel_outputs_binary = calculate_using_hardware(input_list, A2_list, kernel_list, kernel_size_list, dilation_list, step_mode_list, scale_list, pe_mode_list, kernel_num, a1_height, stride, depadding_num, depth, padding_one_dilated, ser=SER, using_mode=Using_MODE)
            for binary_item in single_channel_outputs_binary:
                outputs.append(binary_to_float_tensor(binary_item))
        for kk in range(len(outputs)):
            outputs[kk] = outputs[kk].to(torch.int)
            outputs[kk] <<= shift
            outputs[kk] = outputs[kk].to(torch.float)

        # 结果合并
        output = torch.stack(outputs, dim=0).unsqueeze(0)
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
        return output

def dconv_hardware_compute_layerwise(input:torch.tensor, layer:nn.ConvTranspose2d, shift_mat, T=4, force_ann=False):
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
    output = channel_parallel_compute_layerwise(padded_input, conv_layer, shift_mat, T, force_ann)

    if t_dimension is not None:
        output = output.unsqueeze(0)

    # 6. 加上偏置
    if bias is not None:
        bias = bias.view(1, -1, 1, 1)
        if t_dimension is not None:
            bias = bias.unsqueeze(0)
        if t_dimension is not None:
            output += bias[:,:,0,:,:]
        else:
            output += bias[:,0,:,:]

    return output


if __name__ == '__main__':
    # 初始化硬件配置
    SER = make_hardware_config()

    def manual_get_shift(number):
        shift = 0
        tmp_num = number
        while tmp_num < -128 or tmp_num > 127:
            tmp_num >>= 1
            shift += 1
        return shift
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_project', default='conv', type=str)

    args = parser.parse_args()

    if args.test_project == 'conv':
        input = torch.clip(torch.round(8*torch.randn(1, 1, 400, 400)*2), min=-8, max=7)
        conv_layer = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1, padding=0, dilation=(1,1), groups=1, bias=False)
        conv_weights = torch.clip(torch.round(8*torch.randn(1, 1, 3, 3)*2), min=-8, max=7)
        conv_layer.weight.data = conv_weights
        
        input1 = input + 8
        bias_input = torch.ones_like(input1)*8
        original_output = conv_layer(input1)
        bias_output = conv_layer(bias_input)
        original_output = original_output.to(torch.int)
        scale_m = torch.max(torch.abs(original_output))
        shift = manual_get_shift(scale_m)
        original_output >>= shift
        original_output <<= shift
        original_output = original_output.to(torch.float)
        original_output = original_output - bias_output

        hardware_output = channel_parallel_compute_layerwise(input, conv_layer, shift, T=4, force_ann=False)
    
        print(f"input_shape: {input.shape}")
        print(f"kernel_size: {conv_weights.shape}")

        print(f"original_output_shape: {original_output.shape}")
        print(f"hardware_output_shape: {hardware_output.shape}")

        print(f"original_output:\n {original_output}")
        print(f"hardware_output:\n {hardware_output}")
        loss = nn.MSELoss()(original_output, hardware_output)
        print(f"loss: {loss}")

    elif args.test_project == 'gemm':
        input = torch.clip(torch.round(4*torch.rand(1, 4, 4)), min=-8, max=7) # C H W
        FC_layer = nn.Linear(4, 8, bias=False)
        FC_weight = torch.clip(torch.round(4*torch.randn(8, 4)), min=-8, max=7)
        FC_layer.weight.data = FC_weight

        output = FC_layer(input)

        hardware_output = FC_hardware_compute(input, FC_layer)
        print(output)
        print(hardware_output)
        print(f"MSELoss: {nn.MSELoss()(output, hardware_output)}")

    elif args.test_project == 'dconv':
        dconv_layer = nn.ConvTranspose2d(1, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=True)
        random_input = torch.clip(torch.round(4*torch.rand(1, 1, 20, 20)), min=-8, max=7)
        random_weight = torch.clip(torch.round(torch.randn(1, 3, 3, 3)), min=-8, max=7)
        dconv_layer.weight.data = random_weight

        original_output = dconv_layer(random_input)
        hardware_output = dconv_hardware_compute_layerwise(random_input, dconv_layer)
        print("original_output shape: ", original_output.shape)
        print("hardware_output shape: ", hardware_output.shape)

        print("loss: ", nn.MSELoss()(original_output, hardware_output))





