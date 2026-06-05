import numpy as np
import torch
import torch.nn as nn
from gen_inst_fixed import calculate_using_hardware, get_serial_port_list, open_serial_port

MAX_KERNEL_NUM = 10
Using_MODE = "uart"

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

def FC_hardware_calculate(inptut:torch.tensor, layer:nn.Linear):
    c_in = input.shape[0]
    c_out = input.shape[0]
    layer_weight = layer.weight.transpose(-2, -1).detach()
    p = 0
    k_h = 3
    k_w = 3
    H = input.shape[1]
    W = input.shape[2]
    s = 1
    T = 4
    output_h = input.shape[1]
    output_w = layer_weight.shape[1]
    d_h = 1
    d_w = 1
    depth = 1
    kernel_num = c_in
    zhanwei_kernel = torch.round(torch.rand(3,3)).to(torch.int64).detach().cpu().numpy()

    kernel_size_list = [f'{k_h}*{k_w}' for i in range(kernel_num)]
    kernel_list = [zhanwei_kernel for i in range(kernel_num)]
    input_list = [np.vectorize(lambda v: format(int(v), "04b"))(input[i,:,:].view(H,W)) for i in range(kernel_num)]
    A2_list = [np.array(layer_weight, dtype=int).reshape(output_h, output_w)]
    dilation_list = [f'{d_h}*{d_w}' for i in range(kernel_num)]
    step_mode_list = [f"{T}" for i in range(kernel_num)]
    scale_list = [0x0000 for i in range(kernel_num)]
    pe_mode_list = ['gemm' for i in range(kernel_num)]
    a1_height = output_w
    stride = s
    depadding_num = output_h
    padding_one_dilated = 0

    output = calculate_using_hardware(input_list, A2_list, kernel_list, kernel_size_list, dilation_list, step_mode_list, scale_list, pe_mode_list, kernel_num, a1_height, stride, depadding_num, depth, padding_one_dilated, ser=SER, using_mode=Using_MODE)

    return binary_to_float_tensor(output[0])


def channel_parallel_compute(input:torch.tensor, layer:nn.Conv2d, dilation=(1,1), T=4):
    # 定义可能用到的参数
    conv_weights = layer.weight.data
    c_in = conv_weights.shape[1]
    c_out = conv_weights.shape[0]
    p = 0
    k_h = conv_weights.shape[2]
    k_w = conv_weights.shape[3]
    H = input.shape[2]
    W = input.shape[3]
    s, _ = layer.stride
    d_h, d_w = dilation
    T = T
    output_h = int((H+2*p-d_h*(k_h-1)-1)/s) + 1
    output_w = int((W+2*p-d_w*(k_w-1)-1)/s) + 1
    kernel_num = c_in

    assert kernel_num <= MAX_KERNEL_NUM, f"kernel num is toot big: {kernel_num} > MAX_KERNEL_NUM={MAX_KERNEL_NUM}"

    # 将输入和卷积核通道进行拆分 并且分组调用硬件仿真 API, 最后完成通道合并累加
    outputs = []
    for j in range(c_out):
        if (k_h, k_w) == (3, 3):
            kernel_list = [conv_weights[j, i, :, :].view(k_h,k_w).to(torch.int64).detach().cpu().numpy() for i in range(kernel_num)]
            A2_list = [conv_weights[j,i,:,:].view(k_h,k_w).to(torch.int64).detach().cpu().numpy() for i in range(kernel_num)]
        elif (k_h, k_w) == (1, 3):
            kernel_list = []
            A2_list = []
            for i in range(kernel_num):
                single_channel_weight = conv_weights[j, i, :, :].view(k_h,k_w)
                single_channel_weight_expanded = torch.cat([single_channel_weight for _ in range(3)], dim=0).view(3,3).to(torch.int64).detach().cpu().numpy()
                kernel_list.append(single_channel_weight_expanded)
                A2_list.append(single_channel_weight_expanded)
        elif (k_h, k_w) == (3, 1):
            kernel_list = []
            A2_list = []
            for i in range(kernel_num):
                single_channel_weight = conv_weights[j, i, :, :].view(k_h,k_w).transpose(-2, -1)
                single_channel_weight_expanded = torch.cat([single_channel_weight for _ in range(3)], dim=0).view(3,3).to(torch.int64).detach().cpu().numpy()
                kernel_list.append(single_channel_weight_expanded)
                A2_list.append(single_channel_weight_expanded)
        else:
            raise AssertionError("kernel_size doesn't match.")

        kernel_size_list = [f'{k_h}*{k_w}' for i in range(kernel_num)]
        input_list = [np.vectorize(lambda v: format(int(v), "04b"))(input[:,i,:,:].view(H,W)) for i in range(kernel_num)]
        dilation_list = [f'{d_h}*{d_w}' for i in range(kernel_num)]
        step_mode_list = [f"{T}" for i in range(kernel_num)]
        scale_list = [0x0000 for i in range(kernel_num)]
        pe_mode_list = ['snn' for i in range(kernel_num)]
        a1_height_list = [f"H" for i in range(kernel_num)]
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
        unaccumulated_outputs_binary = calculate_using_hardware(input_list, A2_list, kernel_list, kernel_size_list, dilation_list, step_mode_list, scale_list, pe_mode_list, kernel_num, a1_height_list, stride, depadding_num, depth, padding_one_dilated, ser=SER, using_mode=Using_MODE)
        unaccumulated_outputs = []
        for binary_item in unaccumulated_outputs_binary:
            unaccumulated_outputs.append(binary_to_float_tensor(binary_item))
        # 通道累加
        accumulated_outputs = torch.stack(unaccumulated_outputs, dim=0)
        accumulated_outputs = torch.sum(accumulated_outputs, dim=0, keepdim=False)
        outputs.append(accumulated_outputs)
    
    # 结果合并
    output = torch.stack(outputs, dim=0).unsqueeze(0)

    # 返回一致结果
    return output


if __name__ == '__main__':
    # 初始化硬件配置
    SER = make_hardware_config()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_project', default='conv', type=str)

    args = parser.parse_args()

    if args.test_project == 'conv':
        input = torch.round(4*torch.rand(1, 1, 20, 20))
        conv_layer = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1, padding=0, dilation=(1,1), bias=False)
        conv_weights = torch.round(torch.randn(1, 1, 3, 3))
        conv_layer.weight.data = conv_weights

        original_output = conv_layer(input)
        
        hardware_output = channel_parallel_compute(input, conv_layer, dilation=(1,1), T=4)
    
        print(f"input_shape: {input.shape}")
        print(f"kernel_size: {conv_weights.shape}")

        print(f"original_output_shape: {original_output.shape}")
        print(f"hardware_output_shape: {hardware_output.shape}")

        print(f"original_output:\n {original_output}")
        print(f"hardware_output:\n {hardware_output}")
        loss = nn.MSELoss()(original_output, hardware_output)
        print(f"loss: {loss}")

    elif args.test_project == 'gemm':
        input = torch.round(4*torch.rand(1, 10, 10)) # C H W
        FC_layer = nn.Linear(10, 20, bias=False)
        FC_weight = torch.round(4*torch.rand(20, 10))
        FC_layer.weight.data = FC_weight

        output = FC_layer(input)

        hardware_output = FC_hardware_calculate(input, FC_layer)
        print(output)
        print(hardware_output)
        print(f"MSELoss: {nn.MSELoss()(output, hardware_output)}")

