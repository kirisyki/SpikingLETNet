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
from spikingjelly.activation_based import neuron, surrogate, layer, functional

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
    
    @staticmethod
    def spiking_function(x, alpha):
         return quant.apply(x)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()

class QIFNode(neuron.BaseNode):
    def __init__(self, v_threshold = 1.0, v_reset = None, surrogate_function = Quant(), detach_reset = False, step_mode='s', backend='torch', store_v_seq = False, bin=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.bin = bin

    def neuronal_fire(self):
        assert isinstance(self.surrogate_function, Quant)
        if self.bin:
            return expand_tensor_cumulative(self.surrogate_function(self.v))
        else:
            return self.surrogate_function(self.v)

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

# SpikingLETNet_shallow
if __name__ == '__main__':
    input = torch.tensor(np.array([[1,2,3],
                                  [4,-2,-1],
                                  [7,0,-7]]), dtype=torch.float32).view(1,1,1,3,3)  # T B C H W
    net1 = nn.Sequential(
        layer.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False),
        QIFNode(bin=True),
        # layer.BatchNorm2d(num_features=1),
        layer.Conv2d(1,1,kernel_size=3, padding=1, stride=1, bias=False)
    )

    net2 = copy.deepcopy(net1)
    net2[1] = QIFNode(bin=False)
    functional.set_step_mode(net1, 'm')
    functional.set_step_mode(net2, 'm')
    print(f"input:{input}")
    output1 = net1(input)
    output2 = net2(input)
    print(f"output.sum(0):{output1.sum(0)}")
    print(f"output2: {output2}")

    # test_conv = layer.Conv2d(1, 1, kernel_size=1, padding=1, stride=1, bias=False)
    # functional.set_step_mode(test_conv, 'm')

    # test_input1 = torch.tensor(np.array([1,1]), dtype=torch.float32).view(2, 1, 1, 1, 1)
    # test_input2 = torch.tensor(np.array([2]), dtype=torch.float32).view(1, 1, 1, 1, 1)

    # test_output1 = test_conv(test_input1)
    # test_output2 = test_conv(test_input2)
    # print(test_output1)
    # print(test_output2)
