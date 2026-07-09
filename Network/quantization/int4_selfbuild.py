import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.activation_based import layer
import copy
import math
import sys
import os


print_info = False
freeze_weight = False

def manual_get_shift(number):
    shift = 0
    tmp_num = number
    while tmp_num < -128 or tmp_num > 127:
        tmp_num >>= 1
        shift += 1
    return shift

def check_frozen_scale_x(model):
    for name, child in model.named_children():
            if isinstance(child, QLayer):
                print(f"name: {name} frozen_scale_x: {child.scale_x}")
            else:
                plot_scale_x(child)

def plot_scale_x(model_t):
    for name, child in model_t.named_children():
            if isinstance(child, QLayer):
                print(name)
                scale_x_rec = child.scale_x_rec
                import matplotlib.pyplot as plt
                plt.hist(scale_x_rec, bins=20, density=True)
                plt.xlabel('scale_x')
                plt.ylabel('density')
                plt.title(f'scale_x distribution of {name}')
                plt.savefig(f"/home/wyl/projects/LETNet/scale_x_distributions/{name}.png", dpi=600)
            else:
                plot_scale_x(child)

def round_to_power_of_two(scale):
    """
    将scale值四舍五入到最近的2^k值
    例如: 
    - scale=1.5 取 2
    - scale=0.49 取 0.5
    - scale=3.2 取 4
    - scale=0.8 取 1
    """
    if scale <= 0:
        return scale
    
    # 计算log2(scale)
    log2_scale = math.log2(scale)
    
    # 四舍五入到最近的整数
    rounded_exponent = round(log2_scale)
    
    # 返回2^rounded_exponent
    return 2.0 ** rounded_exponent

class QLayer(nn.Module):
    """
    量化层类，整合输入和参数的量化逻辑
    """
    def __init__(self, original_layer, name, k=4, quant=True, activation_quant=True, activation_quant_mode='per_tensor'):
        super(QLayer, self).__init__()
        self.name = name
        self.k = k
        self.layer = copy.deepcopy(original_layer)
        self.quant_range = 2**(k-1)  # 有符号量化范围
        self.quant = quant
        self.activation_quant = activation_quant
        self.activation_quant_mode = self._normalize_activation_quant_mode(activation_quant_mode)
        self.scale_m_static = None
        self.hardware_computing = False
        self.T = 8
        

    @staticmethod
    def _normalize_activation_quant_mode(mode):
        mode = mode.replace('-', '_')
        valid_modes = {'per_tensor', 'per_image', 'per_channel'}
        if mode not in valid_modes:
            raise ValueError(f"activation_quant_mode must be one of {sorted(valid_modes)}, got {mode}")
        return mode

    def _get_activation_scale(self, x):
        if self.activation_quant_mode == 'per_tensor':
            scale_x = torch.max(torch.abs(x.detach())).item() / self.quant_range
            return max(1e-5, scale_x)

        if self.activation_quant_mode == 'per_image':
            if x.dim() == 4:
                reduce_dims = (1, 2, 3)
                keepdim = True
            elif x.dim() == 2:
                reduce_dims = 1
                keepdim = True
            else:
                raise ValueError(f"per_image activation quantization expects 2D or 4D input, got {x.dim()}D")
        elif self.activation_quant_mode == 'per_channel':
            if x.dim() == 4:
                reduce_dims = (0, 2, 3)
                keepdim = True
            elif x.dim() == 2:
                reduce_dims = 0
                keepdim = True
            else:
                raise ValueError(f"per_channel activation quantization expects 2D or 4D input, got {x.dim()}D")

        scale_x = torch.amax(torch.abs(x.detach()), dim=reduce_dims, keepdim=keepdim) / self.quant_range
        return torch.clamp(scale_x, min=1e-5)

    def quantize_weight(self):
        if self.quant:
            with torch.no_grad():
                self.scale_w = torch.max(torch.abs(self.layer.weight.detach())).item() / self.quant_range
                self.scale_w = max(1e-5, self.scale_w)
            
            weight_q = torch.clamp(torch.round(self.layer.weight.detach() / self.scale_w), min=-1*self.quant_range, max=self.quant_range-1)
            weight_q  = weight_q * self.scale_w
            weight_q = self.layer.weight + (weight_q - self.layer.weight).detach() # Straight-Through Estimator
            self.weight_q = weight_q
            return weight_q

        else:
            self.scale_w = 1
            return self.layer.weight


    def quantize_input(self, x):
        """量化输入"""
        if self.quant and self.activation_quant:
            self.input_already_quantized = (torch.all(x==torch.round(x)) 
                                            and torch.all(x>=-1*self.quant_range) 
                                            and torch.all(x<=self.quant_range))
            
            if self.input_already_quantized:
                if print_info:
                    print(f"layer_{self.name}'s input is integer based.")
                x_q = x
                scale_x = 1
            else:
                with torch.no_grad():
                    scale_x = self._get_activation_scale(x)
                x_q = torch.clip(torch.round(x.detach() / scale_x), min=-1*self.quant_range, max=self.quant_range-1)
                x_q = x_q * scale_x
                x_q = (x_q - x).detach() + x # Straight-Through Estimator
                if print_info:
                    activation_quant_loss = nn.L1Loss()(x_q, x)
                    print(f"{self.name} activation quantization loss: {activation_quant_loss.item()}")
        else:
            scale_x = 1
            x_q = x

        return x_q, scale_x

    def dequantize_output(self, y, scale_x):
        """反量化输出"""
        if self.quant and self.activation_quant:
            return y * scale_x
        else:
            return torch.clip(y, -100, 100)
        
    def forward(self, x):
        # 处理输入时间维度
        t_dimension = False
        if x.dim() == 5:  # [T, B, C, H, W]
            t_dimension = True
            T, B, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
        elif x.dim() == 3:  # [T, B, D]
            t_dimension = True
            T, B, D = x.shape
            x = x.view(B*T, D)
        # 量化权重
        if freeze_weight:
            weight_q = self.weight_q
        else:
            weight_q = self.quantize_weight()
        # 量化输入
        x_q, scale_x = self.quantize_input(x)
        if isinstance(self.layer, nn.Conv2d):
            y_q = F.conv2d(x_q, weight_q, bias=self.layer.bias, stride=self.layer.stride, padding=self.layer.padding, dilation=self.layer.dilation, groups=self.layer.groups)
        elif isinstance(self.layer, nn.Linear):
            y_q = F.linear(x_q, weight_q, bias=self.layer.bias)
        elif isinstance(self.layer, nn.ConvTranspose2d):
            y_q = F.conv_transpose2d(x_q, weight_q, bias=self.layer.bias, stride=self.layer.stride, padding=self.layer.padding, output_padding=self.layer.output_padding, groups=self.layer.groups, dilation=self.layer.dilation)
        else:
            raise NotImplementedError(f"Unsupported layer type: {type(self.layer)}")

        y = y_q

        # 恢复时间维度
        if t_dimension:
            if y.dim() == 4:  # [B*T, C_out, H_out, W_out]
                B_T, C_out, H_out, W_out = y.shape
                y = y.view(T, B, C_out, H_out, W_out)
            elif y.dim() == 2:  # [B*T, F_out]
                B_T, F_out = y.shape
                y = y.view(T, B, F_out)

        return y

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8, 128, bias=False),  # 假设输入图片 32x32
            nn.ReLU(),
            nn.Linear(128, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleSNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleSNN, self).__init__()
        self.features = nn.Sequential(
            layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m'),
            nn.ReLU(),
            layer.MaxPool2d(2, step_mode='m'),
            layer.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m'),
            nn.ReLU(),
            layer.MaxPool2d(2, step_mode='m')
        )
        self.classifier = nn.Sequential(
            layer.Flatten(step_mode='m'),
            layer.Linear(32*8*8, 128, bias=False, step_mode='m'),  # 假设输入图片 32x32
            nn.ReLU(),
            layer.Linear(128, num_classes, bias=False, step_mode='m')
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def quantize_model(model, k=4, inplace=False, quant=True, activation_quant=True, quant_start_layer=0, activation_quant_mode='per_tensor'):
    """
    将全精度网络转换为量化网络
    
    Args:
        model: 模型实例
        k: 量化位数 默认为4
        inplace: 是否原地修改模型 默认为False
        quant: 是否量化权重
        activation_quant: 是否量化激活值
        quant_start_layer: 从第几个可量化层开始量化，0-based 编号
        activation_quant_mode: 激活量化方式，支持 per_tensor / per_image / per_channel

    Returns:
        量化后的模型
    """
    if not inplace:
        model_t = copy.deepcopy(model)
    else:
        model_t = model

    if quant_start_layer < 0:
        raise ValueError(f"quant_start_layer must be >= 0, got {quant_start_layer}")

    quantizable_layers = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
    layer_idx = 0

    # 递归函数来量化模型中的所有可量化层
    def _quantize_module(module, prefix=""):
        nonlocal layer_idx
        for name, child in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, quantizable_layers):
                enable_weight_quant = quant and layer_idx >= quant_start_layer
                enable_activation_quant = activation_quant and layer_idx >= quant_start_layer
                # if print_info:
                #     print(
                #         f"layer_idx={layer_idx} name={layer_name} "
                #         f"weight_quant={enable_weight_quant} "
                #         f"activation_quant={enable_activation_quant} "
                #         f"activation_quant_mode={activation_quant_mode} "
                #         f"bias={child.bias is not None}"
                #     )
                setattr(
                    module,
                    name,
                    QLayer(
                        child,
                        name=layer_name,
                        k=k,
                        quant=enable_weight_quant,
                        activation_quant=enable_activation_quant,
                        activation_quant_mode=activation_quant_mode,
                    ),
                )
                layer_idx += 1
            else:
                _quantize_module(child, layer_name)
    
    _quantize_module(model_t)
    return model_t


if __name__ == '__main__':
    simple_cnn = SimpleSNN()
    simple_cnn_quantized = quantize_model(simple_cnn, k=4, inplace=False, quant=True, activation_quant=True, quant_start_layer=1, activation_quant_mode='per_tensor')
    random_input = torch.randn(1, 1, 3, 32, 32)
    output_q = simple_cnn_quantized(random_input)
    output_f = simple_cnn(random_input)
    loss = nn.L1Loss()(output_q, output_f)
    print("Quantized output: ", output_q)
    print("Full-precision output: ", output_f)
    print(f"Loss between quantized and full-precision outputs: {loss.item()}")
