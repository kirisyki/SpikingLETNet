import torch
import torch.nn as nn
import torch.nn.functional as F

def manual_deconv2d(input_tensor, layer: nn.ConvTranspose2d):
    # 1. 获取参数
    weight = layer.weight.data
    stride = layer.stride[0]
    padding = layer.padding[0]
    output_padding = layer.output_padding[0]
    bias = layer.bias

    batch_size, in_channels, iH, iW = input_tensor.shape
    out_channels, in_channels_per_group, kH, kW = weight.shape
    
    # 2. 对输入进行插空补零 (Stride Handling)
    if stride > 1:
        new_h, new_w = iH + (iH - 1) * (stride - 1), iW + (iW - 1) * (stride - 1)
        spaced_input = torch.zeros((batch_size, in_channels, new_h, new_w), device=input_tensor.device)
        spaced_input[:, :, ::stride, ::stride] = input_tensor
    else:
        spaced_input = input_tensor

    # 3. 【核心修正】计算实际的 Padding 边界
    # PyTorch 的逻辑实际上是：先在四周补 (kH - 1 - padding) 的零
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
    my_conv_layer = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=out_channels, 
        kernel_size=(kH, kW), 
        stride=1, 
        padding=0, 
        bias=False
    )

    my_conv_layer.weight.data = flipped_weight
    output = my_conv_layer(padded_input)

    # 6. 加上偏置
    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output

if __name__ == "__main__":
    conv_layer = nn.Conv2d(4, 4, kernel_size=(3,3), stride=1, groups=4, bias=False)
    print(conv_layer.weight.data.shape)
    # dconv_layer = nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), bias=False)
    # random_input = torch.clip(torch.round(4*torch.rand(1, 1, 2, 2)), min=-8, max=7)
    # random_weight = torch.clip(torch.round(torch.randn(1, 1, 3, 3)), min=-8, max=7)
    # dconv_layer.weight.data = random_weight

    # original_output = dconv_layer(random_input)
    # hardware_output = manual_deconv2d(random_input, dconv_layer)
    # print("original_output shape: ", original_output)
    # print("hardware_output shape: ", hardware_output)

    # print("loss: ", nn.MSELoss()(original_output, hardware_output))