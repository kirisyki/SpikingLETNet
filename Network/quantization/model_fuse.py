import torch
import torch.nn as nn
import copy
from typing import Optional

def _fuse_conv_and_bn(conv: nn.Module, bn: nn.Module):
    """
    Fuse a Conv{1,2}d, ConvTranspose{1,2}d (conv) and a BatchNorm{1,2}d (bn) in-place.
    """
    if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        raise ValueError("bn must be BatchNorm1d or BatchNorm2d")

    if bn.running_mean is None or bn.running_var is None:
        raise RuntimeError("BatchNorm has no running stats; run calibration or training before fusing.")

    # === 1. 获取参数 ===
    w = conv.weight.clone().detach()
    device = w.device
    dtype = w.dtype

    if conv.bias is not None:
        b_conv = conv.bias.clone().detach()
    else:
        # ConvTranspose2d 输出通道是 conv.in_channels（不同于 Conv2d）
        if isinstance(conv, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
            out_channels = conv.in_channels
        else:
            out_channels = conv.out_channels
        b_conv = torch.zeros(out_channels, device=device, dtype=dtype)

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    # === 2. 计算融合参数 ===
    std = torch.sqrt(var + eps)

    # 判断卷积类型决定缩放维度
    if isinstance(conv, (nn.Conv2d, nn.ConvTranspose2d)):
        shape = [-1, 1, 1, 1]
    elif isinstance(conv, (nn.Conv1d, nn.ConvTranspose1d)):
        shape = [-1, 1, 1]
    else:
        shape = [-1] + [1] * (w.ndim - 1)

    # 对于 Conv2d： out_channels = conv.out_channels
    # 对于 ConvTranspose2d： out_channels = conv.in_channels
    # 但 BN 总是跟随输出特征通道 => 即卷积的 *输出通道维度*
    if isinstance(conv, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
        # ConvTranspose 输出通道在 weight 的第 1 维
        # weight shape: (in_channels, out_channels, kH, kW)
        # 我们需要对 out_channels 维度（即 dim=1）进行缩放
        scale = (gamma / std).view(*shape)
        w_fused = w * scale.transpose(0, 1)
        b_fused = beta + (b_conv - mean) * (gamma / std)
    else:
        # 普通 Conv
        w_fused = w * (gamma / std).view(*shape)
        b_fused = beta + (b_conv - mean) * (gamma / std)

    # === 3. 写回融合参数 ===
    conv.weight.data.copy_(w_fused.to(conv.weight.device, conv.weight.dtype))
    if conv.bias is None:
        conv.bias = nn.Parameter(b_fused.to(conv.weight.device, conv.weight.dtype))
    else:
        conv.bias.data.copy_(b_fused.to(conv.weight.device, conv.bias.dtype))

    # === 4. BN 后续由上层设置为 Identity ===


def _find_bn_attr(module):
    """
    Return attribute name for BN if module contains typical patterns.
    """
    if hasattr(module, "bn") and isinstance(getattr(module, "bn"), (nn.BatchNorm1d, nn.BatchNorm2d)):
        return ("bn", None)
    for wrapper_name in ("bn_prelu", "bn_relu", "bn_prelu_2", "bn_prelu_1"):
        if hasattr(module, wrapper_name):
            inner = getattr(module, wrapper_name)
            if hasattr(inner, "bn") and isinstance(getattr(inner, "bn"), (nn.BatchNorm1d, nn.BatchNorm2d)):
                return (wrapper_name, "bn")
            if isinstance(inner, (nn.BatchNorm1d, nn.BatchNorm2d)):
                return (wrapper_name, None)
    return (None, None)


def fuse_model(model: nn.Module, inplace: bool = True):
    """
    Recursively fuse Conv/ConvTranspose + BatchNorm modules in a model.
    """
    if not inplace:
        model = copy.deepcopy(model)

    was_training = model.training
    model.eval()

    for name, child in model.named_children():
        fuse_model(child, inplace=True)

        # pattern 1: child has 'conv' attribute
        if hasattr(child, "conv") and isinstance(
            getattr(child, "conv"),
            (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d),
        ):
            conv_attr_name = "conv"
            conv_obj = getattr(child, conv_attr_name)
            wrapper_name, bn_attr = _find_bn_attr(child)
            if wrapper_name is not None:
                wrapper = getattr(child, wrapper_name)
                if bn_attr is None:
                    bn_obj = wrapper if isinstance(wrapper, (nn.BatchNorm1d, nn.BatchNorm2d)) else None
                    bn_parent_attr = wrapper_name
                    bn_sub_attr = None
                else:
                    bn_obj = getattr(wrapper, bn_attr)
                    bn_parent_attr = wrapper_name
                    bn_sub_attr = bn_attr
                if bn_obj is not None:
                    try:
                        _fuse_conv_and_bn(conv_obj, bn_obj)
                    except Exception as e:
                        print(f"[WARN] failed to fuse {name}.{conv_attr_name} with {bn_parent_attr}.{bn_sub_attr}: {e}")
                        continue
                    # set BN to identity
                    if bn_sub_attr is None:
                        setattr(child, bn_parent_attr, nn.Identity())
                    else:
                        wrapper = getattr(child, bn_parent_attr)
                        setattr(wrapper, bn_sub_attr, nn.Identity())
                    continue

    if was_training:
        model.train()

    return model
