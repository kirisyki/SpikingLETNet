"""Shared, architecture-neutral helpers for comparison quantized layers."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_LAYER_TYPES = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
TimeShape = Tuple[int, int] | None


def flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, TimeShape]:
    """Flatten the leading ``[T, B]`` axes used by multi-step SNN layers."""

    if x.dim() in (3, 5):
        time_steps, batch_size = x.shape[:2]
        return x.flatten(0, 1), (time_steps, batch_size)
    return x, None


def restore_time(y: torch.Tensor, time_shape: TimeShape) -> torch.Tensor:
    if time_shape is None:
        return y
    return y.unflatten(0, time_shape)


def apply_layer(
    layer: nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply a supported layer with an explicitly supplied weight tensor."""

    if isinstance(layer, nn.Conv2d):
        return F.conv2d(
            x,
            weight,
            bias=layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
    if isinstance(layer, nn.Linear):
        return F.linear(x, weight, bias=layer.bias)
    if isinstance(layer, nn.ConvTranspose2d):
        return F.conv_transpose2d(
            x,
            weight,
            bias=layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=layer.output_padding,
            groups=layer.groups,
            dilation=layer.dilation,
        )
    raise TypeError(f"unsupported comparison layer: {type(layer)!r}")


def validate_supported_layer(layer: nn.Module) -> None:
    if not isinstance(layer, SUPPORTED_LAYER_TYPES):
        raise TypeError(f"unsupported comparison layer: {type(layer)!r}")


def temporal_average(output: torch.Tensor) -> torch.Tensor:
    """Average the time axis of segmentation/classification model output."""

    if output.dim() in (3, 5):
        return output.mean(0)
    return output


def repeat_time(images: torch.Tensor, time_steps: int) -> torch.Tensor:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    return images.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)
