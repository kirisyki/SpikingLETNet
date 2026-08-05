"""Independent signed symmetric per-tensor W4 fake-quantized layers."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer


QMIN = -7
QMAX = 7


def quantize_weight_ste(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dequantized W4 weight, detached scale, and integer-valued codes."""

    scale = (weight.detach().abs().amax() / float(QMAX)).clamp_min(
        torch.finfo(weight.dtype).eps
    )
    code = torch.round(weight / scale).clamp(QMIN, QMAX)
    dequantized = code * scale
    quantized_ste = weight + (dequantized - weight).detach()
    return quantized_ste, scale, code.detach()


class W4Conv2d(layer.Conv2d):
    @classmethod
    def from_float(cls, source: nn.Conv2d) -> "W4Conv2d":
        converted = cls(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            source.padding,
            source.dilation,
            source.groups,
            source.bias is not None,
            source.padding_mode,
            getattr(source, "step_mode", "s"),
        )
        converted.weight = source.weight
        converted.bias = source.bias
        return converted

    def _single(self, x: torch.Tensor) -> torch.Tensor:
        weight, _, _ = quantize_weight_ste(self.weight)
        return self._conv_forward(x, weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self._single(x)
        if self.step_mode == "m":
            if x.ndim != 5:
                raise ValueError(f"expected [T,N,C,H,W], got {tuple(x.shape)}")
            return functional.seq_to_ann_forward(x, self._single)
        raise ValueError(self.step_mode)


class W4Linear(nn.Linear):
    @classmethod
    def from_float(cls, source: nn.Linear) -> "W4Linear":
        converted = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        converted.weight = source.weight
        converted.bias = source.bias
        return converted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, _, _ = quantize_weight_ste(self.weight)
        return F.linear(x, weight, self.bias)


class W4ConvTranspose2d(layer.ConvTranspose2d):
    @classmethod
    def from_float(cls, source: nn.ConvTranspose2d) -> "W4ConvTranspose2d":
        converted = cls(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            source.padding,
            source.output_padding,
            source.groups,
            source.bias is not None,
            source.dilation,
            source.padding_mode,
            getattr(source, "step_mode", "s"),
        )
        converted.weight = source.weight
        converted.bias = source.bias
        return converted

    def _single(self, x: torch.Tensor) -> torch.Tensor:
        weight, _, _ = quantize_weight_ste(self.weight)
        return F.conv_transpose2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self._single(x)
        if self.step_mode == "m":
            if x.ndim != 5:
                raise ValueError(f"expected [T,N,C,H,W], got {tuple(x.shape)}")
            return functional.seq_to_ann_forward(x, self._single)
        raise ValueError(self.step_mode)


W4_TYPES = (W4Conv2d, W4Linear, W4ConvTranspose2d)


def iter_w4_layers(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, child in module.named_modules():
        if isinstance(child, W4_TYPES):
            yield name, child


def replace_weight_layers(module: nn.Module) -> list[str]:
    """Recursively replace Conv2d/Linear/ConvTranspose2d and return names."""

    replaced: list[str] = []

    def recurse(parent: nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, W4_TYPES):
                continue
            if isinstance(child, nn.ConvTranspose2d):
                replacement: nn.Module = W4ConvTranspose2d.from_float(child)
            elif isinstance(child, nn.Conv2d):
                replacement = W4Conv2d.from_float(child)
            elif isinstance(child, nn.Linear):
                replacement = W4Linear.from_float(child)
            else:
                recurse(child, full_name)
                continue
            setattr(parent, child_name, replacement)
            replaced.append(full_name)

    recurse(module, "")
    return replaced

