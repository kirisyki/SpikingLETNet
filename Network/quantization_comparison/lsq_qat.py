"""Clean-room per-tensor W4A4 Learned Step Size Quantization (LSQ)."""

from __future__ import annotations

import copy
import math
from typing import Iterator

import torch
import torch.nn as nn

from .layer_adapter import (
    apply_layer,
    flatten_time,
    restore_time,
    validate_supported_layer,
)


_EPS = 1e-8


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity in forward, multiply the backward gradient by ``scale``."""

    return x.detach() + (x - x.detach()) * scale


def round_pass(x: torch.Tensor) -> torch.Tensor:
    """Round in forward and use an identity straight-through gradient."""

    return x.round().detach() + (x - x.detach())


class LSQQuantizer(nn.Module):
    """Signed, per-tensor LSQ quantizer with a trainable scalar step size."""

    def __init__(
        self,
        bits: int = 4,
        *,
        initial_tensor: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if bits != 4:
            raise ValueError("the comparison protocol is frozen to 4 bits")
        self.bits = bits
        self.qn = -(2 ** (bits - 1))
        self.qp = 2 ** (bits - 1) - 1
        self.step_size = nn.Parameter(torch.ones(()))
        self.register_buffer(
            "initialized", torch.tensor(initial_tensor is not None, dtype=torch.bool)
        )
        if initial_tensor is not None:
            self.initialize(initial_tensor)

    @torch.no_grad()
    def initialize(self, tensor: torch.Tensor) -> None:
        initial = (
            2.0
            * tensor.detach().abs().mean()
            / math.sqrt(float(self.qp))
        ).clamp_min(_EPS)
        self.step_size.copy_(initial.to(self.step_size))
        self.initialized.fill_(True)

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not bool(self.initialized):
            self.initialize(tensor)
        scale_factor = 1.0 / math.sqrt(tensor.numel() * float(self.qp))
        step = grad_scale(self.step_size.clamp_min(_EPS), scale_factor)
        normalized = tensor / step
        codes = round_pass(normalized.clamp(self.qn, self.qp))
        return codes * step, codes, step

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        self.step_size.clamp_(min=_EPS)


class LSQQuantizedLayer(nn.Module):
    """Linear/Conv wrapper with independent LSQ weight and input quantizers."""

    def __init__(self, original_layer: nn.Module, name: str) -> None:
        super().__init__()
        validate_supported_layer(original_layer)
        self.name = name
        self.layer = copy.deepcopy(original_layer)
        self.weight_quantizer = LSQQuantizer(
            bits=4, initial_tensor=self.layer.weight.detach()
        )
        self.activation_quantizer = LSQQuantizer(bits=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, time_shape = flatten_time(x)
        weight_q, _, _ = self.weight_quantizer(self.layer.weight)
        x_q, _, _ = self.activation_quantizer(x)
        output = apply_layer(self.layer, x_q, weight_q)
        return restore_time(output, time_shape)

    @torch.no_grad()
    def clamp_quantizer_parameters(self) -> None:
        self.weight_quantizer.clamp_parameters()
        self.activation_quantizer.clamp_parameters()


def iter_lsq_layers(model: nn.Module) -> Iterator[tuple[str, LSQQuantizedLayer]]:
    for name, module in model.named_modules():
        if isinstance(module, LSQQuantizedLayer):
            yield name, module
