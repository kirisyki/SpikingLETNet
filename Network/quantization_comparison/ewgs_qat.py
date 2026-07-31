"""Clean-room W4A4 Element-Wise Gradient Scaling (EWGS) layers."""

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


_EPS = 1e-6


class EWGSDiscretizer(torch.autograd.Function):
    """Uniform forward discretization with the EWGS backward rule."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        normalized: torch.Tensor,
        levels: int,
        backward_scale: torch.Tensor,
    ) -> torch.Tensor:
        if levels < 2:
            raise ValueError("levels must be at least two")
        quantized = torch.round(normalized * (levels - 1)) / (levels - 1)
        ctx.save_for_backward(normalized - quantized, backward_scale)
        return quantized

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        quantization_error, backward_scale = ctx.saved_tensors
        correction = (
            1.0
            + backward_scale
            * torch.sign(grad_output)
            * quantization_error
        )
        return grad_output * correction, None, None


class EWGSQuantizedLayer(nn.Module):
    """EWGS wrapper with learned clipping bounds and output scale."""

    def __init__(self, original_layer: nn.Module, name: str, levels: int = 16):
        super().__init__()
        validate_supported_layer(original_layer)
        if levels != 16:
            raise ValueError("the comparison protocol is frozen to 16 levels")
        self.name = name
        self.levels = levels
        self.layer = copy.deepcopy(original_layer)

        weight = self.layer.weight.detach()
        weight_std = weight.std(unbiased=False).clamp_min(_EPS)
        self.lower_weight = nn.Parameter((-3.0 * weight_std).reshape(()))
        self.upper_weight = nn.Parameter((3.0 * weight_std).reshape(()))
        self.lower_activation = nn.Parameter(torch.zeros(()))
        self.upper_activation = nn.Parameter(torch.ones(()))
        self.output_scale = nn.Parameter(torch.ones(()))

        self.register_buffer("activation_initialized", torch.tensor(False))
        self.register_buffer("weight_backward_scale", torch.zeros(()))
        self.register_buffer("activation_backward_scale", torch.zeros(()))
        self.capture_hessian_tensors = False
        self.last_quantized_weight: torch.Tensor | None = None
        self.last_quantized_activation: torch.Tensor | None = None

    @staticmethod
    def _safe_gap(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return (upper - lower).clamp_min(_EPS)

    def _normalize(
        self,
        tensor: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        return ((tensor - lower) / self._safe_gap(lower, upper)).clamp(0.0, 1.0)

    @torch.no_grad()
    def _initialize_activation(self, x: torch.Tensor) -> None:
        x_detached = x.detach()
        lower = x_detached.amin()
        half_normal_std = math.sqrt(1.0 - 2.0 / math.pi)
        upper = 3.0 * x_detached.std(unbiased=False) / half_normal_std
        upper = torch.maximum(upper, lower + x_detached.new_tensor(_EPS))
        self.lower_activation.copy_(lower.to(self.lower_activation))
        self.upper_activation.copy_(upper.to(self.upper_activation))
        self.activation_initialized.fill_(True)

    def quantized_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self._normalize(
            self.layer.weight, self.lower_weight, self.upper_weight
        )
        unit_codes = EWGSDiscretizer.apply(
            normalized, self.levels, self.weight_backward_scale
        )
        return 2.0 * unit_codes - 1.0, unit_codes

    def quantized_activation(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.activation_initialized):
            self._initialize_activation(x)
        normalized = self._normalize(
            x, self.lower_activation, self.upper_activation
        )
        return EWGSDiscretizer.apply(
            normalized, self.levels, self.activation_backward_scale
        )

    @torch.no_grad()
    def _initialize_output_scale(
        self,
        x: torch.Tensor,
        activation_q: torch.Tensor,
        weight_q: torch.Tensor,
    ) -> None:
        full_output = apply_layer(self.layer, x, self.layer.weight)
        quant_output = apply_layer(self.layer, activation_q, weight_q)
        numerator = full_output.abs().mean()
        denominator = quant_output.abs().mean().clamp_min(_EPS)
        self.output_scale.copy_((numerator / denominator).to(self.output_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, time_shape = flatten_time(x)
        first_activation_batch = not bool(self.activation_initialized)
        activation_q = self.quantized_activation(x)
        weight_q, _ = self.quantized_weight()
        if first_activation_batch:
            self._initialize_output_scale(x, activation_q, weight_q)

        if self.capture_hessian_tensors:
            self.last_quantized_weight = weight_q
            self.last_quantized_activation = activation_q
        else:
            self.last_quantized_weight = None
            self.last_quantized_activation = None

        output = apply_layer(self.layer, activation_q, weight_q)
        output = output * self.output_scale.abs()
        return restore_time(output, time_shape)

    @torch.no_grad()
    def clamp_quantizer_parameters(self) -> None:
        if self.upper_weight <= self.lower_weight + _EPS:
            self.upper_weight.copy_(self.lower_weight + _EPS)
        if self.upper_activation <= self.lower_activation + _EPS:
            self.upper_activation.copy_(self.lower_activation + _EPS)
        self.output_scale.copy_(self.output_scale.abs().clamp_min(_EPS))

    @torch.no_grad()
    def set_backward_scales(
        self, weight_scale: float | torch.Tensor, activation_scale: float | torch.Tensor
    ) -> None:
        weight = torch.as_tensor(
            weight_scale,
            device=self.weight_backward_scale.device,
            dtype=self.weight_backward_scale.dtype,
        ).clamp_min(0.0)
        activation = torch.as_tensor(
            activation_scale,
            device=self.activation_backward_scale.device,
            dtype=self.activation_backward_scale.dtype,
        ).clamp_min(0.0)
        self.weight_backward_scale.copy_(weight)
        self.activation_backward_scale.copy_(activation)


def iter_ewgs_layers(model: nn.Module) -> Iterator[tuple[str, EWGSQuantizedLayer]]:
    for name, module in model.named_modules():
        if isinstance(module, EWGSQuantizedLayer):
            yield name, module
