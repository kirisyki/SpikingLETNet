"""Factorized inference-only reproduction of the historical W4A4 quantizer.

This module is intentionally independent from the training implementation.  It
does not modify the historical quantizer and it exposes weight and operator-input
quantization as separate switches for the frozen-checkpoint causal probe.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_LAYER_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
EXPECTED_QUANTIZED_LAYER_COUNT = 71
MODES = ("fp_qif", "w4_qif", "a4_input_qif", "w4a4_input_qif")


def _flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
    if x.dim() in (3, 5):
        time_steps, batch_size = x.shape[:2]
        return x.flatten(0, 1), (time_steps, batch_size)
    return x, None


def _restore_time(
    y: torch.Tensor, time_shape: tuple[int, int] | None
) -> torch.Tensor:
    if time_shape is None:
        return y
    return y.unflatten(0, time_shape)


def _apply_layer(
    layer: nn.Module, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
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
    if isinstance(layer, nn.Linear):
        return F.linear(x, weight, bias=layer.bias)
    raise TypeError(f"unsupported probe layer: {type(layer)!r}")


@dataclass
class BypassAudit:
    calls: int = 0
    integer_bypass_calls: int = 0
    plus8_bypass_calls: int = 0
    plus8_values: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "integer_bypass_calls": self.integer_bypass_calls,
            "plus8_bypass_calls": self.plus8_bypass_calls,
            "plus8_values": self.plus8_values,
        }


class FactorizedQuantizedLayer(nn.Module):
    """Inference-only wrapper with independent historical W4 and A4 switches."""

    def __init__(
        self,
        original_layer: nn.Module,
        name: str,
        *,
        quantize_weight: bool,
        quantize_input: bool,
    ) -> None:
        super().__init__()
        if not isinstance(original_layer, SUPPORTED_LAYER_TYPES):
            raise TypeError(f"unsupported probe layer: {type(original_layer)!r}")
        self.layer = copy.deepcopy(original_layer)
        self.name = name
        self.quantize_weight_enabled = quantize_weight
        self.quantize_input_enabled = quantize_input
        self.quant_range = 8
        self.audit = BypassAudit()

    def quantized_weight(self) -> torch.Tensor:
        weight = self.layer.weight
        if not self.quantize_weight_enabled:
            return weight
        scale = max(1e-5, torch.max(torch.abs(weight.detach())).item() / 8.0)
        codes = torch.clamp(torch.round(weight.detach() / scale), -8, 7)
        return codes * scale

    def quantized_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_input_enabled:
            return x

        self.audit.calls += 1
        integer_bypass = bool(
            torch.all(x == torch.round(x))
            and torch.all(x >= -8)
            and torch.all(x <= 8)
        )
        if integer_bypass:
            self.audit.integer_bypass_calls += 1
            plus8_values = int((x == 8).sum().item())
            if plus8_values:
                self.audit.plus8_bypass_calls += 1
                self.audit.plus8_values += plus8_values
            return x

        scale = max(1e-5, torch.max(torch.abs(x.detach())).item() / 8.0)
        codes = torch.clamp(torch.round(x.detach() / scale), -8, 7)
        return codes * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat, time_shape = _flatten_time(x)
        output = _apply_layer(
            self.layer,
            self.quantized_input(flat),
            self.quantized_weight(),
        )
        return _restore_time(output, time_shape)


def mode_switches(mode: str) -> tuple[bool, bool]:
    if mode not in MODES:
        raise ValueError(f"unknown probe mode {mode!r}; expected one of {MODES}")
    return {
        "fp_qif": (False, False),
        "w4_qif": (True, False),
        "a4_input_qif": (False, True),
        "w4a4_input_qif": (True, True),
    }[mode]


def build_probe_model(
    source: nn.Module,
    mode: str,
    *,
    expected_layer_count: int = EXPECTED_QUANTIZED_LAYER_COUNT,
) -> nn.Module:
    """Deep-copy ``source`` and apply the requested factorized probe mode."""

    quantize_weight, quantize_input = mode_switches(mode)
    converted = copy.deepcopy(source)
    if mode == "fp_qif":
        return converted

    count = 0

    def replace(module: nn.Module, prefix: str = "") -> None:
        nonlocal count
        for child_name, child in list(module.named_children()):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, SUPPORTED_LAYER_TYPES):
                setattr(
                    module,
                    child_name,
                    FactorizedQuantizedLayer(
                        child,
                        qualified,
                        quantize_weight=quantize_weight,
                        quantize_input=quantize_input,
                    ),
                )
                count += 1
            else:
                replace(child, qualified)

    replace(converted)
    if count != expected_layer_count:
        raise RuntimeError(
            f"expected {expected_layer_count} probe layers, observed {count}"
        )
    return converted


def iter_probe_layers(
    model: nn.Module,
) -> Iterator[tuple[str, FactorizedQuantizedLayer]]:
    for name, module in model.named_modules():
        if isinstance(module, FactorizedQuantizedLayer):
            yield name, module


def bypass_audit(model: nn.Module) -> dict[str, object]:
    layers = {name: module.audit.as_dict() for name, module in iter_probe_layers(model)}
    totals = {
        field: sum(record[field] for record in layers.values())
        for field in (
            "calls",
            "integer_bypass_calls",
            "plus8_bypass_calls",
            "plus8_values",
        )
    }
    return {"totals": totals, "layers": layers}
