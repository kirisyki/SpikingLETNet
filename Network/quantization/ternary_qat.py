"""Independent ternary quantization primitives for SpikingLETNet QAT.

This module deliberately does not import or modify ``int4_selfbuild``.  It
keeps trainable full-precision parameters inside wrapped layers, uses ternary
weights during each forward pass, and applies a straight-through estimator
(STE) for optimization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterator, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ActivationMode = Literal["a4", "ternary"]
_QUANTIZABLE_LAYERS = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
_EPS = 1e-5


@dataclass(frozen=True)
class TernaryModelStats:
    layer_count: int
    weight_count: int
    negative_count: int
    zero_count: int
    positive_count: int

    @property
    def zero_fraction(self) -> float:
        return self.zero_count / self.weight_count if self.weight_count else 0.0

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "layer_count": self.layer_count,
            "weight_count": self.weight_count,
            "negative_count": self.negative_count,
            "zero_count": self.zero_count,
            "positive_count": self.positive_count,
            "zero_fraction": self.zero_fraction,
            "theoretical_bits_per_weight": 1.584962500721156,
        }


def _ste(original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Use ``quantized`` in the forward pass and identity gradients backward."""

    return original + (quantized - original).detach()


class TernaryQLayer(nn.Module):
    """Wrap a Linear/Conv2d/ConvTranspose2d layer for ternary-weight QAT."""

    def __init__(
        self,
        original_layer: nn.Module,
        name: str,
        activation_mode: ActivationMode = "a4",
    ) -> None:
        super().__init__()
        if not isinstance(original_layer, _QUANTIZABLE_LAYERS):
            raise TypeError(f"Unsupported ternary layer type: {type(original_layer)!r}")
        if activation_mode not in ("a4", "ternary"):
            raise ValueError(
                f"activation_mode must be 'a4' or 'ternary', got {activation_mode!r}"
            )

        self.name = name
        self.layer = copy.deepcopy(original_layer)
        self.activation_mode: ActivationMode = activation_mode

    def _weight_absmean_scale(self) -> torch.Tensor:
        """Return detached per-logical-output-channel abs-mean scales."""

        weight = self.layer.weight.detach()
        if isinstance(self.layer, nn.Linear):
            scale = weight.abs().mean(dim=1, keepdim=True)
        elif isinstance(self.layer, nn.Conv2d):
            scale = weight.abs().mean(dim=(1, 2, 3), keepdim=True)
        elif isinstance(self.layer, nn.ConvTranspose2d):
            # ConvTranspose2d stores [in_channels, out_channels/groups, kH, kW].
            # Reshape by groups so each logical output channel gets one scale.
            groups = self.layer.groups
            in_per_group = self.layer.in_channels // groups
            out_per_group = self.layer.out_channels // groups
            grouped = weight.reshape(
                groups,
                in_per_group,
                out_per_group,
                *weight.shape[2:],
            )
            grouped_scale = grouped.abs().mean(dim=(1, 3, 4), keepdim=True)
            scale = grouped_scale.expand(
                groups,
                in_per_group,
                out_per_group,
                1,
                1,
            ).reshape(self.layer.in_channels, out_per_group, 1, 1)
        else:  # pragma: no cover - guarded by __init__
            raise TypeError(f"Unsupported ternary layer type: {type(self.layer)!r}")
        return scale.clamp_min(_EPS)

    def quantized_weight(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return STE weight, integer ternary codes, and detached scales."""

        weight = self.layer.weight
        scale = self._weight_absmean_scale()
        codes = torch.round(weight.detach() / scale).clamp_(-1, 1)
        dequantized = codes * scale
        return _ste(weight, dequantized), codes, scale

    @staticmethod
    def _is_integer_a4(x: torch.Tensor) -> bool:
        # This intentionally mirrors the existing A4 integer-bypass behavior,
        # including acceptance of +8.
        return bool(
            torch.all(x == torch.round(x))
            and torch.all(x >= -8)
            and torch.all(x <= 8)
        )

    @staticmethod
    def _is_strict_ternary(x: torch.Tensor) -> bool:
        return bool(torch.all((x == -1) | (x == 0) | (x == 1)))

    def _quantize_activation_a4(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_integer_a4(x):
            return x
        scale = x.detach().abs().amax().div(8.0).clamp_min(_EPS)
        codes = torch.round(x.detach() / scale).clamp_(-8, 7)
        return _ste(x, codes * scale)

    def _quantize_activation_ternary(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_strict_ternary(x):
            return x
        scale = x.detach().abs().mean().clamp_min(_EPS)
        codes = torch.round(x.detach() / scale).clamp_(-1, 1)
        return _ste(x, codes * scale)

    def quantized_activation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return STE activation, integer codes, and the applied scale."""

        if self.activation_mode == "a4":
            if self._is_integer_a4(x):
                return x, x.detach(), x.new_tensor(1.0)
            scale = x.detach().abs().amax().div(8.0).clamp_min(_EPS)
            codes = torch.round(x.detach() / scale).clamp_(-8, 7)
        else:
            if self._is_strict_ternary(x):
                return x, x.detach(), x.new_tensor(1.0)
            scale = x.detach().abs().mean().clamp_min(_EPS)
            codes = torch.round(x.detach() / scale).clamp_(-1, 1)
        return _ste(x, codes * scale), codes, scale

    def _flatten_time_dimension(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int] | None]:
        if x.dim() == 5:  # [T, B, C, H, W]
            time_steps, batch = x.shape[:2]
            return x.flatten(0, 1), (time_steps, batch)
        if x.dim() == 3:  # [T, B, D]
            time_steps, batch = x.shape[:2]
            return x.flatten(0, 1), (time_steps, batch)
        return x, None

    @staticmethod
    def _restore_time_dimension(
        y: torch.Tensor, time_shape: Tuple[int, int] | None
    ) -> torch.Tensor:
        if time_shape is None:
            return y
        time_steps, batch = time_shape
        return y.unflatten(0, (time_steps, batch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, time_shape = self._flatten_time_dimension(x)
        weight, _, _ = self.quantized_weight()
        x, _, _ = self.quantized_activation(x)

        if isinstance(self.layer, nn.Conv2d):
            y = F.conv2d(
                x,
                weight,
                bias=self.layer.bias,
                stride=self.layer.stride,
                padding=self.layer.padding,
                dilation=self.layer.dilation,
                groups=self.layer.groups,
            )
        elif isinstance(self.layer, nn.Linear):
            y = F.linear(x, weight, bias=self.layer.bias)
        elif isinstance(self.layer, nn.ConvTranspose2d):
            y = F.conv_transpose2d(
                x,
                weight,
                bias=self.layer.bias,
                stride=self.layer.stride,
                padding=self.layer.padding,
                output_padding=self.layer.output_padding,
                groups=self.layer.groups,
                dilation=self.layer.dilation,
            )
        else:  # pragma: no cover - guarded by __init__
            raise TypeError(f"Unsupported ternary layer type: {type(self.layer)!r}")
        return self._restore_time_dimension(y, time_shape)


def quantize_ternary_model(
    model: nn.Module,
    activation_mode: ActivationMode = "a4",
    *,
    inplace: bool = False,
) -> nn.Module:
    """Wrap every Linear/Conv2d/ConvTranspose2d without mutating by default."""

    model_q = model if inplace else copy.deepcopy(model)

    def _replace(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, TernaryQLayer):
                continue
            if isinstance(child, _QUANTIZABLE_LAYERS):
                setattr(
                    module,
                    child_name,
                    TernaryQLayer(
                        child,
                        name=qualified_name,
                        activation_mode=activation_mode,
                    ),
                )
            else:
                _replace(child, qualified_name)

    _replace(model_q)
    return model_q


def iter_ternary_layers(model: nn.Module) -> Iterator[Tuple[str, TernaryQLayer]]:
    for name, module in model.named_modules():
        if isinstance(module, TernaryQLayer):
            yield name, module


def export_master_state_dict(model_q: nn.Module) -> Dict[str, torch.Tensor]:
    """Export FP32 master parameters using the unwrapped model's key layout."""

    layer_prefixes = []
    exported: Dict[str, torch.Tensor] = {}

    for name, module in iter_ternary_layers(model_q):
        layer_prefixes.append(name)
        for key, value in module.layer.state_dict().items():
            exported[f"{name}.{key}"] = value.detach().cpu()

    for key, value in model_q.state_dict().items():
        if any(key.startswith(f"{prefix}.layer.") for prefix in layer_prefixes):
            continue
        exported[key] = value.detach().cpu()
    return exported


@torch.no_grad()
def ternary_model_stats(model_q: nn.Module) -> TernaryModelStats:
    negative = zero = positive = total = layer_count = 0
    for _, module in iter_ternary_layers(model_q):
        _, codes, _ = module.quantized_weight()
        layer_count += 1
        total += codes.numel()
        negative += int((codes < 0).sum().item())
        zero += int((codes == 0).sum().item())
        positive += int((codes > 0).sum().item())
    return TernaryModelStats(
        layer_count=layer_count,
        weight_count=total,
        negative_count=negative,
        zero_count=zero,
        positive_count=positive,
    )
