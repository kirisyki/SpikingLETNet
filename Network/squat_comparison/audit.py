"""Accuracy-independent spike, state, and theoretical storage audits."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .lif_neuron import DirectLIF
from .state_quantizer import ThresholdCenteredStateQuantizer
from .weight_quantizer import W4_TYPES, quantize_weight_ste


class OutputAudit:
    def __init__(self, classes: int) -> None:
        self.classes = classes
        self.pixels = 0
        self.all_zero_pixels = 0
        self.tied_pixels = 0
        self.maximum_count = 0.0
        self.class_count_sum = torch.zeros(classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, counts: torch.Tensor) -> None:
        detached = counts.detach()
        if detached.ndim != 4 or detached.shape[1] != self.classes:
            raise ValueError(f"unexpected count output shape: {tuple(detached.shape)}")
        pixels = detached.shape[0] * detached.shape[2] * detached.shape[3]
        maxima = detached.max(dim=1, keepdim=True).values
        self.pixels += int(pixels)
        self.all_zero_pixels += int((detached.sum(dim=1) == 0).sum().item())
        self.tied_pixels += int(((detached == maxima).sum(dim=1) > 1).sum().item())
        self.maximum_count = max(self.maximum_count, float(detached.max().item()))
        self.class_count_sum += detached.sum(dim=(0, 2, 3), dtype=torch.float64).cpu()

    def result(self) -> dict[str, Any]:
        return {
            "pixels": self.pixels,
            "all_zero_pixel_fraction": (
                self.all_zero_pixels / self.pixels if self.pixels else float("nan")
            ),
            "maximum_tie_fraction": (
                self.tied_pixels / self.pixels if self.pixels else float("nan")
            ),
            "maximum_spike_count": self.maximum_count,
            "mean_count_per_class": [
                float(x / self.pixels) if self.pixels else float("nan")
                for x in self.class_count_sum.tolist()
            ],
        }


class NeuronAuditCollector:
    """Collect full-sequence spikes and every quantized state via hooks."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: list[Any] = []
        self.spikes: dict[str, dict[str, Any]] = {}
        self.states: dict[str, dict[str, Any]] = {}
        module_names = {id(module): name for name, module in model.named_modules()}
        for name, module in model.named_modules():
            if isinstance(module, DirectLIF):
                self.handles.append(module.register_forward_hook(self._spike_hook(name)))
            elif isinstance(module, ThresholdCenteredStateQuantizer):
                parent_name = name.rsplit(".state_quantizer", 1)[0]
                self.handles.append(module.register_forward_hook(self._state_hook(parent_name)))

    def _spike_hook(self, name: str):
        @torch.no_grad()
        def hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
            del module, inputs
            entry = self.spikes.setdefault(name, {"sum": None, "elements": 0})
            value = output.detach().sum(dtype=torch.float64)
            entry["sum"] = value if entry["sum"] is None else entry["sum"] + value
            entry["elements"] += output.numel()

        return hook

    def _state_hook(self, name: str):
        @torch.no_grad()
        def hook(
            module: ThresholdCenteredStateQuantizer,
            inputs: Any,
            output: torch.Tensor,
        ) -> None:
            del inputs
            levels = module.levels.to(output)
            midpoints = (levels[:-1] + levels[1:]) * 0.5
            indices = torch.bucketize(output.detach().contiguous(), midpoints, right=False)
            counts = torch.bincount(indices.flatten(), minlength=levels.numel())
            entry = self.states.setdefault(
                name,
                {"counts": None, "levels": module.levels.detach().cpu()},
            )
            entry["counts"] = counts if entry["counts"] is None else entry["counts"] + counts

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for name, spike_entry in self.spikes.items():
            elements = int(spike_entry["elements"])
            spike_sum = float(spike_entry["sum"].item()) if elements else 0.0
            layer: dict[str, Any] = {
                "elements": elements,
                "spikes": spike_sum,
                "firing_rate": spike_sum / elements if elements else float("nan"),
            }
            state_entry = self.states.get(name)
            if state_entry is not None:
                counts = [int(x) for x in state_entry["counts"].cpu().tolist()]
                levels = [float(x) for x in state_entry["levels"].tolist()]
                total = sum(counts)
                near = sum(
                    count
                    for level, count in zip(levels, counts)
                    if abs(level - 1.0) <= 0.1
                )
                layer["state"] = {
                    "levels": levels,
                    "counts": counts,
                    "total": total,
                    "endpoint_fraction": (
                        (counts[0] + counts[-1]) / total if total else float("nan")
                    ),
                    "threshold_neighborhood": "abs(level - 1.0) <= 0.1",
                    "threshold_neighborhood_fraction": near / total if total else float("nan"),
                }
            report[name] = layer
        return report


@torch.no_grad()
def weight_and_storage_audit(model: nn.Module) -> dict[str, Any]:
    quantized_ids: set[int] = set()
    bias_ids: set[int] = set()
    layers: dict[str, Any] = {}
    quantized_elements = saturated = zero_codes = 0
    for name, module in model.named_modules():
        if isinstance(module, W4_TYPES):
            _, scale, code = quantize_weight_ste(module.weight)
            quantized_ids.add(id(module.weight))
            quantized_elements += code.numel()
            saturated += int((code.abs() == 7).sum().item())
            zero_codes += int((code == 0).sum().item())
            if module.bias is not None:
                bias_ids.add(id(module.bias))
            layers[name] = {
                "elements": code.numel(),
                "scale": float(scale.item()),
                "minimum_code": int(code.min().item()),
                "maximum_code": int(code.max().item()),
                "saturation_fraction": float((code.abs() == 7).float().mean().item()),
                "zero_fraction": float((code == 0).float().mean().item()),
                "illegal_code_count": int(((code < -7) | (code > 7)).sum().item()),
            }
    nonquantized_elements = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in quantized_ids
    )
    fp32_bias_elements = sum(
        parameter.numel() for parameter in model.parameters() if id(parameter) in bias_ids
    )
    return {
        "quantized_layer_count": len(layers),
        "quantized_weight_elements": quantized_elements,
        "aggregate_saturation_fraction": saturated / max(quantized_elements, 1),
        "aggregate_zero_code_fraction": zero_codes / max(quantized_elements, 1),
        "theoretical_quantized_weight_bits": quantized_elements * 4,
        "nonquantized_parameter_elements": nonquantized_elements,
        "theoretical_nonquantized_parameter_bits_assuming_fp32": nonquantized_elements * 32,
        "fp32_bias_elements_inside_quantized_layers": fp32_bias_elements,
        "scope_note": "W4 storage is theoretical; PyTorch fake-quant tensors remain FP32",
        "layers": layers,
    }

