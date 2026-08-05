"""Capture and compare paired Integer-LIF/QIF output codes."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from model.module.neuron import QIFNode


class QIFCapture:
    def __init__(self, model: nn.Module) -> None:
        self.declared_names: list[str] = []
        self.outputs: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        self.handles: list[Any] = []
        for name, module in model.named_modules():
            if not isinstance(module, QIFNode):
                continue
            self.declared_names.append(name)
            self.outputs[name] = []

            def hook(
                _module: nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                layer_name: str = name,
            ) -> None:
                if not torch.is_tensor(output):
                    raise TypeError(f"QIF output is not a tensor at {layer_name}")
                self.outputs[layer_name].append(output.detach())

            self.handles.append(module.register_forward_hook(hook))

    def clear(self) -> None:
        for values in self.outputs.values():
            values.clear()

    @property
    def active_names(self) -> list[str]:
        return [name for name, values in self.outputs.items() if values]

    @property
    def inactive_names(self) -> list[str]:
        return [name for name, values in self.outputs.items() if not values]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _batch_flatten(x: torch.Tensor, batch_size: int, layer_name: str) -> torch.Tensor:
    if x.dim() >= 2 and x.shape[0] == 1 and x.shape[1] == batch_size:
        x = x.squeeze(0)
    elif x.shape[0] != batch_size:
        raise ValueError(
            f"cannot identify batch axis at {layer_name}: "
            f"shape={tuple(x.shape)}, batch_size={batch_size}"
        )
    return x.reshape(batch_size, -1)


def _validate_codes(x: torch.Tensor, layer_name: str) -> None:
    if not bool(torch.all(x == torch.round(x))):
        raise ValueError(f"non-integer QIF output at {layer_name}")
    if x.numel() and (float(x.min()) < 0.0 or float(x.max()) > 8.0):
        raise ValueError(
            f"QIF output outside [0,8] at {layer_name}: "
            f"min={float(x.min())}, max={float(x.max())}"
        )


def compare_qif_captures(
    reference: QIFCapture,
    variant: QIFCapture,
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor | str | int]]:
    """Return per-layer, per-sample raw paired code counts on the GPU."""

    if reference.declared_names != variant.declared_names:
        raise ValueError("reference and variant QIF declarations differ")
    if reference.active_names != variant.active_names:
        raise ValueError("reference and variant active QIF sets differ")

    rows: list[dict[str, torch.Tensor | str | int]] = []
    for name in reference.active_names:
        ref_calls = reference.outputs[name]
        var_calls = variant.outputs[name]
        if len(ref_calls) != len(var_calls):
            raise ValueError(
                f"QIF invocation count differs at {name}: "
                f"reference={len(ref_calls)}, variant={len(var_calls)}"
            )

        count = torch.zeros(batch_size, dtype=torch.int64, device=ref_calls[0].device)
        disagreement = torch.zeros_like(count)
        abs_error = torch.zeros_like(count)
        signed_error = torch.zeros_like(count)
        reference_zero = torch.zeros_like(count)
        variant_zero = torch.zeros_like(count)
        reference_saturation = torch.zeros_like(count)
        variant_saturation = torch.zeros_like(count)

        for ref_output, var_output in zip(ref_calls, var_calls):
            if ref_output.shape != var_output.shape:
                raise ValueError(
                    f"QIF output shape differs at {name}: "
                    f"reference={tuple(ref_output.shape)}, "
                    f"variant={tuple(var_output.shape)}"
                )
            _validate_codes(ref_output, name)
            _validate_codes(var_output, name)
            ref = _batch_flatten(ref_output, batch_size, name).to(torch.int16)
            var = _batch_flatten(var_output, batch_size, name).to(torch.int16)
            delta = var - ref
            count += ref.shape[1]
            disagreement += (delta != 0).sum(1)
            abs_error += delta.abs().sum(1)
            signed_error += delta.sum(1)
            reference_zero += (ref == 0).sum(1)
            variant_zero += (var == 0).sum(1)
            reference_saturation += (ref == 8).sum(1)
            variant_saturation += (var == 8).sum(1)

        rows.append(
            {
                "layer": name,
                "invocations": len(ref_calls),
                "count": count,
                "disagreement_count": disagreement,
                "abs_error_sum": abs_error,
                "signed_error_sum": signed_error,
                "reference_zero_count": reference_zero,
                "variant_zero_count": variant_zero,
                "reference_saturation_count": reference_saturation,
                "variant_saturation_count": variant_saturation,
            }
        )
    return rows
