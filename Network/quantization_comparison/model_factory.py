"""Non-mutating model conversion and parameter grouping for W4A4 baselines."""

from __future__ import annotations

import copy
from typing import Iterator

import torch.nn as nn

from quantization.int4_selfbuild import QLayer

from .ewgs_qat import EWGSQuantizedLayer, iter_ewgs_layers
from .layer_adapter import SUPPORTED_LAYER_TYPES
from .lsq_qat import LSQQuantizedLayer, iter_lsq_layers
from .ste_qat import build_ste_model, iter_ste_layers


METHODS = ("ste", "lsq", "ewgs")
EXPECTED_LAYER_COUNT = 71
ComparisonLayer = QLayer | LSQQuantizedLayer | EWGSQuantizedLayer


def _replace_layers(
    model: nn.Module,
    wrapper_type: type[LSQQuantizedLayer] | type[EWGSQuantizedLayer],
) -> nn.Module:
    converted = copy.deepcopy(model)

    def replace(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module.named_children()):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, SUPPORTED_LAYER_TYPES):
                setattr(module, child_name, wrapper_type(child, qualified))
            else:
                replace(child, qualified)

    replace(converted)
    return converted


def build_quantized_model(
    method: str,
    model: nn.Module,
    *,
    expected_layer_count: int | None = EXPECTED_LAYER_COUNT,
) -> nn.Module:
    if method == "ste":
        converted = build_ste_model(model)
    elif method == "lsq":
        converted = _replace_layers(model, LSQQuantizedLayer)
    elif method == "ewgs":
        converted = _replace_layers(model, EWGSQuantizedLayer)
    else:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    count = len(quantized_layer_names(converted))
    if expected_layer_count is not None and count != expected_layer_count:
        raise RuntimeError(
            f"expected {expected_layer_count} quantized layers, observed {count}"
        )
    return converted


def iter_quantized_layers(
    model: nn.Module,
) -> Iterator[tuple[str, ComparisonLayer]]:
    for name, module in model.named_modules():
        if isinstance(module, (QLayer, LSQQuantizedLayer, EWGSQuantizedLayer)):
            yield name, module


def quantized_layer_names(model: nn.Module) -> list[str]:
    return [name for name, _ in iter_quantized_layers(model)]


def quantizer_parameters(method: str, model: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    if method == "lsq":
        for _, layer in iter_lsq_layers(model):
            parameters.extend(
                [
                    layer.weight_quantizer.step_size,
                    layer.activation_quantizer.step_size,
                ]
            )
    elif method == "ewgs":
        for _, layer in iter_ewgs_layers(model):
            parameters.extend(
                [
                    layer.lower_weight,
                    layer.upper_weight,
                    layer.lower_activation,
                    layer.upper_activation,
                    layer.output_scale,
                ]
            )
    elif method != "ste":
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return parameters


def model_parameters(method: str, model: nn.Module) -> list[nn.Parameter]:
    quantizer_ids = {id(parameter) for parameter in quantizer_parameters(method, model)}
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in quantizer_ids
    ]


def clamp_quantizer_parameters(model: nn.Module) -> None:
    for _, layer in iter_lsq_layers(model):
        layer.clamp_quantizer_parameters()
    for _, layer in iter_ewgs_layers(model):
        layer.clamp_quantizer_parameters()


def ste_layers(model: nn.Module):
    return iter_ste_layers(model)
