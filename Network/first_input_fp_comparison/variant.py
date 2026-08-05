"""Keep the first layer's W4 weight while bypassing only image-input A4."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from quantization.int4_selfbuild import QLayer


EXPECTED_FIRST_LAYER = "init_conv.0.conv"


def _qlayers(model: nn.Module) -> list[tuple[str, QLayer]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, QLayer)
    ]


def qlayer_input_policy(model: nn.Module) -> list[dict[str, Any]]:
    """Return the reader-visible weight/input quantization policy."""

    return [
        {
            "index": index,
            "name": name,
            "weight_quantized": bool(module.quant),
            "input_quantized": bool(module.quant and module.activation_quant),
            "bits": int(module.k),
        }
        for index, (name, module) in enumerate(_qlayers(model))
    ]


def exempt_first_qlayer_input(
    model: nn.Module,
    *,
    expected_first_layer: str = EXPECTED_FIRST_LAYER,
    expected_layer_count: int = 71,
) -> dict[str, Any]:
    """Disable only the first QLayer input quantizer, preserving W4 weights."""

    layers = _qlayers(model)
    if len(layers) != expected_layer_count:
        raise RuntimeError(
            f"expected {expected_layer_count} QLayers, observed {len(layers)}"
        )
    first_name, first = layers[0]
    if first_name != expected_first_layer:
        raise RuntimeError(
            f"expected first QLayer {expected_first_layer!r}, observed {first_name!r}"
        )
    if not first.quant:
        raise RuntimeError("first-layer weight quantization is unexpectedly disabled")
    if not first.activation_quant:
        raise RuntimeError("first-layer input quantization was already disabled")
    if any(not module.activation_quant for _, module in layers[1:]):
        disabled = [name for name, module in layers[1:] if not module.activation_quant]
        raise RuntimeError(f"later QLayer input quantizers already disabled: {disabled}")

    first.activation_quant = False
    policy = qlayer_input_policy(model)
    if not policy[0]["weight_quantized"] or policy[0]["input_quantized"]:
        raise RuntimeError("failed to preserve W4 weight while bypassing image A4")
    if any(not row["input_quantized"] for row in policy[1:]):
        raise RuntimeError("the exemption leaked beyond the first QLayer")
    return {
        "variant": "first_image_input_fp",
        "first_layer": first_name,
        "first_layer_weight_bits": int(first.k),
        "first_layer_input_bits": 32,
        "later_layer_input_bits": int(first.k),
        "quantized_weight_layer_count": sum(
            int(row["weight_quantized"]) for row in policy
        ),
        "quantized_input_layer_count": sum(
            int(row["input_quantized"]) for row in policy
        ),
    }
