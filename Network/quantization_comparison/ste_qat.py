"""Task-loss-only STE-QAT baseline using the repository's frozen W4A4 layer."""

from __future__ import annotations

from typing import Iterator

import torch.nn as nn

from quantization.int4_selfbuild import QLayer, quantize_model


def build_ste_model(model: nn.Module) -> nn.Module:
    """Quantize all supported layers without mutating ``model``."""

    return quantize_model(
        model,
        k=4,
        inplace=False,
        quant=True,
        activation_quant=True,
        quant_start_layer=0,
        activation_quant_mode="per_tensor",
    )


def iter_ste_layers(model: nn.Module) -> Iterator[tuple[str, QLayer]]:
    for name, module in model.named_modules():
        if isinstance(module, QLayer):
            yield name, module
