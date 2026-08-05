"""Create independent FP32-LIF and W4M4S1 model copies."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from builders.model_builder import build_model
from model.module.neuron import QIFNode
from utils.utils import init_weight

from .lif_neuron import DirectLIF
from .weight_quantizer import W4_TYPES, replace_weight_layers


@dataclass(frozen=True)
class FactoryAudit:
    qif_replaced: int
    direct_lif_count: int
    output_lif_added: int
    weight_layers_quantized: int
    weight_layer_names: tuple[str, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _replace_qif(parent: nn.Module, *, quantize_state: bool) -> int:
    count = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, QIFNode):
            setattr(
                parent,
                name,
                DirectLIF(
                    beta=0.5,
                    threshold=1.0,
                    alpha=2.0,
                    quantize_state=quantize_state,
                ),
            )
            count += 1
        else:
            count += _replace_qif(child, quantize_state=quantize_state)
    return count


def _replace_existing_lif_state_mode(parent: nn.Module, *, quantize_state: bool) -> int:
    count = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, DirectLIF):
            replacement = DirectLIF(
                beta=child.beta,
                threshold=child.threshold,
                alpha=child.alpha,
                quantize_state=quantize_state,
                step_mode=child.step_mode,
            )
            setattr(parent, name, replacement)
            count += 1
        else:
            count += _replace_existing_lif_state_mode(
                child, quantize_state=quantize_state
            )
    return count


def _audit_model(
    model: nn.Module,
    *,
    qif_replaced: int,
    output_lif_added: int,
    weight_names: list[str],
) -> FactoryAudit:
    remaining_qif = sum(isinstance(m, QIFNode) for m in model.modules())
    if remaining_qif:
        raise RuntimeError(f"factory left {remaining_qif} QIF nodes in the model")
    lif_count = sum(isinstance(m, DirectLIF) for m in model.modules())
    w4_count = sum(isinstance(m, W4_TYPES) for m in model.modules())
    if w4_count != len(weight_names):
        raise RuntimeError("W4 layer name/count audit mismatch")
    return FactoryAudit(
        qif_replaced=qif_replaced,
        direct_lif_count=lif_count,
        output_lif_added=output_lif_added,
        weight_layers_quantized=w4_count,
        weight_layer_names=tuple(weight_names),
    )


def convert_qif_copy(
    source: nn.Module,
    *,
    quantize_state: bool = False,
    quantize_weights: bool = False,
    expected_weight_layers: int | None = 71,
) -> tuple[nn.Module, FactoryAudit]:
    """Deep-copy ``source`` and convert only the copy."""

    model = copy.deepcopy(source)
    functional.reset_net(model)
    qif_count = _replace_qif(model, quantize_state=quantize_state)
    if not hasattr(model, "classifier") or not isinstance(model.classifier, nn.Sequential):
        raise TypeError("expected model.classifier to be nn.Sequential")
    model.classifier.add_module(
        "output_lif",
        DirectLIF(
            beta=0.5,
            threshold=1.0,
            alpha=2.0,
            quantize_state=quantize_state,
        ),
    )
    weight_names = replace_weight_layers(model) if quantize_weights else []
    if quantize_weights and expected_weight_layers is not None:
        if len(weight_names) != expected_weight_layers:
            raise RuntimeError(
                f"expected {expected_weight_layers} W4 targets, found {len(weight_names)}"
            )
    functional.set_step_mode(model, step_mode="m")
    audit = _audit_model(
        model,
        qif_replaced=qif_count,
        output_lif_added=1,
        weight_names=weight_names,
    )
    return model, audit


def build_fp32_lif_model(
    *, config: Path | str, classes: int = 6, seed: int = 1234
) -> tuple[nn.Module, FactoryAudit]:
    """Build a random-init native LIF model without any historical checkpoint."""

    seed_everything(seed)
    qif_source = build_model(
        "SpikingLETNet_shallow_max", classes, config=str(config)
    )
    init_weight(
        qif_source,
        nn.init.kaiming_normal_,
        nn.BatchNorm2d,
        1e-3,
        0.1,
        mode="fan_in",
    )
    return convert_qif_copy(
        qif_source,
        quantize_state=False,
        quantize_weights=False,
        expected_weight_layers=None,
    )


def build_squat_from_fp32(
    fp32_model: nn.Module,
    *, expected_weight_layers: int = 71,
) -> tuple[nn.Module, FactoryAudit]:
    """Clone the best FP32-LIF model and enable W4 weights plus M4 states."""

    source_keys = tuple(fp32_model.state_dict().keys())
    model = copy.deepcopy(fp32_model)
    functional.reset_net(model)
    lif_count = _replace_existing_lif_state_mode(model, quantize_state=True)
    if lif_count == 0:
        raise RuntimeError("FP32 source contains no DirectLIF neurons")
    weight_names = replace_weight_layers(model)
    if len(weight_names) != expected_weight_layers:
        raise RuntimeError(
            f"expected {expected_weight_layers} W4 targets, found {len(weight_names)}"
        )
    if tuple(model.state_dict().keys()) != source_keys:
        raise RuntimeError("FP32 and SQUAT state_dict keys differ")
    functional.set_step_mode(model, step_mode="m")
    return model, _audit_model(
        model,
        qif_replaced=0,
        output_lif_added=1,
        weight_names=weight_names,
    )

