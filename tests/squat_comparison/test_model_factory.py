from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from builders.model_builder import build_model
from model.module.neuron import QIFNode
from squat_comparison.lif_neuron import DirectLIF
from squat_comparison.model_factory import build_squat_from_fp32, convert_qif_copy
from squat_comparison.weight_quantizer import W4_TYPES


CONFIG = Path(__file__).resolve().parents[2] / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"


def test_factory_isolated_replacement_and_71_weight_targets() -> None:
    source = build_model("SpikingLETNet_shallow_max", 6, config=str(CONFIG))
    source_keys = tuple(source.state_dict())
    source_values = {key: value.clone() for key, value in source.state_dict().items()}
    source_qif = sum(isinstance(m, QIFNode) for m in source.modules())

    fp32, fp_audit = convert_qif_copy(source)
    assert fp_audit.qif_replaced == source_qif > 0
    assert fp_audit.direct_lif_count == source_qif + 1
    assert sum(isinstance(m, QIFNode) for m in fp32.modules()) == 0
    assert isinstance(fp32.classifier[-1], DirectLIF)

    squat, q_audit = build_squat_from_fp32(fp32)
    assert q_audit.weight_layers_quantized == 71
    assert sum(isinstance(m, W4_TYPES) for m in squat.modules()) == 71
    assert q_audit.weight_layer_names[0] == "init_conv.0.conv"
    assert "classifier.0.conv" in q_audit.weight_layer_names
    assert tuple(fp32.state_dict()) == tuple(squat.state_dict())

    assert tuple(source.state_dict()) == source_keys
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, source_values[key])
    assert sum(isinstance(m, QIFNode) for m in source.modules()) == source_qif


def test_small_forward_returns_binary_temporal_output_and_count_range() -> None:
    source = build_model("SpikingLETNet_shallow_max", 6, config=str(CONFIG))
    model, _ = convert_qif_copy(source)
    model.eval()
    x = torch.randn(2, 1, 3, 32, 32)
    with torch.no_grad():
        spikes = model(x)
    assert spikes.shape == (2, 1, 6, 32, 32)
    assert bool(torch.all((spikes == 0) | (spikes == 1)))
    counts = spikes.sum(0)
    assert counts.min() >= 0 and counts.max() <= 2

