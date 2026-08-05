from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch.nn as nn


NETWORK_DIR = Path(__file__).resolve().parents[2] / "Network"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))

from first_input_fp_comparison.variant import (  # noqa: E402
    exempt_first_qlayer_input,
    qlayer_input_policy,
)
from quantization.int4_selfbuild import QLayer  # noqa: E402


def two_layer_model() -> nn.Module:
    return nn.Sequential(
        QLayer(nn.Conv2d(3, 4, 3), name="first"),
        QLayer(nn.Conv2d(4, 4, 3), name="second"),
    )


def test_exemption_preserves_first_weight_and_later_input_quantization() -> None:
    model = two_layer_model()
    audit = exempt_first_qlayer_input(
        model, expected_first_layer="0", expected_layer_count=2
    )
    policy = qlayer_input_policy(model)
    assert audit["quantized_weight_layer_count"] == 2
    assert audit["quantized_input_layer_count"] == 1
    assert policy[0]["weight_quantized"] is True
    assert policy[0]["input_quantized"] is False
    assert policy[1]["input_quantized"] is True


def test_exemption_rejects_unexpected_first_layer() -> None:
    with pytest.raises(RuntimeError, match="expected first QLayer"):
        exempt_first_qlayer_input(
            two_layer_model(), expected_first_layer="wrong", expected_layer_count=2
        )
