from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = REPO_ROOT / "Network"
if str(NETWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(NETWORK_ROOT))

from quantization.int4_selfbuild import QLayer  # noqa: E402
from quantization_comparison.ewgs_qat import EWGSQuantizedLayer  # noqa: E402
from quantization_comparison.lsq_qat import LSQQuantizedLayer  # noqa: E402
from quantization_comparison.model_factory import (  # noqa: E402
    build_quantized_model,
    model_parameters,
    quantized_layer_names,
    quantizer_parameters,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 2, 1),
        )
        self.head = nn.Linear(2, 3)

    def forward(self, x):
        x = self.features(x).mean((2, 3))
        return self.head(x)


@pytest.mark.parametrize(
    "method,wrapper",
    [
        ("ste", QLayer),
        ("lsq", LSQQuantizedLayer),
        ("ewgs", EWGSQuantizedLayer),
    ],
)
def test_conversion_is_complete_non_mutating_and_trainable(method, wrapper):
    source = TinyModel()
    source_state = {key: value.clone() for key, value in source.state_dict().items()}
    converted = build_quantized_model(method, source, expected_layer_count=3)

    assert not any(isinstance(module, wrapper) for module in source.modules())
    assert len(quantized_layer_names(converted)) == 3
    assert sum(isinstance(module, wrapper) for module in converted.modules()) == 3
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, source_state[key])

    output = converted(torch.randn(2, 3, 8, 8))
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model_parameters(method, converted))


def test_parameter_groups_are_disjoint_and_complete():
    for method in ("ste", "lsq", "ewgs"):
        model = build_quantized_model(method, TinyModel(), expected_layer_count=3)
        model_group = model_parameters(method, model)
        quantizer_group = quantizer_parameters(method, model)
        assert {id(parameter) for parameter in model_group}.isdisjoint(
            {id(parameter) for parameter in quantizer_group}
        )
        assert {id(parameter) for parameter in model_group + quantizer_group} == {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }


def test_expected_layer_count_is_enforced():
    with pytest.raises(RuntimeError, match="expected 71"):
        build_quantized_model("lsq", TinyModel())
