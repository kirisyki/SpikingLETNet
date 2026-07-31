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

from quantization_comparison.ewgs_qat import (  # noqa: E402
    EWGSDiscretizer,
    EWGSQuantizedLayer,
)
from quantization_comparison.lsq_qat import (  # noqa: E402
    LSQQuantizedLayer,
    LSQQuantizer,
)


@pytest.mark.parametrize(
    "layer,input_shape,output_shape",
    [
        (nn.Conv2d(3, 4, 3, padding=1), (2, 3, 8, 8), (2, 4, 8, 8)),
        (
            nn.ConvTranspose2d(4, 6, 3, groups=2),
            (2, 4, 8, 8),
            (2, 6, 10, 10),
        ),
        (nn.Linear(5, 3), (2, 5), (2, 3)),
    ],
)
def test_lsq_wrapper_has_w4a4_codes_and_gradients(
    layer, input_shape, output_shape
):
    wrapper = LSQQuantizedLayer(layer, "target")
    x = torch.randn(*input_shape, requires_grad=True)
    x_flat = x
    weight_q, weight_codes, _ = wrapper.weight_quantizer(wrapper.layer.weight)
    _, activation_codes, _ = wrapper.activation_quantizer(x_flat)
    del weight_q
    assert weight_codes.min() >= -8
    assert weight_codes.max() <= 7
    assert activation_codes.min() >= -8
    assert activation_codes.max() <= 7

    output = wrapper(x)
    assert output.shape == output_shape
    output.sum().backward()
    assert wrapper.layer.weight.grad is not None
    assert wrapper.weight_quantizer.step_size.grad is not None
    assert wrapper.activation_quantizer.step_size.grad is not None


def test_lsq_time_dimension_is_preserved():
    wrapper = LSQQuantizedLayer(nn.Conv2d(3, 4, 3, padding=1), "conv")
    output = wrapper(torch.randn(2, 3, 3, 8, 8))
    assert output.shape == (2, 3, 4, 8, 8)


def test_lsq_initialization_and_positive_clamp():
    source = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    quantizer = LSQQuantizer(initial_tensor=source)
    expected = 2.0 * source.abs().mean() / (7.0**0.5)
    assert quantizer.step_size.item() == pytest.approx(expected.item())
    with torch.no_grad():
        quantizer.step_size.fill_(-1.0)
    quantizer.clamp_parameters()
    assert quantizer.step_size.item() > 0


def test_ewgs_backward_matches_elementwise_rule():
    normalized = torch.tensor([0.12, 0.48, 0.91], requires_grad=True)
    delta = torch.tensor(0.7)
    upstream = torch.tensor([2.0, -3.0, 4.0])
    quantized = EWGSDiscretizer.apply(normalized, 16, delta)
    expected = upstream * (
        1.0 + delta * upstream.sign() * (normalized.detach() - quantized.detach())
    )
    (quantized * upstream).sum().backward()
    torch.testing.assert_close(normalized.grad, expected)
    assert torch.all((quantized.detach() * 15).round() == quantized.detach() * 15)


@pytest.mark.parametrize(
    "layer,input_shape,output_shape",
    [
        (nn.Conv2d(3, 4, 3, padding=1), (2, 3, 8, 8), (2, 4, 8, 8)),
        (nn.ConvTranspose2d(4, 2, 1), (2, 4, 8, 8), (2, 2, 8, 8)),
        (nn.Linear(5, 3), (2, 5), (2, 3)),
    ],
)
def test_ewgs_wrapper_initializes_and_backpropagates(
    layer, input_shape, output_shape
):
    wrapper = EWGSQuantizedLayer(layer, "target")
    x = torch.randn(*input_shape, requires_grad=True)
    output = wrapper(x)
    assert output.shape == output_shape
    assert bool(wrapper.activation_initialized)
    assert wrapper.upper_activation > wrapper.lower_activation
    assert wrapper.upper_weight > wrapper.lower_weight
    assert wrapper.output_scale > 0
    output.square().mean().backward()
    assert wrapper.layer.weight.grad is not None
    assert wrapper.lower_weight.grad is not None
    assert wrapper.upper_weight.grad is not None


def test_ewgs_time_dimension_is_preserved():
    wrapper = EWGSQuantizedLayer(nn.Conv2d(3, 4, 3, padding=1), "conv")
    output = wrapper(torch.randn(2, 3, 3, 8, 8))
    assert output.shape == (2, 3, 4, 8, 8)
