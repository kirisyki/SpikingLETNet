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

from quantization.ternary_qat import (  # noqa: E402
    TernaryQLayer,
    export_master_state_dict,
    iter_ternary_layers,
    quantize_ternary_model,
    ternary_model_stats,
)


@pytest.mark.parametrize(
    "layer,input_shape",
    [
        (nn.Conv2d(3, 4, 3, padding=1, bias=True), (2, 3, 8, 8)),
        (nn.ConvTranspose2d(4, 6, 3, groups=2, bias=True), (2, 4, 8, 8)),
        (nn.Linear(5, 3, bias=True), (2, 5)),
    ],
)
def test_weight_codes_are_strictly_ternary_and_gradients_flow(layer, input_shape):
    wrapper = TernaryQLayer(layer, "target", activation_mode="a4")
    _, codes, scale = wrapper.quantized_weight()
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.all(scale > 0)

    x = torch.randn(*input_shape, requires_grad=True)
    wrapper(x).sum().backward()
    assert x.grad is not None
    assert wrapper.layer.weight.grad is not None
    assert torch.isfinite(wrapper.layer.weight.grad).all()


def test_conv2d_absmean_scale_is_per_output_channel():
    layer = nn.Conv2d(1, 2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[[2.0]]], [[[8.0]]]]))
    wrapper = TernaryQLayer(layer, "conv", activation_mode="a4")
    _, codes, scale = wrapper.quantized_weight()
    torch.testing.assert_close(scale.flatten(), torch.tensor([2.0, 8.0]))
    torch.testing.assert_close(codes.flatten(), torch.tensor([1.0, 1.0]))


def test_conv_transpose_scale_tracks_logical_output_channels():
    layer = nn.ConvTranspose2d(2, 2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[[1.0]], [[10.0]]], [[[3.0]], [[30.0]]]]))
    wrapper = TernaryQLayer(layer, "deconv", activation_mode="a4")
    _, _, scale = wrapper.quantized_weight()
    torch.testing.assert_close(scale[:, 0, 0, 0], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(scale[:, 1, 0, 0], torch.tensor([20.0, 20.0]))


def test_a4_activation_matches_existing_levels_and_integer_bypass():
    wrapper = TernaryQLayer(nn.Linear(4, 1), "linear", activation_mode="a4")
    integer_spikes = torch.tensor([[0.0, 1.0, 4.0, 8.0]])
    quantized, codes, scale = wrapper.quantized_activation(integer_spikes)
    assert quantized.data_ptr() == integer_spikes.data_ptr()
    torch.testing.assert_close(codes, integer_spikes)
    assert scale.item() == 1.0

    continuous = torch.tensor([[-9.0, -0.4, 0.4, 9.0]], requires_grad=True)
    quantized, codes, scale = wrapper.quantized_activation(continuous)
    assert set(codes.unique().tolist()) <= set(range(-8, 8))
    assert codes.min().item() == -8
    assert codes.max().item() == 7
    quantized.sum().backward()
    torch.testing.assert_close(continuous.grad, torch.ones_like(continuous))
    assert scale.item() == pytest.approx(9.0 / 8.0)


def test_ternary_activation_has_three_levels_and_strict_bypass():
    wrapper = TernaryQLayer(nn.Linear(4, 1), "linear", activation_mode="ternary")
    already_ternary = torch.tensor([[-1.0, 0.0, 1.0, 0.0]])
    quantized, codes, scale = wrapper.quantized_activation(already_ternary)
    assert quantized.data_ptr() == already_ternary.data_ptr()
    assert scale.item() == 1.0
    torch.testing.assert_close(codes, already_ternary)

    continuous = torch.tensor([[-3.0, -0.1, 0.1, 3.0]], requires_grad=True)
    quantized, codes, scale = wrapper.quantized_activation(continuous)
    assert set(codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert scale.item() == pytest.approx(1.55)
    quantized.sum().backward()
    torch.testing.assert_close(continuous.grad, torch.ones_like(continuous))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.deconv = nn.ConvTranspose2d(4, 2, 1)
        self.head = nn.Linear(2, 2)

    def forward(self, x):
        x = self.deconv(self.norm(self.conv(x))).mean((2, 3))
        return self.head(x)


def test_model_conversion_is_non_mutating_and_export_is_loadable():
    source = TinyModel()
    source_state = {key: value.clone() for key, value in source.state_dict().items()}
    quantized = quantize_ternary_model(source, activation_mode="ternary")

    assert not any(isinstance(module, TernaryQLayer) for module in source.modules())
    assert len(list(iter_ternary_layers(quantized))) == 3
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, source_state[key])

    exported = export_master_state_dict(quantized)
    clone = TinyModel()
    clone.load_state_dict(exported, strict=True)
    stats = ternary_model_stats(quantized)
    assert stats.layer_count == 3
    assert stats.weight_count == sum(
        module.layer.weight.numel() for _, module in iter_ternary_layers(quantized)
    )
    assert stats.negative_count + stats.zero_count + stats.positive_count == stats.weight_count


def test_time_dimension_is_preserved():
    wrapper = TernaryQLayer(nn.Conv2d(3, 4, 3, padding=1), "conv", "a4")
    output = wrapper(torch.randn(2, 3, 3, 8, 8))
    assert output.shape == (2, 3, 4, 8, 8)
