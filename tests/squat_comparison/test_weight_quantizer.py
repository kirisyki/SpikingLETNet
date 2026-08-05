from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from squat_comparison.weight_quantizer import (
    W4Conv2d,
    W4ConvTranspose2d,
    W4Linear,
    quantize_weight_ste,
    replace_weight_layers,
)


def test_w4_codebook_scale_and_identity_ste() -> None:
    weight = torch.tensor([-2.0, -0.3, 0.0, 0.4, 2.0], requires_grad=True)
    quantized, scale, code = quantize_weight_ste(weight)
    torch.testing.assert_close(scale, torch.tensor(2.0 / 7.0))
    assert code.min() >= -7 and code.max() <= 7
    torch.testing.assert_close(quantized.detach(), code * scale)
    quantized.sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


def test_replacement_preserves_bias_and_parameter_keys() -> None:
    model = nn.Sequential(
        layer.Conv2d(2, 3, 3, bias=True),
        nn.Flatten(),
        nn.Linear(12, 4, bias=True),
        layer.ConvTranspose2d(4, 2, 2, bias=True),
    )
    keys = tuple(model.state_dict())
    biases = [model[0].bias, model[2].bias, model[3].bias]
    names = replace_weight_layers(model)
    assert names == ["0", "2", "3"]
    assert isinstance(model[0], W4Conv2d)
    assert isinstance(model[2], W4Linear)
    assert isinstance(model[3], W4ConvTranspose2d)
    assert [model[0].bias, model[2].bias, model[3].bias] == biases
    assert tuple(model.state_dict()) == keys

