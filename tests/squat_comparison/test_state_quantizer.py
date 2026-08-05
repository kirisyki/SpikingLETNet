from __future__ import annotations

import math

import torch

from squat_comparison.state_quantizer import (
    ThresholdCenteredStateQuantizer,
    threshold_centered_levels,
)


def reference_levels() -> torch.Tensor:
    threshold, lower_limit, upper_limit, multiplier = 1.0, 0.0, 0.2, 0.5
    num_levels = 2 << (4 - 1)
    maximum = threshold + threshold * upper_limit
    minimum = -(threshold + threshold * lower_limit)
    lower_range = threshold - minimum
    upper_range = maximum - threshold
    lower_count = math.floor(num_levels * lower_range / (maximum - minimum))
    upper_count = num_levels - lower_count
    values = []
    lower_room = sum(multiplier**j for j in reversed(range(lower_count))) / lower_range
    current = minimum
    for j in range(lower_count):
        values.append(current)
        current += multiplier**j / lower_room
    upper_room = sum(multiplier**j for j in reversed(range(upper_count))) / upper_range
    current = threshold
    for j in reversed(range(upper_count)):
        current += multiplier**j / upper_room
        values.append(current)
    return torch.tensor(values)


def test_levels_match_public_reference_semantics() -> None:
    levels = threshold_centered_levels()
    assert levels.numel() == 16
    torch.testing.assert_close(levels, reference_levels(), rtol=0, atol=0)
    assert levels[0].item() == -1.0
    assert torch.isclose(levels[-1], torch.tensor(1.2))


def test_nearest_endpoint_and_lower_tie_behavior() -> None:
    quantizer = ThresholdCenteredStateQuantizer()
    levels = quantizer.levels
    midpoint = (levels[4] + levels[5]) / 2
    values = torch.tensor([-100.0, midpoint.item(), 100.0])
    result = quantizer(values)
    torch.testing.assert_close(result, torch.stack((levels[0], levels[4], levels[-1])))


def test_state_quantizer_backward_is_identity_ste() -> None:
    quantizer = ThresholdCenteredStateQuantizer()
    value = torch.tensor([-1.4, -0.2, 0.8, 1.5], requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
    (quantizer(value) * weights).sum().backward()
    torch.testing.assert_close(value.grad, weights)

