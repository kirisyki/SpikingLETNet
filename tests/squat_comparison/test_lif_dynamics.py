from __future__ import annotations

import math

import torch

from squat_comparison.lif_neuron import DirectLIF, atan_spike


def test_delayed_subtractive_reset_order() -> None:
    lif = DirectLIF(step_mode="s")
    first = lif(torch.tensor([1.2]))
    assert first.item() == 1.0
    torch.testing.assert_close(lif.mem, torch.tensor([1.2]))
    second = lif(torch.tensor([0.0]))
    assert second.item() == 0.0
    torch.testing.assert_close(lif.mem, torch.tensor([-0.4]))


def test_multistep_is_recurrent_and_reset_clears_batch_state() -> None:
    lif = DirectLIF(step_mode="m")
    sequence = torch.full((3, 1), 0.6)
    recurrent = lif(sequence)
    torch.testing.assert_close(recurrent, torch.tensor([[0.0], [0.0], [1.0]]))
    lif.reset()
    assert lif.mem is None
    repeated_independent = torch.stack(
        [DirectLIF(step_mode="s")(sequence[t]) for t in range(3)]
    )
    assert not torch.equal(recurrent, repeated_independent)


def test_state_is_quantized_before_firing() -> None:
    lif = DirectLIF(quantize_state=True, step_mode="s")
    spike = lif(torch.tensor([1.01]))
    assert any(torch.equal(lif.mem, level.reshape_as(lif.mem)) for level in lif.state_quantizer.levels)
    assert spike.item() in (0.0, 1.0)


def test_atan_surrogate_derivative_matches_formula() -> None:
    value = torch.tensor([-0.4, 0.0, 0.7], dtype=torch.double, requires_grad=True)
    atan_spike(value, 2.0).sum().backward()
    expected = 1.0 / (1.0 + (math.pi * value.detach()).square())
    torch.testing.assert_close(value.grad, expected)
    assert atan_spike(torch.tensor([0.0, 1e-6])).tolist() == [0.0, 1.0]

