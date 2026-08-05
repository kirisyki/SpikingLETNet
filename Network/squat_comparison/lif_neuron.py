"""True recurrent LIF neuron used by the direct-SNN comparison route."""

from __future__ import annotations

import math
from typing import Any

import torch
from spikingjelly.activation_based import base

from .state_quantizer import ThresholdCenteredStateQuantizer


class _ATanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.alpha = float(alpha)
        return (value > 0).to(value)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors
        alpha = ctx.alpha
        denominator = 1.0 + (math.pi * alpha * value * 0.5).square()
        return grad_output * (alpha * 0.5) / denominator, None


def atan_spike(value: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Binary step with the snnTorch arctangent surrogate derivative."""

    return _ATanSpike.apply(value, float(alpha))


class DirectLIF(base.MemoryModule):
    """LIF with delayed subtractive reset and optional SQUAT state quantization.

    At time ``t`` the reset decision is made from the saved state at ``t-1``.
    The new state is charged, optionally quantized, saved, and then used to
    generate the current spike. Multi-step mode always iterates explicitly.
    """

    def __init__(
        self,
        *,
        beta: float = 0.5,
        threshold: float = 1.0,
        alpha: float = 2.0,
        quantize_state: bool = False,
        step_mode: str = "s",
    ) -> None:
        super().__init__()
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if threshold <= 0 or alpha <= 0:
            raise ValueError("threshold and alpha must be positive")
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.quantize_state = bool(quantize_state)
        self.state_quantizer = (
            ThresholdCenteredStateQuantizer(threshold=threshold)
            if quantize_state
            else None
        )
        self.register_memory("mem", None)
        self.step_mode = step_mode

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem is None:
            previous = torch.zeros_like(x)
        else:
            if self.mem.shape != x.shape:
                raise RuntimeError(
                    "LIF state shape changed without reset: "
                    f"state={tuple(self.mem.shape)} input={tuple(x.shape)}"
                )
            previous = self.mem
        delayed_reset = (previous > self.threshold).to(x).detach()
        candidate = self.beta * previous + x - delayed_reset * self.threshold
        if self.state_quantizer is not None:
            candidate = self.state_quantizer(candidate)
        self.mem = candidate
        return atan_spike(candidate - self.threshold, self.alpha)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError("multi-step input must have leading [T, B, ...] axes")
        return torch.stack(
            [self.single_step_forward(x_seq[t]) for t in range(x_seq.shape[0])],
            dim=0,
        )

    def extra_repr(self) -> str:
        return (
            f"beta={self.beta}, threshold={self.threshold}, alpha={self.alpha}, "
            f"quantize_state={self.quantize_state}, step_mode={self.step_mode}"
        )

