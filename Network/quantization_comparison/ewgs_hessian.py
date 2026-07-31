"""Hutchinson-trace update for EWGS element-wise backward scales.

The production defaults reproduce the public EWGS protocol: ten batches,
up to fifty Hutchinson iterations, and a relative tolerance of ``1e-3``.
Smaller values are exposed only for explicitly marked smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from .ewgs_qat import EWGSQuantizedLayer, iter_ewgs_layers
from .layer_adapter import repeat_time, temporal_average


@dataclass(frozen=True)
class HessianUpdateResult:
    batches: int
    mean_iterations: float
    weight_scales: dict[str, float]
    activation_scales: dict[str, float]
    inactive_layers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "batches": self.batches,
            "mean_iterations": self.mean_iterations,
            "weight_scales": self.weight_scales,
            "activation_scales": self.activation_scales,
            "inactive_layers": list(self.inactive_layers),
        }


def _rademacher_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor).bernoulli_(0.5).mul_(2.0).sub_(1.0)


def hutchinson_trace(
    loss: torch.Tensor,
    tensors: Sequence[torch.Tensor],
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    """Estimate Hessian traces for all tensors in one shared HVP loop."""

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    gradients = torch.autograd.grad(
        loss,
        tensors,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    if any(gradient is None for gradient in gradients):
        missing = [index for index, gradient in enumerate(gradients) if gradient is None]
        raise RuntimeError(f"EWGS Hessian tensors disconnected at indices {missing}")

    typed_gradients = [gradient for gradient in gradients if gradient is not None]
    running = [tensor.new_zeros(()) for tensor in tensors]
    previous: list[torch.Tensor] | None = None

    for iteration in range(1, max_iterations + 1):
        vectors = [_rademacher_like(tensor) for tensor in tensors]
        products = torch.autograd.grad(
            typed_gradients,
            tensors,
            grad_outputs=vectors,
            retain_graph=True,
            allow_unused=True,
        )
        if any(product is None for product in products):
            missing = [index for index, product in enumerate(products) if product is None]
            raise RuntimeError(f"EWGS Hessian-vector products missing at {missing}")

        current: list[torch.Tensor] = []
        for index, (product, vector) in enumerate(zip(products, vectors)):
            assert product is not None
            sample = (product * vector).sum().detach()
            running[index] = running[index] + sample
            current.append(running[index] / iteration)

        if previous is not None:
            relative_changes = [
                ((new - old).abs() / new.abs().clamp_min(1e-12)).item()
                for new, old in zip(current, previous)
            ]
            if max(relative_changes, default=0.0) < tolerance:
                return current, typed_gradients, iteration
        previous = [value.clone() for value in current]

    assert previous is not None
    return previous, typed_gradients, max_iterations


def _unpack_batch(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("expected a batch containing images and labels")
    return batch[0], batch[1]


def update_ewgs_backward_scales(
    model: nn.Module,
    loader: Iterable[object],
    criterion: nn.Module,
    device: torch.device,
    *,
    time_steps: int = 1,
    batches: int = 10,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
) -> HessianUpdateResult:
    """Estimate and assign the two EWGS delta values for every wrapped layer."""

    if batches <= 0:
        raise ValueError("batches must be positive")
    named_layers = list(iter_ewgs_layers(model))
    if not named_layers:
        raise ValueError("model has no EWGS layers")

    for _, layer in named_layers:
        layer.capture_hessian_tensors = True

    weight_totals = {name: 0.0 for name, _ in named_layers}
    activation_totals = {name: 0.0 for name, _ in named_layers}
    iteration_counts: list[int] = []
    processed = 0
    inactive_names: tuple[str, ...] | None = None
    model.train()

    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches:
                break
            images, labels = _unpack_batch(batch)
            images = repeat_time(
                images.to(device, non_blocking=True), time_steps
            )
            labels = labels.long().to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            forward_output = model.forward_qat(images)
            logits = forward_output[0] if isinstance(forward_output, tuple) else forward_output
            loss = criterion(temporal_average(logits), labels)

            active_named_layers = [
                (name, layer)
                for name, layer in named_layers
                if (
                    layer.last_quantized_weight is not None
                    and layer.last_quantized_activation is not None
                )
            ]
            current_inactive_names = tuple(
                name
                for name, layer in named_layers
                if (
                    layer.last_quantized_weight is None
                    or layer.last_quantized_activation is None
                )
            )
            if inactive_names is None:
                inactive_names = current_inactive_names
            elif inactive_names != current_inactive_names:
                raise RuntimeError("EWGS active layer set changed across Hessian batches")
            if not active_named_layers:
                raise RuntimeError("EWGS forward did not activate any quantized layer")

            tensors: list[torch.Tensor] = []
            for _, layer in active_named_layers:
                assert layer.last_quantized_weight is not None
                assert layer.last_quantized_activation is not None
                tensors.extend(
                    [layer.last_quantized_weight, layer.last_quantized_activation]
                )

            traces, gradients, used_iterations = hutchinson_trace(
                loss,
                tensors,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            iteration_counts.append(used_iterations)

            for layer_index, (name, _) in enumerate(active_named_layers):
                weight_position = 2 * layer_index
                activation_position = weight_position + 1
                weight_gradient = gradients[weight_position].detach()
                activation_gradient = gradients[activation_position].detach()
                weight_denominator = (
                    3.0 * weight_gradient.std(unbiased=False)
                ).clamp_min(1e-12)
                activation_denominator = (
                    3.0 * activation_gradient.std(unbiased=False)
                ).clamp_min(1e-12)
                weight_delta = (
                    traces[weight_position]
                    / tensors[weight_position].numel()
                    / weight_denominator
                ).clamp_min(0.0)
                activation_delta = (
                    traces[activation_position]
                    / tensors[activation_position].numel()
                    / activation_denominator
                ).clamp_min(0.0)
                weight_totals[name] += float(weight_delta.item())
                activation_totals[name] += float(activation_delta.item())

            processed += 1
            functional.reset_net(model)
            del loss, logits, forward_output, tensors, traces, gradients
    finally:
        for _, layer in named_layers:
            layer.capture_hessian_tensors = False
            layer.last_quantized_weight = None
            layer.last_quantized_activation = None
        model.zero_grad(set_to_none=True)
        functional.reset_net(model)

    if processed != batches:
        raise RuntimeError(f"requested {batches} Hessian batches, observed {processed}")

    for name, layer in named_layers:
        layer.set_backward_scales(
            weight_totals[name] / processed,
            activation_totals[name] / processed,
        )

    return HessianUpdateResult(
        batches=processed,
        mean_iterations=sum(iteration_counts) / len(iteration_counts),
        weight_scales={
            name: weight_totals[name] / processed for name, _ in named_layers
        },
        activation_scales={
            name: activation_totals[name] / processed for name, _ in named_layers
        },
        inactive_layers=inactive_names or (),
    )
