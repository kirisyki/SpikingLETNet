"""Compatibility shim for CPU-only DataLoader RNG state during CUDA resume."""

from __future__ import annotations

from typing import Any

import torch


def install_cpu_checkpoint_restore() -> None:
    """Make ``training.run_training_stage`` load resume payloads on CPU.

    PyTorch model and optimizer loaders copy/cast CPU state into CUDA-backed
    objects. Keeping the payload on CPU is required because the DataLoader's
    ``torch.Generator`` rejects CUDA RNG-state tensors.
    """

    from . import checkpointing, training

    if getattr(training.load_checkpoint, "_squat_cpu_restore", False):
        return

    def cpu_restore(
        path: Any,
        *,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        restore_rng: bool = False,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        del map_location
        return checkpointing.load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=restore_rng,
            map_location="cpu",
        )

    cpu_restore._squat_cpu_restore = True  # type: ignore[attr-defined]
    training.load_checkpoint = cpu_restore

