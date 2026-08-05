"""Atomic, resumable checkpoints for the isolated direct-SNN stages."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .protocol import assert_writable_experiment_path


FORMAT = "direct-snn-squat-v1"


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_miou: float,
    protocol: dict[str, Any],
    loader_generator_state: torch.Tensor,
) -> None:
    resolved = assert_writable_experiment_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_miou": float(best_miou),
        "protocol": protocol,
        "rng_state": capture_rng_state(),
        "loader_generator_state": loader_generator_state,
    }
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, resolved)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {path}")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload

