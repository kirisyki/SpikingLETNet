"""Auditable helpers for the user-approved 127-to-16 epoch amendment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .checkpointing import CHECKPOINT_FORMAT, atomic_torch_save


ALLOWED_OLD_VERSION = "w4a4-comparison-v1"
AMENDED_VERSION = "w4a4-comparison-v2-short16"
OLD_RUN_EPOCHS = 127
AMENDED_RUN_EPOCHS = 16


def is_compatible_epoch_amendment(
    saved: dict[str, Any], current: dict[str, Any]
) -> bool:
    """Accept only the exact approved version/run-budget change."""

    saved_copy = dict(saved)
    current_copy = dict(current)
    saved_version = saved_copy.pop("protocol_version", None)
    current_version = current_copy.pop("protocol_version", None)
    saved_epochs = saved_copy.pop("run_epochs", None)
    current_epochs = current_copy.pop("run_epochs", None)
    return (
        saved_version == ALLOWED_OLD_VERSION
        and current_version == AMENDED_VERSION
        and saved_epochs == OLD_RUN_EPOCHS
        and current_epochs == AMENDED_RUN_EPOCHS
        and saved_copy == current_copy
    )


def amend_checkpoint_metadata(
    path: Path,
    *,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Rewrite metadata only; model, optimizer and RNG states remain byte-equal."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format in {path}")
    old_protocol = payload.get("protocol")
    if old_protocol != protocol and not is_compatible_epoch_amendment(
        old_protocol, protocol
    ):
        raise RuntimeError(f"checkpoint is not eligible for epoch amendment: {path}")
    payload["protocol"] = protocol
    payload["manifest"] = manifest
    payload["protocol_amendment"] = {
        "old_run_epochs": OLD_RUN_EPOCHS,
        "new_run_epochs": AMENDED_RUN_EPOCHS,
        "learning_rate_schedule_epochs": 150,
        "reason": "user requested fewer epochs to reduce total training time",
    }
    atomic_torch_save(payload, path)
