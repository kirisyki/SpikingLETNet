#!/usr/bin/env python3
"""Durable tmux controller with CPU-safe CUDA checkpoint resume."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NETWORK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = NETWORK_DIR.parent
sys.path.insert(0, str(NETWORK_DIR))
os.chdir(REPO_ROOT)

import torch  # noqa: E402

from squat_comparison.protocol import OUTPUT_ROOT, atomic_write_json  # noqa: E402


LOG_PATH = OUTPUT_ROOT / "tmux_pipeline.log"
STATUS_PATH = OUTPUT_ROOT / "tmux_pipeline_status.json"


def install_durable_log() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    os.close(descriptor)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


def checkpoint_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def write_status(state: str, **extra: Any) -> None:
    atomic_write_json(
        STATUS_PATH,
        {
            "format": "qad-vs-squat-tmux-controller-v2",
            "state": state,
            "controller_pid": os.getpid(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def run_stage(state: str, command: list[str], **status: Any) -> None:
    write_status(state, command=command, **status)
    print(f"[{datetime.now(timezone.utc).isoformat()}] starting {state}", flush=True)
    print("command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"[{datetime.now(timezone.utc).isoformat()}] completed {state}", flush=True)


def main() -> None:
    install_durable_log()
    python = sys.executable
    amendment = OUTPUT_ROOT / "protocol_amendment_100_40.json"
    approved = json.loads(amendment.read_text(encoding="utf-8"))["overrides"]
    if approved["fp32_epochs"] != 100 or approved["squat_epochs"] != 40:
        raise RuntimeError("tmux controller requires the approved 100+40 amendment")

    fp_last = OUTPUT_ROOT / "fp32_snn/checkpoint_last.pth"
    fp_epoch = checkpoint_epoch(fp_last)
    if fp_epoch is None or fp_epoch > 100:
        raise RuntimeError(f"invalid FP32 resume epoch: {fp_epoch}")
    if fp_epoch < 100:
        run_stage(
            "training_fp32",
            [
                python,
                "Network/squat_comparison/train_fp32_amended_100_resume_safe.py",
                "--resume",
                str(fp_last),
                "--log-interval",
                "20",
            ],
            resume_epoch=fp_epoch,
            target_epoch=100,
            resume_compatibility="checkpoint payload kept on CPU for DataLoader RNG",
        )
    if checkpoint_epoch(fp_last) != 100:
        raise RuntimeError("FP32 stage returned without epoch 100")

    squat_last = OUTPUT_ROOT / "w4m4s1_squat/checkpoint_last.pth"
    squat_epoch = checkpoint_epoch(squat_last)
    if squat_epoch is not None and squat_epoch > 40:
        raise RuntimeError(f"invalid SQUAT checkpoint epoch: {squat_epoch}")
    if squat_epoch != 40:
        if squat_epoch is None:
            command = [
                python,
                "Network/squat_comparison/train_squat_amended_40.py",
                "--log-interval",
                "20",
            ]
        else:
            command = [
                python,
                "Network/squat_comparison/train_squat_amended_40_resume_safe.py",
                "--resume",
                str(squat_last),
                "--log-interval",
                "20",
            ]
        run_stage(
            "training_squat",
            command,
            resume_epoch=squat_epoch,
            target_epoch=40,
        )
    if checkpoint_epoch(squat_last) != 40:
        raise RuntimeError("SQUAT stage returned without epoch 40")

    comparison = OUTPUT_ROOT / "evaluation/comparison.json"
    if not comparison.is_file():
        run_stage(
            "evaluating",
            [python, "Network/squat_comparison/evaluate.py"],
            fp32_epoch=100,
            squat_epoch=40,
        )
    write_status("complete", fp32_epoch=100, squat_epoch=40, comparison=str(comparison))
    print(f"[{datetime.now(timezone.utc).isoformat()}] pipeline complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            write_status("failed", error_type=type(error).__name__, error=str(error))
        finally:
            raise

