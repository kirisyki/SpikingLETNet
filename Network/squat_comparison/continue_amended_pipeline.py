#!/usr/bin/env python3
"""Safely continue SQUAT/evaluation after the amended FP32 stage completes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

NETWORK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = NETWORK_DIR.parent
sys.path.insert(0, str(NETWORK_DIR))

import torch  # noqa: E402

from squat_comparison.protocol import OUTPUT_ROOT, atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def write_status(state: str, **extra: object) -> None:
    atomic_write_json(
        OUTPUT_ROOT / "continuation_status.json",
        {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        },
    )


def checkpoint_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def main() -> None:
    args = parse_args()
    if not 5 <= args.poll_seconds <= 60:
        raise ValueError("poll interval must be within 5..60 seconds")
    last_path = OUTPUT_ROOT / "fp32_snn/checkpoint_last.pth"
    last_mtime: int | None = None
    observed_epoch: int | None = None
    write_status("waiting_for_fp32", fp_pid=args.fp_pid, observed_epoch=None)

    while True:
        if last_path.is_file():
            mtime = last_path.stat().st_mtime_ns
            if mtime != last_mtime:
                observed_epoch = checkpoint_epoch(last_path)
                last_mtime = mtime
                print(f"observed FP32 checkpoint epoch={observed_epoch}", flush=True)
                write_status(
                    "waiting_for_fp32",
                    fp_pid=args.fp_pid,
                    observed_epoch=observed_epoch,
                )
        alive = process_alive(args.fp_pid)
        if observed_epoch == 100:
            if alive:
                time.sleep(args.poll_seconds)
                continue
            break
        if not alive:
            write_status(
                "fp32_incomplete",
                fp_pid=args.fp_pid,
                observed_epoch=observed_epoch,
                action="SQUAT not started",
            )
            raise RuntimeError(
                f"FP32 process exited before epoch 100 (observed={observed_epoch})"
            )
        time.sleep(args.poll_seconds)

    write_status("starting_squat", fp_epoch=observed_epoch)
    subprocess.run(
        [sys.executable, "Network/squat_comparison/train_squat_amended_40.py"],
        cwd=REPO_ROOT,
        check=True,
    )
    write_status("starting_evaluation", fp_epoch=100, squat_epoch=40)
    subprocess.run(
        [sys.executable, "Network/squat_comparison/evaluate.py"],
        cwd=REPO_ROOT,
        check=True,
    )
    write_status("complete", fp_epoch=100, squat_epoch=40)


if __name__ == "__main__":
    main()

