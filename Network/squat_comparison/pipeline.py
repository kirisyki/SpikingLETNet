#!/usr/bin/env python3
"""Run the approved isolated pipeline; intentionally has no QAD compute node."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_STAGES = (
    "unit_tests",
    "preflight",
    "train_fp32_lif_from_random_init",
    "train_w4m4s1_squat_from_best_fp32_lif",
    "evaluate_new_models_and_read_frozen_qad_json",
)
PROHIBITED_TOKENS = ("qad_train", "qad_eval", "qad_convert")


def commands() -> list[list[str]]:
    python = sys.executable
    return [
        [python, "-m", "pytest", "-q", "tests/squat_comparison"],
        [python, "Network/squat_comparison/preflight.py"],
        [python, "Network/squat_comparison/train_fp32.py"],
        [python, "Network/squat_comparison/train_squat.py"],
        [python, "Network/squat_comparison/evaluate.py"],
    ]


def assert_no_qad_compute_nodes() -> None:
    lowered = " ".join(PIPELINE_STAGES).lower()
    if any(token in lowered for token in PROHIBITED_TOKENS):
        raise RuntimeError("QAD compute node found in isolated pipeline")


def main() -> None:
    assert_no_qad_compute_nodes()
    for command in commands():
        print(f"running: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

