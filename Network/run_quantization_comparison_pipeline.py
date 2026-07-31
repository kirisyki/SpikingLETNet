#!/usr/bin/env python3
"""Stage runner for the approved W4A4 comparison protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from quantization_comparison.protocol import ExperimentProtocol


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
RESULT_ROOT = REPO_ROOT / "quantization_comparison_results/seed1234"
SMOKE_ROOT = RESULT_ROOT / "smoke"
CHECKPOINT_ROOT = REPO_ROOT / "quantization_comparison_checkpoint/udd/seed1234"
EWGS_GPU_HOUR_GATE = 72.0
METHOD_DIRS = {
    "ste": "ste_qat_w4a4",
    "lsq": "lsq_qat_w4a4",
    "ewgs": "ewgs_qat_w4a4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "preflight",
            "smoke",
            "formal-ste-lsq",
            "formal-ewgs",
            "evaluate",
            "all",
        ],
    )
    return parser.parse_args()


def run(command: Sequence[str], *, cwd: Path) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def preflight() -> None:
    tests = [
        "tests/test_quantization_comparison_quantizers.py",
        "tests/test_quantization_comparison_model_factory.py",
        "tests/test_quantization_comparison_protocol.py",
        "tests/test_quantization_comparison_checkpointing.py",
    ]
    run([sys.executable, "-m", "pytest", "-q", *tests], cwd=REPO_ROOT)


def smoke() -> None:
    for method in ("ste", "lsq", "ewgs"):
        output = SMOKE_ROOT / METHOD_DIRS[method]
        run(
            [
                sys.executable,
                "train_quantization_baseline.py",
                "--method",
                method,
                "--smoke",
                "--output-dir",
                str(output),
            ],
            cwd=NETWORK_DIR,
        )


def ewgs_projection() -> float:
    summary_path = SMOKE_ROOT / METHOD_DIRS["ewgs"] / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"EWGS smoke summary required before the gate: {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("hessian_records") or []
    if not records:
        raise KeyError("EWGS smoke summary has no Hessian timing record")
    observed_seconds = float(records[-1]["seconds"])
    update_epochs = max(ExperimentProtocol("ewgs").run_epochs - 1, 0)
    return observed_seconds * update_epochs * 10 * 50 / 3600.0


def enforce_ewgs_gate() -> None:
    projection = ewgs_projection()
    gate = {
        "projected_ewgs_hessian_gpu_hours": projection,
        "gate_gpu_hours": EWGS_GPU_HOUR_GATE,
        "passed": projection <= EWGS_GPU_HOUR_GATE,
        "action": (
            "formal EWGS may proceed"
            if projection <= EWGS_GPU_HOUR_GATE
            else "pause for explicit user approval; do not reduce Hessian settings"
        ),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "ewgs_runtime_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2), flush=True)
    if not gate["passed"]:
        raise RuntimeError(
            "EWGS formal run blocked by the preregistered 72 GPU-hour gate"
        )


def formal_ste_lsq() -> None:
    for method in ("ste", "lsq"):
        run(
            [
                sys.executable,
                "train_quantization_baseline.py",
                "--method",
                method,
                "--output-dir",
                str(CHECKPOINT_ROOT / METHOD_DIRS[method]),
            ],
            cwd=NETWORK_DIR,
        )


def formal_ewgs() -> None:
    enforce_ewgs_gate()
    run(
        [
            sys.executable,
            "train_quantization_baseline.py",
            "--method",
            "ewgs",
            "--output-dir",
            str(CHECKPOINT_ROOT / METHOD_DIRS["ewgs"]),
        ],
        cwd=NETWORK_DIR,
    )


def evaluate() -> None:
    run(
        [
            sys.executable,
            "evaluate_quantization_comparison.py",
            "--output-dir",
            str(RESULT_ROOT),
        ],
        cwd=NETWORK_DIR,
    )


def main() -> None:
    stage = parse_args().stage
    if stage in ("preflight", "all"):
        preflight()
    if stage in ("smoke", "all"):
        smoke()
    if stage in ("formal-ste-lsq", "all"):
        formal_ste_lsq()
    if stage in ("formal-ewgs", "all"):
        formal_ewgs()
    if stage in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
