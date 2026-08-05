#!/usr/bin/env python3
"""Train the approved random-init native FP32 LIF-SNN stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

NETWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NETWORK_DIR))
os.chdir(NETWORK_DIR)

import torch  # noqa: E402

from squat_comparison.model_factory import build_fp32_lif_model  # noqa: E402
from squat_comparison.protocol import (  # noqa: E402
    DEFAULT_CONFIG,
    OUTPUT_ROOT,
    ExperimentProtocol,
    atomic_write_json,
    assert_writable_experiment_path,
    prepare_new_directory,
)
from squat_comparison.training import StageSettings, run_training_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = ExperimentProtocol()
    preflight_path = OUTPUT_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("approved preflight must run before formal training")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    budget = preflight["budget_rule"]
    if not budget["formal_training_allowed"]:
        raise RuntimeError(budget["stop_reason"])
    output_dir = OUTPUT_ROOT / "fp32_snn"
    if args.resume is None:
        prepare_new_directory(output_dir)
    else:
        resume = assert_writable_experiment_path(args.resume)
        if resume.parent != output_dir.resolve():
            raise ValueError("resume checkpoint must belong to fp32_snn output")
        args.resume = resume
    model, audit = build_fp32_lif_model(
        config=DEFAULT_CONFIG, classes=protocol.classes, seed=protocol.seed
    )
    atomic_write_json(output_dir / "factory_audit.json", audit.__dict__)
    settings = StageSettings(
        name="fp32_lif_snn",
        epochs=int(budget["selected_fp32_epochs"]),
        learning_rate=protocol.fp32_learning_rate,
        physical_batch=int(preflight["fp32"]["physical_batch"]),
    )
    try:
        run_training_stage(
            model=model,
            protocol=protocol,
            settings=settings,
            output_dir=output_dir,
            device=torch.device("cuda"),
            resume=args.resume,
            log_interval=args.log_interval,
        )
    except FloatingPointError as error:
        atomic_write_json(
            output_dir / "numerical_failure.json",
            {"stage": settings.name, "error": str(error), "policy": "run terminated"},
        )
        raise


if __name__ == "__main__":
    main()

