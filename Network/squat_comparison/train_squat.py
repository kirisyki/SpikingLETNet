#!/usr/bin/env python3
"""Train W4M4S1 QAT+SQUAT from the best native FP32-LIF checkpoint."""

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

from squat_comparison.model_factory import (  # noqa: E402
    build_fp32_lif_model,
    build_squat_from_fp32,
)
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
    preflight = json.loads((OUTPUT_ROOT / "preflight.json").read_text(encoding="utf-8"))
    budget = preflight["budget_rule"]
    if not budget["formal_training_allowed"]:
        raise RuntimeError(budget["stop_reason"])
    fp_checkpoint_path = OUTPUT_ROOT / "fp32_snn/checkpoint_best.pth"
    if not fp_checkpoint_path.is_file():
        raise FileNotFoundError("best native FP32-LIF checkpoint is required")
    fp_checkpoint = torch.load(fp_checkpoint_path, map_location="cpu", weights_only=False)
    fp_model, _ = build_fp32_lif_model(
        config=DEFAULT_CONFIG, classes=protocol.classes, seed=protocol.seed
    )
    fp_model.load_state_dict(fp_checkpoint["model_state"], strict=True)
    model, audit = build_squat_from_fp32(
        fp_model, expected_weight_layers=protocol.expected_weight_layers
    )
    output_dir = OUTPUT_ROOT / "w4m4s1_squat"
    if args.resume is None:
        prepare_new_directory(output_dir)
    else:
        resume = assert_writable_experiment_path(args.resume)
        if resume.parent != output_dir.resolve():
            raise ValueError("resume checkpoint must belong to w4m4s1_squat output")
        args.resume = resume
    atomic_write_json(
        output_dir / "factory_audit.json",
        {
            **audit.__dict__,
            "initialization_checkpoint": str(fp_checkpoint_path),
            "initialization_checkpoint_epoch": fp_checkpoint["epoch"],
            "initialization_checkpoint_best_miou": fp_checkpoint["best_miou"],
        },
    )
    settings = StageSettings(
        name="w4m4s1_qat_squat",
        epochs=int(budget["selected_squat_epochs"]),
        learning_rate=protocol.squat_learning_rate,
        physical_batch=int(preflight["squat"]["physical_batch"]),
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

