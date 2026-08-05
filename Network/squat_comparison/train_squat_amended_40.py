#!/usr/bin/env python3
"""Run the explicitly approved 40-epoch W4M4S1 QAT+SQUAT amendment."""

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
    amendment_path = OUTPUT_ROOT / "protocol_amendment_100_40.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    overrides = amendment["overrides"]
    if overrides["fp32_epochs"] != 100 or overrides["squat_epochs"] != 40:
        raise RuntimeError("invalid 100+40 protocol amendment")
    preflight = json.loads((OUTPUT_ROOT / "preflight.json").read_text(encoding="utf-8"))
    fp_checkpoint_path = OUTPUT_ROOT / "fp32_snn/checkpoint_best.pth"
    fp_last_path = OUTPUT_ROOT / "fp32_snn/checkpoint_last.pth"
    if not fp_checkpoint_path.is_file() or not fp_last_path.is_file():
        raise FileNotFoundError("completed FP32 stage checkpoints are required")
    fp_last = torch.load(fp_last_path, map_location="cpu", weights_only=False)
    if int(fp_last["epoch"]) != 100:
        raise RuntimeError(f"FP32 stage is incomplete: epoch={fp_last['epoch']}")
    fp_best = torch.load(fp_checkpoint_path, map_location="cpu", weights_only=False)
    fp_model, _ = build_fp32_lif_model(
        config=DEFAULT_CONFIG, classes=protocol.classes, seed=protocol.seed
    )
    fp_model.load_state_dict(fp_best["model_state"], strict=True)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    atomic_write_json(
        output_dir / "factory_audit.json",
        {
            **audit.__dict__,
            "protocol_amendment": str(amendment_path),
            "initialization_checkpoint": str(fp_checkpoint_path),
            "initialization_checkpoint_epoch": fp_best["epoch"],
            "initialization_checkpoint_best_miou": fp_best["best_miou"],
        },
    )
    settings = StageSettings(
        name="w4m4s1_qat_squat_amended_40_epochs",
        epochs=40,
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

