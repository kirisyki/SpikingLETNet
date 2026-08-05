#!/usr/bin/env python3
"""Run the explicitly approved 100-epoch FP32-LIF protocol amendment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
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


AMENDMENT_PATH = OUTPUT_ROOT / "protocol_amendment_100_40.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def validated_amendment() -> tuple[ExperimentProtocol, dict[str, object]]:
    protocol = ExperimentProtocol()
    preflight_path = OUTPUT_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("preflight.json is required")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    rule = preflight["budget_rule"]
    if rule["selected_fp32_epochs"] != 100 or rule["selected_squat_epochs"] != 40:
        raise RuntimeError("preflight minimum budget is not 100+40")
    if rule["formal_training_allowed"] is not False:
        raise RuntimeError("this amendment is only valid for the explicit >=48h override")
    amendment: dict[str, object] = {
        "format": "qad-vs-squat-protocol-amendment-v1",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approval": "user explicitly requested training with the 100+40 plan",
        "overrides": {
            "formal_training_allowed": True,
            "fp32_epochs": 100,
            "squat_epochs": 40,
            "accepted_estimated_hours": rule["minimum_100_plus_40_estimated_hours"],
        },
        "unchanged": {
            "physical_batch": 16,
            "effective_batch": 64,
            "time_steps": 8,
            "full_validation_each_epoch": True,
            "historical_qad_compute": False,
        },
    }
    return protocol, {"preflight": preflight, "amendment": amendment}


def main() -> None:
    args = parse_args()
    protocol, context = validated_amendment()
    preflight = context["preflight"]
    amendment = context["amendment"]
    print(json.dumps(amendment, indent=2), flush=True)
    if args.check_only:
        return
    atomic_write_json(AMENDMENT_PATH, amendment)
    output_dir = OUTPUT_ROOT / "fp32_snn"
    if args.resume is None:
        prepare_new_directory(output_dir)
    else:
        resume = assert_writable_experiment_path(args.resume)
        if resume.parent != output_dir.resolve():
            raise ValueError("resume checkpoint must belong to fp32_snn output")
        args.resume = resume
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, audit = build_fp32_lif_model(
        config=DEFAULT_CONFIG, classes=protocol.classes, seed=protocol.seed
    )
    atomic_write_json(
        output_dir / "factory_audit.json",
        {
            **audit.__dict__,
            "protocol_amendment": str(AMENDMENT_PATH),
        },
    )
    settings = StageSettings(
        name="fp32_lif_snn_amended_100_epochs",
        epochs=100,
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

