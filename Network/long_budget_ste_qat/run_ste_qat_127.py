#!/usr/bin/env python3
"""Run a fresh 127-epoch task-only W4A4 STE-QAT comparison on UDD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


NETWORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NETWORK_DIR.parent
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))

import train_quantization_baseline as training  # noqa: E402
from long_budget_ste_qat.protocol import LongBudgetSTEProtocol  # noqa: E402
from quantization.int4_selfbuild import QLayer  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "quantization_comparison_checkpoint/udd/seed1234/"
    "ste_qat_w4a4_long127"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-iters", type=int, default=1)
    parser.add_argument("--max-val-iters", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def audited_build_quantized_model(
    method: str,
    model: Any,
    *,
    expected_layer_count: int | None = 71,
) -> Any:
    converted = ORIGINAL_BUILD_MODEL(
        method, model, expected_layer_count=expected_layer_count
    )
    layers = [
        (name, module)
        for name, module in converted.named_modules()
        if isinstance(module, QLayer)
    ]
    if len(layers) != 71:
        raise RuntimeError(f"expected 71 historical QLayers, observed {len(layers)}")
    if layers[0][0] != "init_conv.0.conv":
        raise RuntimeError(f"unexpected first QLayer: {layers[0][0]}")
    if any(not module.quant for _, module in layers):
        raise RuntimeError("all 71 weights must remain W4")
    if any(not module.activation_quant for _, module in layers):
        raise RuntimeError("all 71 layer inputs, including the image, must remain A4")
    return converted


def long_source_manifest(protocol: LongBudgetSTEProtocol) -> dict[str, Any]:
    manifest = ORIGINAL_SOURCE_MANIFEST(protocol)
    manifest["long_budget_experiment"] = {
        "objective": "remove the 16-vs-127 epoch confound in QAD versus STE-QAT",
        "fresh_start_from_fp_checkpoint": True,
        "resume_from_short16": False,
        "student_loss": "segmentation cross-entropy only",
        "teacher_used": False,
        "weight_policy": "W4 for all 71 QLayers",
        "input_policy": "historical A4/integer-bypass policy for all 71 QLayers",
        "normalized_image_input": "A4 dynamic per-tensor fake quantization",
        "run_epochs": 127,
        "schedule_epochs": 150,
        "selection": "best full-validation mIoU across 127 epochs",
        "comparison_target": {
            "method": "historical QAD",
            "observed_training_epochs": 127,
            "best_checkpoint_epoch": 119,
            "unified_validation_miou": 0.6195473169172718,
        },
    }
    return manifest


ORIGINAL_BUILD_MODEL = training.build_quantized_model
ORIGINAL_SOURCE_MANIFEST = training.source_manifest


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (REPO_ROOT / args.output_dir).resolve()
    )
    if output_dir == DEFAULT_OUTPUT.parent:
        raise RuntimeError("refusing to use the checkpoint root itself as output")

    training.ExperimentProtocol = LongBudgetSTEProtocol
    training.build_quantized_model = audited_build_quantized_model
    training.source_manifest = long_source_manifest

    forwarded = [
        str(Path(training.__file__).resolve()),
        "--method",
        "ste",
        "--output-dir",
        str(output_dir),
        "--log-interval",
        str(args.log_interval),
    ]
    if args.smoke:
        forwarded.extend(
            [
                "--smoke",
                "--max-train-iters",
                str(args.max_train_iters),
                "--max-val-iters",
                str(args.max_val_iters),
            ]
        )
    sys.argv = forwarded
    print(
        json.dumps(
            {
                "entrypoint": "long_budget_ste_qat",
                "output_dir": str(output_dir),
                "protocol": LongBudgetSTEProtocol("ste").to_dict(),
                "smoke": args.smoke,
            },
            indent=2,
        ),
        flush=True,
    )
    training.main()


if __name__ == "__main__":
    main()
