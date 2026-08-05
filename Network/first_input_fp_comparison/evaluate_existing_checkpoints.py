#!/usr/bin/env python3
"""Paired inference ablation of image-input A4 for existing QAD and STE-QAT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


NETWORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NETWORK_DIR.parent
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))
os.chdir(NETWORK_DIR)

import evaluate_quantization_comparison as common  # noqa: E402
from first_input_fp_comparison.variant import (  # noqa: E402
    exempt_first_qlayer_input,
    qlayer_input_policy,
)
from quantization_comparison.protocol import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    sha256_file,
    validate_protected_hashes,
)


DEFAULT_STE = DEFAULT_CHECKPOINT_ROOT / "ste_qat_w4a4/checkpoint_best.pth"
DEFAULT_OUTPUT = REPO_ROOT / "first_input_fp_results/existing_checkpoints_20260805"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ste-checkpoint", type=Path, default=DEFAULT_STE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-val-iters", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_model(
    method: str,
    ste_checkpoint: Path,
    *,
    first_input_fp: bool,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    builders: dict[str, Callable[[], tuple[torch.nn.Module, dict[str, Any]]]] = {
        "qad": common.build_qad,
        "ste": lambda: common.build_baseline("ste", ste_checkpoint),
    }
    model, checkpoint = builders[method]()
    if first_input_fp:
        variant = exempt_first_qlayer_input(model)
    else:
        policy = qlayer_input_policy(model)
        if len(policy) != 71 or any(not row["input_quantized"] for row in policy):
            raise RuntimeError("control model is not the expected full-input W4A4 model")
        variant = {
            "variant": "first_image_input_a4",
            "first_layer": policy[0]["name"],
            "first_layer_weight_bits": policy[0]["bits"],
            "first_layer_input_bits": policy[0]["bits"],
            "later_layer_input_bits": policy[1]["bits"],
            "quantized_weight_layer_count": 71,
            "quantized_input_layer_count": 71,
        }
    return model, checkpoint, variant


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "first_input",
        "miou",
        "delta_vs_input_a4_miou_points",
        *[f"iou_{name}" for name in common.CLASS_NAMES],
        "checkpoint_epoch",
        "evaluated_batches",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(payload: dict[str, Any], path: Path) -> None:
    results = payload["results"]
    lines = [
        "# First image-input A4 inference ablation",
        "",
        "Only the normalized image input to `init_conv.0.conv` is changed. "
        "The first convolution weight remains W4 and every later QLayer input remains A4.",
        "",
        "This is a paired inference sensitivity test on existing checkpoints, not "
        "training with the alternative policy. Historical QAD observed 127 training "
        "epochs; STE-QAT used the shortened 16-epoch budget.",
        "",
        "| Method | First image input | mIoU | Delta vs input A4 (points) |",
        "|---|---|---:|---:|",
    ]
    for method in ("qad", "ste"):
        for mode in ("input_a4", "input_fp"):
            row = results[method][mode]
            lines.append(
                f"| {method.upper()} | {mode.removeprefix('input_').upper()} | "
                f"{row['miou']:.6f} | {row['delta_vs_input_a4_miou_points']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "A within-checkpoint absolute change of at least 0.5 mIoU points is "
            "treated as material enough to justify matched retraining. Smaller "
            "changes remain descriptive because this toggle was not present during training.",
            "",
            "The QAD-versus-STE gap must not be interpreted causally here because the "
            "training budgets differ.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_protected_hashes()
    ste_checkpoint = args.ste_checkpoint.resolve()
    if not ste_checkpoint.is_file():
        raise FileNotFoundError(ste_checkpoint)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (REPO_ROOT / args.output_dir).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    _, loader = common.build_dataset_test(
        "udd", num_workers=6, none_gt=False, batch_size=20
    )
    if len(loader.dataset) != common.EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(
            f"expected {common.EXPECTED_VALIDATION_SAMPLES} validation samples, "
            f"observed {len(loader.dataset)}"
        )
    limit = args.max_val_iters if args.max_val_iters > 0 else None
    results: dict[str, dict[str, Any]] = {"qad": {}, "ste": {}}
    csv_rows: list[dict[str, Any]] = []
    for method in ("qad", "ste"):
        for mode, first_input_fp in (("input_a4", False), ("input_fp", True)):
            print(f"evaluating method={method} mode={mode}", flush=True)
            model, checkpoint, variant = build_model(
                method, ste_checkpoint, first_input_fp=first_input_fp
            )
            metrics = common.evaluate_model(
                model=model,
                loader=loader,
                device=device,
                limit=limit,
                audit_bypass=True,
            )
            results[method][mode] = {
                **metrics,
                "checkpoint": checkpoint,
                "input_policy": variant,
            }
            del model
            torch.cuda.empty_cache()

        control = results[method]["input_a4"]["miou"]
        for mode in ("input_a4", "input_fp"):
            row = results[method][mode]
            delta = (row["miou"] - control) * 100.0
            row["delta_vs_input_a4_miou_points"] = delta
            csv_rows.append(
                {
                    "method": method,
                    "first_input": mode.removeprefix("input_"),
                    "miou": row["miou"],
                    "delta_vs_input_a4_miou_points": delta,
                    **{
                        f"iou_{name}": row["per_class_iou"][name]
                        for name in common.CLASS_NAMES
                    },
                    "checkpoint_epoch": row["checkpoint"].get("epoch"),
                    "evaluated_batches": row["evaluated_batches"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }
            )

    if limit is None:
        error = abs(results["qad"]["input_a4"]["miou"] - common.HISTORICAL_QAD_MIOU)
        if error > common.HISTORICAL_QAD_TOLERANCE:
            raise RuntimeError(
                "QAD control reproduction failed: "
                f"observed={results['qad']['input_a4']['miou']:.9f}"
            )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "design": "paired inference-only first-image-input quantization ablation",
            "dataset": "UDD val_patches.txt",
            "validation_samples": len(loader.dataset),
            "batch_size": 20,
            "time_steps": 1,
            "first_layer": "init_conv.0.conv",
            "control": "first-layer W4 weight plus A4 normalized-image input",
            "variant": "first-layer W4 weight plus FP32 normalized-image input",
            "later_inputs": "historical A4/integer-bypass policy unchanged",
            "materiality_threshold_miou_points": 0.5,
            "qad_training_epochs_observed": 127,
            "ste_training_budget_epochs": 16,
            "max_validation_batches": limit,
        },
        "results": results,
        "checkpoint_hashes": {
            "qad": sha256_file(common.DEFAULT_QAD_CHECKPOINT),
            "ste": sha256_file(ste_checkpoint),
        },
        "protected_files": validate_protected_hashes(),
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(csv_rows, output_dir / "results.csv")
    write_report(payload, output_dir / "report.md")
    print(json.dumps({m: {k: v["miou"] for k, v in r.items()} for m, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
