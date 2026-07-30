"""Unified FP32, W4/A4, W1.58/A4 and W1.58/A1.58 evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from builders.dataset_builder import build_dataset_test
from model.SpikingLETNet_shallow_max import SpikingLETNet_shallow_max
from quantization.int4_selfbuild import quantize_model
from quantization.ternary_qat import (
    quantize_ternary_model,
    ternary_model_stats,
)
from spikingjelly.activation_based import functional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_FP = (
    REPO_ROOT
    / "checkpoint/udd/"
    "SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth"
)
DEFAULT_W4 = (
    REPO_ROOT
    / "QAT_checkpoint/udd/"
    "SpikingLETNet_shallow_maxbs64gpu1_trainval20260612-135249/"
    "model_q_best.pth"
)
DEFAULT_TERNARY_A4 = (
    REPO_ROOT
    / "ternary_QAT_checkpoint/udd/"
    "SpikingLETNet_shallow_max_w1p58_a4_bs64_seed1234/model_q_best.pth"
)
DEFAULT_TERNARY_A1P58 = (
    REPO_ROOT
    / "ternary_QAT_checkpoint/udd/"
    "SpikingLETNet_shallow_max_w1p58_a1p58_bs64_seed1234/model_q_best.pth"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "ternary_QAT_checkpoint/comparison_seed1234"
)
CLASS_NAMES = [
    "background",
    "facade",
    "road",
    "vegetation",
    "vehicle",
    "roof",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fp-checkpoint", type=Path, default=DEFAULT_FP)
    parser.add_argument("--w4-checkpoint", type=Path, default=DEFAULT_W4)
    parser.add_argument(
        "--ternary-a4-checkpoint", type=Path, default=DEFAULT_TERNARY_A4
    )
    parser.add_argument(
        "--ternary-a1p58-checkpoint",
        type=Path,
        default=DEFAULT_TERNARY_A1P58,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-val-iters", type=int, default=0)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def temporal_average(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() in (3, 5):
        return tensor.mean(0)
    return tensor


def repeat_time(images: torch.Tensor, time_steps: int) -> torch.Tensor:
    return images.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)


class ConfusionMatrix:
    def __init__(self, classes: int, device: torch.device):
        self.classes = classes
        self.matrix = torch.zeros(
            (classes, classes), dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def update(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        valid = (target >= 0) & (target < self.classes)
        indices = (
            self.classes * target[valid].to(torch.int64)
            + prediction[valid].to(torch.int64)
        )
        self.matrix += torch.bincount(
            indices, minlength=self.classes**2
        ).reshape(self.classes, self.classes)

    def compute(self) -> Tuple[float, List[float]]:
        matrix = self.matrix.float()
        union = matrix.sum(1) + matrix.sum(0) - torch.diag(matrix)
        iou = torch.diag(matrix) / union
        return float(torch.nanmean(iou).item()), [
            float(value) for value in iou.cpu().tolist()
        ]


def base_model(config: Path) -> nn.Module:
    model = SpikingLETNet_shallow_max(classes=6, config=str(config))
    functional.set_step_mode(model, step_mode="m")
    return model


def checkpoint_state(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint lacks model state: {path}")
    metadata = {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "best_miou_recorded": checkpoint.get("best_miou"),
        "quantization_metadata": checkpoint.get("metadata"),
    }
    return checkpoint["model"], metadata


def build_fp(config: Path, path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
    model = base_model(config)
    state, metadata = checkpoint_state(path)
    model.load_state_dict(state, strict=True)
    return model, metadata


def build_w4(config: Path, path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
    model = base_model(config)
    model = quantize_model(
        model,
        k=4,
        inplace=False,
        quant=True,
        activation_quant=True,
        quant_start_layer=0,
        activation_quant_mode="per_tensor",
    )
    functional.set_step_mode(model, step_mode="m")
    state, metadata = checkpoint_state(path)
    model.load_state_dict(state, strict=True)
    return model, metadata


def build_ternary(
    config: Path, path: Path, activation_mode: str
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = quantize_ternary_model(
        base_model(config), activation_mode=activation_mode, inplace=False
    )
    functional.set_step_mode(model, step_mode="m")
    state, metadata = checkpoint_state(path)
    model.load_state_dict(state, strict=True)
    declared = (metadata.get("quantization_metadata") or {}).get(
        "activation_mode"
    )
    if declared is not None and declared != activation_mode:
        raise ValueError(
            f"{path} declares activation_mode={declared!r}, "
            f"expected {activation_mode!r}"
        )
    return model, metadata


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    time_steps: int,
    max_val_iters: int,
) -> Dict[str, Any]:
    model = model.to(device).eval()
    confusion = ConfusionMatrix(6, device)
    total_batches = len(loader)  # type: ignore[arg-type]
    limit = (
        min(total_batches, max_val_iters)
        if max_val_iters > 0
        else total_batches
    )
    started = time.monotonic()
    completed = 0
    for iteration, (images, labels) in enumerate(loader):
        if iteration >= limit:
            break
        images = repeat_time(images.to(device, non_blocking=True), time_steps)
        labels = labels.long().to(device, non_blocking=True)
        output = temporal_average(model(images))
        confusion.update(labels.flatten(), output.argmax(1).flatten())
        functional.reset_net(model)
        completed += 1
        if iteration == 0 or (iteration + 1) % 20 == 0 or iteration + 1 == limit:
            print(
                f"eval iter={iteration + 1}/{limit} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if completed == 0:
        raise RuntimeError("No validation batches evaluated")
    miou, per_class = confusion.compute()
    return {
        "miou": miou,
        "per_class_iou": per_class,
        "evaluated_batches": completed,
        "elapsed_seconds": time.monotonic() - started,
    }


def write_csv(results: Dict[str, Dict[str, Any]], path: Path) -> None:
    fields = [
        "model",
        "weight_bits",
        "activation_bits",
        "miou",
        *[f"iou_{name}" for name in CLASS_NAMES],
        "weight_zero_fraction",
        "checkpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in results.items():
            row = {
                "model": name,
                "weight_bits": result["weight_bits"],
                "activation_bits": result["activation_bits"],
                "miou": result["miou"],
                "weight_zero_fraction": result.get("weight_zero_fraction"),
                "checkpoint": result["checkpoint"]["path"],
            }
            row.update(
                {
                    f"iou_{class_name}": value
                    for class_name, value in zip(
                        CLASS_NAMES, result["per_class_iou"]
                    )
                }
            )
            writer.writerow(row)


def conclusion(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    gap_pp = (
        results["W4/A4"]["miou"] - results["W1.58/A4"]["miou"]
    ) * 100.0
    if gap_pp >= 2.0:
        verdict = "supports_w4_accuracy_value"
        explanation = (
            "W4/A4 exceeds W1.58/A4 by at least 2 absolute mIoU points."
        )
    elif gap_pp <= 0.0:
        verdict = "does_not_support_w4_accuracy_value"
        explanation = "W1.58/A4 matches or exceeds W4/A4."
    else:
        verdict = "ambiguous_add_two_seeds"
        explanation = (
            "The W4/A4 advantage is below 2 absolute mIoU points; "
            "run the two pre-agreed additional seeds."
        )
    return {
        "primary_gap_miou_points_w4_minus_w1p58_a4": gap_pp,
        "threshold_miou_points": 2.0,
        "verdict": verdict,
        "explanation": explanation,
        "secondary_activation_penalty_miou_points": (
            results["W1.58/A4"]["miou"]
            - results["W1.58/A1.58"]["miou"]
        )
        * 100.0,
    }


def write_markdown(
    results: Dict[str, Dict[str, Any]],
    decision: Dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# SpikingLETNet shallow max precision comparison",
        "",
        "All models were re-evaluated with the same UDD validation loader, "
        "preprocessing, T, batch size, and confusion-matrix implementation.",
        "",
        "| Model | Weight bits | Activation bits | mIoU | Weight zero fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        zero = result.get("weight_zero_fraction")
        zero_text = f"{zero:.4%}" if zero is not None else "n/a"
        lines.append(
            f"| {name} | {result['weight_bits']:.4g} | "
            f"{result['activation_bits']:.4g} | {result['miou']:.6f} | "
            f"{zero_text} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered decision",
            "",
            f"- W4/A4 minus W1.58/A4: "
            f"{decision['primary_gap_miou_points_w4_minus_w1p58_a4']:.4f} "
            "mIoU points.",
            f"- Threshold: {decision['threshold_miou_points']:.1f} points.",
            f"- Verdict: `{decision['verdict']}`.",
            f"- Interpretation: {decision['explanation']}",
            f"- W1.58 activation penalty: "
            f"{decision['secondary_activation_penalty_miou_points']:.4f} "
            "mIoU points.",
            "",
            "## Per-class IoU",
            "",
            "| Model | " + " | ".join(CLASS_NAMES) + " |",
            "|---|" + "|".join(["---:"] * len(CLASS_NAMES)) + "|",
        ]
    )
    for name, result in results.items():
        values = " | ".join(
            f"{value:.6f}" for value in result["per_class_iou"]
        )
        lines.append(f"| {name} | {values} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if args.T <= 0:
        raise ValueError("--T must be positive")
    paths = [
        args.config,
        args.fp_checkpoint,
        args.w4_checkpoint,
        args.ternary_a4_checkpoint,
        args.ternary_a1p58_checkpoint,
    ]
    paths = [path.resolve() for path in paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    (
        args.config,
        args.fp_checkpoint,
        args.w4_checkpoint,
        args.ternary_a4_checkpoint,
        args.ternary_a1p58_checkpoint,
    ) = paths

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    _, loader = build_dataset_test(
        "udd",
        args.num_workers,
        none_gt=False,
        batch_size=args.batch_size,
    )

    builders = [
        ("FP32", lambda: build_fp(args.config, args.fp_checkpoint), 32.0, 32.0),
        ("W4/A4", lambda: build_w4(args.config, args.w4_checkpoint), 4.0, 4.0),
        (
            "W1.58/A4",
            lambda: build_ternary(
                args.config, args.ternary_a4_checkpoint, "a4"
            ),
            math.log2(3.0),
            4.0,
        ),
        (
            "W1.58/A1.58",
            lambda: build_ternary(
                args.config, args.ternary_a1p58_checkpoint, "ternary"
            ),
            math.log2(3.0),
            math.log2(3.0),
        ),
    ]
    results: Dict[str, Dict[str, Any]] = {}
    for name, builder, weight_bits, activation_bits in builders:
        print(f"evaluating {name}", flush=True)
        model, checkpoint = builder()
        metrics = evaluate_model(
            model, loader, device, args.T, args.max_val_iters
        )
        result = {
            **metrics,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "checkpoint": checkpoint,
        }
        if name.startswith("W1.58"):
            result["weight_zero_fraction"] = ternary_model_stats(
                model
            ).zero_fraction
        results[name] = result
        del model
        torch.cuda.empty_cache()

    decision = conclusion(results)
    payload = {
        "protocol": {
            "dataset": "UDD val_patches.txt",
            "classes": CLASS_NAMES,
            "T": args.T,
            "batch_size": args.batch_size,
            "max_val_iters": args.max_val_iters,
        },
        "results": results,
        "decision": decision,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(results, output_dir / "comparison.csv")
    write_markdown(results, decision, output_dir / "report.md")
    print(json.dumps(decision, indent=2), flush=True)
    return output_dir


if __name__ == "__main__":
    run(parse_args())
