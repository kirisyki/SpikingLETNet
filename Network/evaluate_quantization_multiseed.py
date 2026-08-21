#!/usr/bin/env python3
"""Evaluate one new W4A4 training seed on the complete UDD validation set."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
os.chdir(NETWORK_DIR)

import evaluate_quantization_comparison as historical_eval  # noqa: E402
from builders.dataset_builder import build_dataset_test  # noqa: E402
from quantization.int4_selfbuild import quantize_model  # noqa: E402
from quantization_comparison.checkpointing import CHECKPOINT_FORMAT  # noqa: E402
from quantization_comparison.model_factory import (  # noqa: E402
    build_quantized_model,
    quantized_layer_names,
)
from quantization_comparison.protocol import (  # noqa: E402
    ExperimentProtocol,
    sha256_file,
    validate_protected_hashes,
    write_json,
)
from spikingjelly.activation_based import functional  # noqa: E402
from train_qad_multiseed import protocol_dict as qad_protocol_dict  # noqa: E402


APPROVED_SEEDS = (2345, 3456)
SMOKE_SEEDS = (1234, *APPROVED_SEEDS)
METHODS = ("qad", "ste", "lsq", "ewgs")
METHOD_DIRS = {
    "qad": "qad_qat_w4a4",
    "ste": "ste_qat_w4a4",
    "lsq": "lsq_qat_w4a4",
    "ewgs": "ewgs_qat_w4a4",
}
CLASS_NAMES = historical_eval.CLASS_NAMES
EXPECTED_VALIDATION_SAMPLES = historical_eval.EXPECTED_VALIDATION_SAMPLES


def checkpoint_root(seed: int) -> Path:
    return (
        REPO_ROOT
        / "quantization_multiseed_checkpoint/udd/w4a4_v1"
        / f"seed{seed}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SMOKE_SEEDS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qad-checkpoint", type=Path)
    parser.add_argument("--ste-checkpoint", type=Path)
    parser.add_argument("--lsq-checkpoint", type=Path)
    parser.add_argument("--ewgs-checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-val-iters", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_checkpoints(args: argparse.Namespace) -> dict[str, Path]:
    root = checkpoint_root(args.seed)
    return {
        method: (
            getattr(args, f"{method}_checkpoint")
            or root / METHOD_DIRS[method] / "checkpoint_best.pth"
        ).resolve()
        for method in METHODS
    }


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid multi-seed checkpoint: {path}")
    return payload


def expected_protocol(
    method: str, seed: int, *, smoke: bool = False
) -> dict[str, Any]:
    if method == "qad":
        return qad_protocol_dict(seed, smoke=smoke)
    return ExperimentProtocol(method, seed=seed).to_dict()


def build_model(
    method: str,
    seed: int,
    path: Path,
    *,
    smoke: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_payload(path)
    expected = expected_protocol(method, seed, smoke=smoke)
    if payload.get("protocol") != expected:
        raise ValueError(f"protocol mismatch for {method} seed {seed}: {path}")

    if method == "qad":
        model = quantize_model(
            historical_eval.base_model(),
            k=4,
            inplace=False,
            quant=True,
            activation_quant=True,
            quant_start_layer=0,
            activation_quant_mode="per_tensor",
        )
    else:
        model = build_quantized_model(method, historical_eval.base_model())
    model.load_state_dict(payload["model_state"], strict=True)
    functional.set_step_mode(model, step_mode="m")
    if len(quantized_layer_names(model)) != 71:
        raise RuntimeError(f"{method} seed {seed} does not contain 71 quantized layers")
    return model, {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": int(payload["epoch"]),
        "best_miou_recorded": float(payload["best_miou"]),
        "protocol": payload["protocol"],
    }


def write_csv(results: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "miou",
        *[f"iou_{name}" for name in CLASS_NAMES],
        "evaluated_batches",
        "elapsed_seconds",
        "best_epoch",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            result = results[method]
            writer.writerow(
                {
                    "method": method,
                    "miou": result["miou"],
                    **{
                        f"iou_{name}": result["per_class_iou"][name]
                        for name in CLASS_NAMES
                    },
                    "evaluated_batches": result["evaluated_batches"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "best_epoch": result["checkpoint"]["epoch"],
                    "checkpoint": result["checkpoint"]["path"],
                }
            )


def write_markdown(seed: int, results: dict[str, dict[str, Any]], path: Path) -> None:
    budgets = {"qad": 127, "ste": 16, "lsq": 16, "ewgs": 16}
    labels = {"qad": "QAD", "ste": "STE-QAT", "lsq": "LSQ-QAT", "ewgs": "EWGS-QAT"}
    lines = [
        f"# W4A4 multi-seed验证结果：seed {seed}",
        "",
        "全量 UDD validation：8,478条目、424 batches、400×400、T=1。",
        "QAD与基线训练预算不同，因此这里只报告路线差异。",
        "",
        "| Method | Epochs | Best epoch | mIoU | " + " | ".join(CLASS_NAMES) + " |",
        "|---|---:|---:|---:|" + "|".join(["---:"] * len(CLASS_NAMES)) + "|",
    ]
    for method in METHODS:
        result = results[method]
        classes = " | ".join(
            f"{result['per_class_iou'][name]:.6f}" for name in CLASS_NAMES
        )
        lines.append(
            f"| {labels[method]} | {budgets[method]} | "
            f"{result['checkpoint']['epoch']} | {result['miou']:.6f} | {classes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        if args.max_val_iters <= 0:
            raise ValueError("--max-val-iters must be positive in smoke mode")
    else:
        if args.seed not in APPROVED_SEEDS:
            raise ValueError(f"formal seed must be one of {APPROVED_SEEDS}")
        if args.max_val_iters != 1:
            raise ValueError("--max-val-iters is smoke-only")
    validate_protected_hashes()
    paths = resolve_checkpoints(args)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    _, loader = build_dataset_test("udd", num_workers=6, none_gt=False, batch_size=20)
    if len(loader.dataset) != EXPECTED_VALIDATION_SAMPLES or len(loader) != 424:
        raise RuntimeError(
            f"unexpected validation cohort: samples={len(loader.dataset)} batches={len(loader)}"
        )

    results: dict[str, dict[str, Any]] = {}
    limit = args.max_val_iters if args.smoke else None
    for method in METHODS:
        print(f"evaluating method={method} seed={args.seed}", flush=True)
        model, checkpoint = build_model(
            method, args.seed, paths[method], smoke=args.smoke
        )
        metrics = historical_eval.evaluate_model(
            model=model,
            loader=loader,
            device=device,
            limit=limit,
            audit_bypass=method in ("qad", "ste"),
        )
        expected_batches = limit or 424
        if metrics["evaluated_batches"] != expected_batches:
            raise RuntimeError(f"incomplete validation for {method}")
        results[method] = {**metrics, "checkpoint": checkpoint}
        del model
        torch.cuda.empty_cache()

    payload = {
        "protocol": {
            "seed": args.seed,
            "dataset": "UDD val_patches.txt",
            "validation_samples": EXPECTED_VALIDATION_SAMPLES,
            "validation_batches": 424,
            "input_size": [400, 400],
            "time_steps": 1,
            "batch_size": 20,
            "training_epochs": {"qad": 127, "ste": 16, "lsq": 16, "ewgs": 16},
            "smoke": args.smoke,
            "evaluated_batches_limit": limit,
        },
        "results": results,
        "protected_files": validate_protected_hashes(),
    }
    write_json(output_dir / "comparison.json", payload)
    write_csv(results, output_dir / "comparison.csv")
    write_markdown(args.seed, results, output_dir / "report.md")


if __name__ == "__main__":
    main()
