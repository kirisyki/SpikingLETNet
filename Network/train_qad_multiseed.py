#!/usr/bin/env python3
"""Run one isolated, resumable QAD W4A4 training seed.

The algorithmic forward/backward and validation functions are imported from
the protected historical QAD implementation.  This entrypoint only adds an
explicit seed, a 127-epoch stop separate from the 150-epoch LR horizon,
non-overwriting output paths, manifests, and exact state/RNG checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
os.chdir(NETWORK_DIR)

import QAT_snn_STE as qad  # noqa: E402
from quantization_comparison.checkpointing import (  # noqa: E402
    atomic_torch_save,
    load_checkpoint,
    save_checkpoint,
)
from quantization_comparison.protocol import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_FP_CHECKPOINT,
    sha256_file,
    validate_protected_hashes,
    write_json,
)


APPROVED_FORMAL_SEEDS = (2345, 3456)
QAD_RUN_EPOCHS = 127
SCHEDULE_EPOCHS = 150
EXPECTED_TRAIN_BATCHES = 401
EXPECTED_QUANTIZED_LAYERS = 71
CLASS_NAMES = ("background", "facade", "road", "vegetation", "vehicle", "roof")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-iters", type=int, default=1)
    parser.add_argument("--max-val-iters", type=int, default=1)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.smoke and args.seed not in APPROVED_FORMAL_SEEDS:
        raise ValueError(
            f"formal seed must be one of {APPROVED_FORMAL_SEEDS}, got {args.seed}"
        )
    if args.smoke:
        if args.max_train_iters <= 0 or args.max_val_iters <= 0:
            raise ValueError("smoke iteration limits must be positive")
    elif args.max_train_iters != 1 or args.max_val_iters != 1:
        raise ValueError("iteration limits are smoke-only")


def protocol_dict(seed: int, *, smoke: bool) -> dict[str, Any]:
    return {
        "protocol_version": "w4a4-multiseed-qad-v1",
        "method": "qad",
        "model": "SpikingLETNet_shallow_max",
        "dataset": "udd",
        "classes": 6,
        "input_height": 400,
        "input_width": 400,
        "train_type": "trainval",
        "seed": seed,
        "time_steps": 1,
        "workers": 6,
        "train_batch_size": 64,
        "validation_batch_size": 20,
        "run_epochs": QAD_RUN_EPOCHS,
        "schedule_epochs": SCHEDULE_EPOCHS,
        "batches_per_epoch": EXPECTED_TRAIN_BATCHES,
        "model_learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "poly_exponent": 0.9,
        "weight_bits": 4,
        "activation_bits": 4,
        "quantization_granularity": "per_tensor",
        "expected_quantized_layers": EXPECTED_QUANTIZED_LAYERS,
        "first_last_layer_exemption": False,
        "kd_weight": 0.1,
        "task_loss": "CrossEntropyLoss2d",
        "feature_distillation": True,
        "smoke": smoke,
    }


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if args.resume is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    resume = args.resume.resolve()
    if not resume.is_file():
        raise FileNotFoundError(resume)
    if resume.parent != output_dir:
        raise ValueError("--resume must be inside --output-dir")
    return output_dir


def training_namespace(args: argparse.Namespace, *, smoke: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model="SpikingLETNet_shallow_max",
        dataset="udd",
        input_size="400,400",
        num_workers=6,
        classes=6,
        train_type="trainval",
        # Historical QAD used a 150-epoch schedule and stopped after epoch 126.
        max_epochs=SCHEDULE_EPOCHS,
        random_mirror=True,
        random_scale=True,
        lr=1e-3,
        batch_size=64,
        optim="adam",
        lr_schedule="poly",
        num_cycles=1,
        poly_exp=0.9,
        warmup_iters=500,
        warmup_factor=1.0 / 3.0,
        use_label_smoothing=False,
        use_ohem=False,
        use_lovaszsoftmax=False,
        use_focal=False,
        quant_bits=4,
        activation_quant=True,
        activation_quant_mode="per_tensor",
        quant_start_layer=0,
        kd_weight=0.1,
        max_train_iters_per_epoch=args.max_train_iters if smoke else 0,
        max_val_iters=args.max_val_iters if smoke else 0,
        cuda=True,
        gpus="0",
        resume=str(args.resume or ""),
        T=1,
        config=str(DEFAULT_CONFIG),
        checkpoint=str(DEFAULT_FP_CHECKPOINT),
    )


def source_manifest(protocol: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = torch.cuda.get_device_properties(0)
    return {
        "protocol": protocol,
        "sources": {
            "config": {
                "path": str(DEFAULT_CONFIG.resolve()),
                "sha256": sha256_file(DEFAULT_CONFIG),
            },
            "fp_checkpoint": {
                "path": str(DEFAULT_FP_CHECKPOINT.resolve()),
                "sha256": sha256_file(DEFAULT_FP_CHECKPOINT),
            },
        },
        "protected_files": validate_protected_hashes(),
        "execution": {
            "started_at": utc_now(),
            "smoke": args.smoke,
            "amp": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0),
            "device_total_memory": int(device.total_memory),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_benchmark": True,
            "cudnn_deterministic": True,
            "command_arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
    }


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "miou",
        "model_lr",
        "train_batches",
        "validation_batches",
        "train_seconds",
        "validation_seconds",
    ]
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def count_quantized_layers(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, qad.QLayer))


def build_training_state(
    args: argparse.Namespace,
) -> tuple[
    SimpleNamespace,
    nn.Module,
    nn.Module,
    nn.Module,
    torch.optim.Optimizer,
    Any,
    Any,
]:
    train_args = training_namespace(args, smoke=args.smoke)
    qad.setup_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    model = qad.build_model(
        train_args.model,
        num_classes=train_args.classes,
        config=train_args.config,
    )
    qad.init_weight(
        model,
        nn.init.kaiming_normal_,
        nn.BatchNorm2d,
        1e-3,
        0.1,
        mode="fan_in",
    )
    fp_payload = torch.load(DEFAULT_FP_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(fp_payload["model"], strict=True)
    qad.functional.set_step_mode(model, step_mode="m")

    model_q = qad.quantize_model(
        model,
        k=4,
        inplace=False,
        quant=True,
        activation_quant=True,
        quant_start_layer=0,
        activation_quant_mode="per_tensor",
    )
    qad.functional.set_step_mode(model_q, step_mode="m")
    qad.freeze_model(model)
    if count_quantized_layers(model_q) != EXPECTED_QUANTIZED_LAYERS:
        raise RuntimeError("QAD model does not contain 71 quantized layers")

    _, train_loader, validation_loader = qad.build_dataset_train(
        "udd", (400, 400), 64, "trainval", True, True, 6
    )
    if len(train_loader) != EXPECTED_TRAIN_BATCHES:
        raise RuntimeError(
            f"expected {EXPECTED_TRAIN_BATCHES} train batches, got {len(train_loader)}"
        )
    train_args.per_iter = len(train_loader)
    train_args.max_iter = SCHEDULE_EPOCHS * len(train_loader)

    device = torch.device("cuda")
    model.to(device)
    model_q.to(device)
    criterion = qad.CrossEntropyLoss2d().to(device)
    optimizer = qad.build_optimizer(train_args, model_q)
    return (
        train_args,
        model,
        model_q,
        criterion,
        optimizer,
        train_loader,
        validation_loader,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    output_dir = resolve_output_dir(args)
    protocol = protocol_dict(args.seed, smoke=args.smoke)

    (
        train_args,
        teacher,
        student,
        criterion,
        optimizer,
        train_loader,
        validation_loader,
    ) = build_training_state(args)

    manifest_path = output_dir / "manifest.json"
    if args.resume is None:
        manifest = source_manifest(protocol, args)
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("execution", {}).setdefault("resume_events", []).append(
            {"at": utc_now(), "checkpoint": str(args.resume.resolve())}
        )
        write_json(manifest_path, manifest)

    start_epoch = 0
    best_miou = float("-inf")
    global_iteration = 0
    if args.resume is not None:
        payload = load_checkpoint(
            args.resume.resolve(),
            model=student,
            optimizer=optimizer,
            restore_rng=True,
            map_location="cuda",
        )
        if payload["protocol"] != protocol:
            raise RuntimeError("resume protocol does not match this QAD run")
        start_epoch = int(payload["epoch"])
        global_iteration = int(payload["global_iteration"])
        best_miou = float(payload["best_miou"])

    run_epochs = 1 if args.smoke else QAD_RUN_EPOCHS
    full_started = time.perf_counter()
    for epoch in range(start_epoch, run_epochs):
        train_started = time.perf_counter()
        train_loss, model_lr = qad.train_qat(
            train_args,
            train_loader,
            teacher,
            student,
            criterion,
            optimizer,
            epoch,
        )
        train_seconds = time.perf_counter() - train_started
        train_batches = args.max_train_iters if args.smoke else len(train_loader)
        global_iteration = epoch * EXPECTED_TRAIN_BATCHES + train_batches

        validation_started = time.perf_counter()
        miou, _ = qad.val(train_args, validation_loader, student)
        validation_seconds = time.perf_counter() - validation_started
        validation_batches = args.max_val_iters if args.smoke else len(validation_loader)

        append_metrics(
            output_dir / "metrics.csv",
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "miou": miou,
                "model_lr": model_lr,
                "train_batches": train_batches,
                "validation_batches": validation_batches,
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
            },
        )

        improved = miou >= best_miou
        best_miou = max(best_miou, miou)
        checkpoint_kwargs = {
            "model": student,
            "optimizer": optimizer,
            "epoch": epoch + 1,
            "global_iteration": global_iteration,
            "best_miou": best_miou,
            "protocol": protocol,
            "manifest": manifest,
        }
        save_checkpoint(output_dir / "checkpoint_last.pth", **checkpoint_kwargs)
        if improved:
            save_checkpoint(output_dir / "checkpoint_best.pth", **checkpoint_kwargs)
            atomic_torch_save(
                {"epoch": epoch + 1, "model": student.state_dict()},
                output_dir / "model_q_best.pth",
            )
        print(
            f"qad seed={args.seed} epoch={epoch + 1}/{run_epochs} "
            f"loss={train_loss:.6f} miou={miou:.6f} best={best_miou:.6f}",
            flush=True,
        )

    summary = {
        "completed_at": utc_now(),
        "method": "qad",
        "seed": args.seed,
        "smoke": args.smoke,
        "epochs_completed": run_epochs,
        "schedule_epochs": SCHEDULE_EPOCHS,
        "quantized_layer_count": count_quantized_layers(student),
        "best_miou": best_miou,
        "elapsed_seconds": time.perf_counter() - full_started,
        "global_iteration": global_iteration,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
