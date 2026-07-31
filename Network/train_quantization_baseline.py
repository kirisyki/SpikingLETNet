#!/usr/bin/env python3
"""Train one isolated W4A4 comparison baseline.

This entrypoint never imports the QAD training script and never rewrites an
existing run directory. Production settings come from ``ExperimentProtocol``;
the only reduced settings are enabled by the explicit ``--smoke`` flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
os.chdir(NETWORK_DIR)

from builders.dataset_builder import build_dataset_train  # noqa: E402
from builders.model_builder import build_model  # noqa: E402
from quantization_comparison.checkpointing import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
)
from quantization_comparison.ewgs_hessian import (  # noqa: E402
    update_ewgs_backward_scales,
)
from quantization_comparison.layer_adapter import (  # noqa: E402
    repeat_time,
    temporal_average,
)
from quantization_comparison.model_factory import (  # noqa: E402
    build_quantized_model,
    clamp_quantizer_parameters,
    model_parameters,
    quantized_layer_names,
    quantizer_parameters,
)
from quantization_comparison.protocol import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_FP_CHECKPOINT,
    ExperimentProtocol,
    source_manifest,
    write_json,
)
from quantization_comparison.protocol_amendment import (  # noqa: E402
    amend_checkpoint_metadata,
    is_compatible_epoch_amendment,
)
from spikingjelly.activation_based import functional  # noqa: E402
from utils.losses.loss import CrossEntropyLoss2d  # noqa: E402


METHOD_OUTPUT_NAMES = {
    "ste": "ste_qat_w4a4",
    "lsq": "lsq_qat_w4a4",
    "ewgs": "ewgs_qat_w4a4",
}
CLASS_NAMES = ("background", "facade", "road", "vegetation", "vehicle", "roof")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(METHOD_OUTPUT_NAMES))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run an explicitly reduced integration test, never a reportable result",
    )
    parser.add_argument("--max-train-iters", type=int, default=1)
    parser.add_argument("--max-val-iters", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--ewgs-hessian-batches", type=int, default=10)
    parser.add_argument("--ewgs-hessian-max-iters", type=int, default=50)
    parser.add_argument("--ewgs-hessian-tolerance", type=float, default=1e-3)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    # Match the historical QAD execution rather than silently changing kernels.
    cudnn.benchmark = True
    cudnn.deterministic = False


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.resume is not None:
        resume = args.resume.resolve()
        if not resume.is_file():
            raise FileNotFoundError(resume)
        inferred = resume.parent
        if args.output_dir is not None and args.output_dir.resolve() != inferred:
            raise ValueError("--output-dir must equal the resume checkpoint directory")
        return inferred

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_CHECKPOINT_ROOT / METHOD_OUTPUT_NAMES[args.method]).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_fp_model(protocol: ExperimentProtocol) -> nn.Module:
    model = build_model(
        protocol.model,
        num_classes=protocol.classes,
        config=str(DEFAULT_CONFIG),
    )
    checkpoint = torch.load(DEFAULT_FP_CHECKPOINT, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"missing 'model' in {DEFAULT_FP_CHECKPOINT}")
    model.load_state_dict(checkpoint["model"], strict=True)
    functional.set_step_mode(model, step_mode="m")
    return model


def create_loaders(
    protocol: ExperimentProtocol,
) -> tuple[Iterable[Any], Iterable[Any]]:
    _, train_loader, validation_loader = build_dataset_train(
        protocol.dataset,
        (protocol.input_height, protocol.input_width),
        protocol.train_batch_size,
        protocol.train_type,
        protocol.random_scale,
        protocol.random_mirror,
        protocol.workers,
    )
    if len(train_loader) != protocol.batches_per_epoch:
        raise RuntimeError(
            f"protocol expects {protocol.batches_per_epoch} train batches, "
            f"observed {len(train_loader)}"
        )
    return train_loader, validation_loader


def create_hessian_loader(
    train_loader: Iterable[Any], protocol: ExperimentProtocol
) -> DataLoader[Any]:
    if not isinstance(train_loader, DataLoader):
        raise TypeError("EWGS requires a torch DataLoader")
    return DataLoader(
        train_loader.dataset,
        batch_size=16,
        shuffle=True,
        num_workers=protocol.workers,
        pin_memory=True,
        drop_last=True,
    )


def build_optimizer(
    protocol: ExperimentProtocol, model: nn.Module
) -> torch.optim.Optimizer:
    quantizer_group = quantizer_parameters(protocol.method, model)
    ordinary_group = model_parameters(protocol.method, model)
    groups: list[dict[str, Any]] = [
        {
            "params": ordinary_group,
            "lr": protocol.model_learning_rate,
            "base_lr": protocol.model_learning_rate,
            "weight_decay": protocol.weight_decay,
            "role": "model",
        }
    ]
    if quantizer_group:
        quantizer_base_lr = (
            protocol.ewgs_quantizer_learning_rate
            if protocol.method == "ewgs"
            else protocol.model_learning_rate
        )
        groups.append(
            {
                "params": quantizer_group,
                "lr": quantizer_base_lr,
                "base_lr": quantizer_base_lr,
                "weight_decay": 0.0,
                "role": "quantizer",
            }
        )
    return torch.optim.Adam(
        groups,
        betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps,
    )


def update_learning_rates(
    optimizer: torch.optim.Optimizer,
    protocol: ExperimentProtocol,
    global_iteration: int,
) -> None:
    multiplier = protocol.lr_multiplier(global_iteration)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier


class DeviceConfusionMatrix:
    def __init__(self, classes: int, device: torch.device):
        self.classes = classes
        self.matrix = torch.zeros(
            classes, classes, dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def update(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        labels = labels.flatten().long()
        predictions = predictions.flatten().long()
        valid = (labels >= 0) & (labels < self.classes)
        indices = self.classes * labels[valid] + predictions[valid]
        self.matrix += torch.bincount(
            indices, minlength=self.classes * self.classes
        ).reshape(self.classes, self.classes)

    def result(self) -> tuple[float, list[float]]:
        matrix = self.matrix.double()
        denominator = matrix.sum(0) + matrix.sum(1) - matrix.diag()
        iou = matrix.diag() / denominator
        return float(torch.nanmean(iou).item()), [
            float(value) for value in iou.cpu().tolist()
        ]


def extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"model returned {type(output)!r}, expected Tensor or tuple")
    return temporal_average(output)


def train_one_epoch(
    *,
    model: nn.Module,
    loader: Iterable[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    protocol: ExperimentProtocol,
    epoch: int,
    device: torch.device,
    max_iterations: int | None,
    log_interval: int,
) -> tuple[float, int, float]:
    model.train()
    losses: list[float] = []
    started = time.perf_counter()
    for iteration, batch in enumerate(loader):
        if max_iterations is not None and iteration >= max_iterations:
            break
        images, labels = batch[:2]
        images = repeat_time(
            images.to(device, non_blocking=True), protocol.time_steps
        )
        labels = labels.long().to(device, non_blocking=True)
        global_iteration = epoch * protocol.batches_per_epoch + iteration
        update_learning_rates(optimizer, protocol, global_iteration)

        optimizer.zero_grad(set_to_none=True)
        logits = extract_logits(model.forward_qat(images))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        clamp_quantizer_parameters(model)
        losses.append(float(loss.item()))
        functional.reset_net(model)

        if iteration % max(log_interval, 1) == 0:
            print(
                f"train method={protocol.method} epoch={epoch + 1} "
                f"iter={iteration + 1}/{len(loader)} loss={loss.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )

    if not losses:
        raise RuntimeError("training epoch processed no batches")
    return (
        sum(losses) / len(losses),
        len(losses),
        time.perf_counter() - started,
    )


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    loader: Iterable[Any],
    protocol: ExperimentProtocol,
    device: torch.device,
    max_iterations: int | None,
) -> tuple[float, list[float], int, float]:
    model.eval()
    confusion = DeviceConfusionMatrix(protocol.classes, device)
    started = time.perf_counter()
    processed = 0
    for iteration, batch in enumerate(loader):
        if max_iterations is not None and iteration >= max_iterations:
            break
        images, labels = batch[:2]
        images = repeat_time(
            images.to(device, non_blocking=True), protocol.time_steps
        )
        labels = labels.long().to(device, non_blocking=True)
        logits = extract_logits(model(images))
        confusion.update(labels, logits.argmax(1))
        functional.reset_net(model)
        processed += 1
    if processed == 0:
        raise RuntimeError("validation processed no batches")
    miou, per_class = confusion.result()
    return miou, per_class, processed, time.perf_counter() - started


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "miou",
        *[f"iou_{name}" for name in CLASS_NAMES],
        "model_lr",
        "quantizer_lr",
        "train_batches",
        "validation_batches",
        "train_seconds",
        "validation_seconds",
    ]
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def validate_smoke_args(args: argparse.Namespace) -> None:
    if args.smoke:
        if args.max_train_iters <= 0 or args.max_val_iters <= 0:
            raise ValueError("smoke iteration limits must be positive")
    elif args.max_train_iters != 1 or args.max_val_iters != 1:
        raise ValueError("iteration limits are smoke-only")

    if not args.smoke and (
        args.ewgs_hessian_batches != 10
        or args.ewgs_hessian_max_iters != 50
        or args.ewgs_hessian_tolerance != 1e-3
    ):
        raise ValueError("production EWGS Hessian settings are frozen to 10/50/1e-3")


def main() -> None:
    args = parse_args()
    validate_smoke_args(args)
    protocol = ExperimentProtocol(args.method)
    setup_seed(protocol.seed)
    output_dir = resolve_output_dir(args)
    manifest = source_manifest(protocol)
    manifest["execution"] = {
        "started_at": utc_now(),
        "smoke": args.smoke,
        "amp": False,
        "task_loss_only": True,
        "command_arguments": vars(args),
    }
    manifest["execution"]["command_arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in manifest["execution"]["command_arguments"].items()
    }
    write_json(output_dir / "manifest.json", manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved comparison runs")
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(device)} output={output_dir}", flush=True)

    fp_model = load_fp_model(protocol)
    model = build_quantized_model(
        protocol.method,
        fp_model,
        expected_layer_count=protocol.expected_quantized_layers,
    )
    del fp_model
    functional.set_step_mode(model, step_mode="m")
    model.to(device)
    criterion = CrossEntropyLoss2d().to(device)
    optimizer = build_optimizer(protocol, model)
    train_loader, validation_loader = create_loaders(protocol)
    hessian_loader = (
        create_hessian_loader(train_loader, protocol)
        if protocol.method == "ewgs"
        else None
    )

    start_epoch = 0
    global_iteration = 0
    best_miou = float("-inf")
    if args.resume is not None:
        payload = load_checkpoint(
            args.resume.resolve(),
            model=model,
            optimizer=optimizer,
            restore_rng=True,
            map_location=device,
        )
        current_protocol = protocol.to_dict()
        saved_protocol = payload["protocol"]
        amended_resume = saved_protocol != current_protocol
        if amended_resume and not is_compatible_epoch_amendment(
            saved_protocol, current_protocol
        ):
            raise RuntimeError("resume protocol does not match this run")
        start_epoch = int(payload["epoch"])
        global_iteration = int(payload["global_iteration"])
        best_miou = float(payload["best_miou"])
        if amended_resume:
            manifest["execution"]["protocol_amendment"] = {
                "old_run_epochs": 127,
                "new_run_epochs": protocol.run_epochs,
                "schedule_epochs": protocol.schedule_epochs,
                "reason": "user requested fewer epochs to reduce total training time",
            }
            write_json(output_dir / "manifest.json", manifest)
            for checkpoint_name in ("checkpoint_last.pth", "checkpoint_best.pth"):
                checkpoint_path = output_dir / checkpoint_name
                if checkpoint_path.is_file():
                    amend_checkpoint_metadata(
                        checkpoint_path,
                        protocol=current_protocol,
                        manifest=manifest,
                    )

    run_epochs = 2 if args.smoke and protocol.method == "ewgs" else (1 if args.smoke else protocol.run_epochs)
    max_train_iterations = args.max_train_iters if args.smoke else None
    max_validation_iterations = args.max_val_iters if args.smoke else None
    hessian_records: list[dict[str, Any]] = []
    full_started = time.perf_counter()

    for epoch in range(start_epoch, run_epochs):
        if protocol.method == "ewgs" and epoch >= 1:
            assert hessian_loader is not None
            hessian_started = time.perf_counter()
            update_result = update_ewgs_backward_scales(
                model,
                hessian_loader,
                criterion,
                device,
                time_steps=protocol.time_steps,
                batches=(1 if args.smoke else args.ewgs_hessian_batches),
                max_iterations=(1 if args.smoke else args.ewgs_hessian_max_iters),
                tolerance=args.ewgs_hessian_tolerance,
            )
            hessian_record = {
                "epoch": epoch,
                "seconds": time.perf_counter() - hessian_started,
                **update_result.to_dict(),
            }
            hessian_records.append(hessian_record)
            write_json(
                output_dir / "ewgs_hessian_latest.json",
                hessian_record,
            )

        train_loss, train_batches, train_seconds = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            protocol=protocol,
            epoch=epoch,
            device=device,
            max_iterations=max_train_iterations,
            log_interval=args.log_interval,
        )
        global_iteration = epoch * protocol.batches_per_epoch + train_batches
        miou, per_class, validation_batches, validation_seconds = validate(
            model=model,
            loader=validation_loader,
            protocol=protocol,
            device=device,
            max_iterations=max_validation_iterations,
        )
        model_lr = float(optimizer.param_groups[0]["lr"])
        quantizer_lr = next(
            (
                float(group["lr"])
                for group in optimizer.param_groups
                if group.get("role") == "quantizer"
            ),
            model_lr,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "miou": miou,
            **{
                f"iou_{name}": value
                for name, value in zip(CLASS_NAMES, per_class)
            },
            "model_lr": model_lr,
            "quantizer_lr": quantizer_lr,
            "train_batches": train_batches,
            "validation_batches": validation_batches,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
        }
        append_metrics(output_dir / "metrics.csv", row)

        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "epoch": epoch + 1,
            "global_iteration": global_iteration,
            "best_miou": max(best_miou, miou),
            "protocol": protocol.to_dict(),
            "manifest": manifest,
        }
        save_checkpoint(output_dir / "checkpoint_last.pth", **checkpoint_kwargs)
        if miou >= best_miou:
            best_miou = miou
            save_checkpoint(
                output_dir / "checkpoint_best.pth",
                **{**checkpoint_kwargs, "best_miou": best_miou},
            )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.6f} "
            f"miou={miou:.6f} best={best_miou:.6f}",
            flush=True,
        )

    elapsed = time.perf_counter() - full_started
    summary = {
        "completed_at": utc_now(),
        "method": protocol.method,
        "smoke": args.smoke,
        "epochs_completed": run_epochs,
        "quantized_layer_count": len(quantized_layer_names(model)),
        "best_miou": best_miou,
        "elapsed_seconds": elapsed,
        "hessian_records": hessian_records,
    }
    if args.smoke:
        observed_training_batches = run_epochs * args.max_train_iters
        seconds_per_training_batch = elapsed / max(observed_training_batches, 1)
        summary["naive_training_hours_projection"] = (
            seconds_per_training_batch * protocol.run_iterations / 3600.0
        )
        if hessian_records:
            observed = hessian_records[-1]["seconds"]
            summary["ewgs_hessian_hours_projection"] = (
                observed
                * max(protocol.run_epochs - 1, 0)
                * 10
                * 50
                / 3600.0
            )
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
