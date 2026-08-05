#!/usr/bin/env python3
"""Correctness/performance preflight and 48-hour budget decision."""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

NETWORK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = NETWORK_DIR.parent
sys.path.insert(0, str(NETWORK_DIR))
os.chdir(NETWORK_DIR)

import torch  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from squat_comparison.model_factory import (  # noqa: E402
    build_fp32_lif_model,
    build_squat_from_fp32,
    seed_everything,
)
from squat_comparison.protocol import (  # noqa: E402
    DEFAULT_CONFIG,
    HISTORICAL_COMPARISON,
    NETWORK_DIR as PROTOCOL_NETWORK_DIR,
    OUTPUT_ROOT,
    ExperimentProtocol,
    atomic_write_json,
    frozen_historical_results,
    prepare_new_directory,
    scaled_epoch_budgets,
    sha256_file,
)
from squat_comparison.training import (  # noqa: E402
    build_optimizer,
    create_loaders,
    spike_counts,
)
from utils.losses.loss import CrossEntropyLoss2d  # noqa: E402


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    return result.stdout.strip()


def environment_manifest(protocol: ExperimentProtocol) -> dict[str, Any]:
    tracked_sources = [
        PROTOCOL_NETWORK_DIR / "model/SpikingLETNet_shallow_max.py",
        PROTOCOL_NETWORK_DIR / "model/module/neuron.py",
        DEFAULT_CONFIG,
        HISTORICAL_COMPARISON,
        Path(__file__).resolve(),
    ]
    return {
        "protocol": protocol.to_dict(),
        "historical": frozen_historical_results(protocol),
        "pipeline_policy": [
            "unit_tests",
            "preflight",
            "train_fp32_lif_from_random_init",
            "train_w4m4s1_squat_from_best_fp32_lif",
            "evaluate_new_models_and_read_frozen_qad_json",
        ],
        "prohibited_pipeline_nodes": ["qad_train", "qad_eval", "qad_convert"],
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--short"),
        "source_sha256": {str(path): sha256_file(path) for path in tracked_sources},
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "amp": False,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "data": {
            "train_list": str(Path("/root/autodl-tmp/UDD/UDD6/preprocessed/train_patches.txt")),
            "validation_list": str(Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")),
            "train_samples": 25700,
            "validation_samples": 8478,
            "augmentation": ["HorizontalFlip", "VerticalFlip", "RandomRotate90", "Resize(400,400)"],
        },
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_training_groups(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    protocol: ExperimentProtocol,
    learning_rate: float,
    device: torch.device,
    warmup_groups: int = 1,
    measured_groups: int = 2,
) -> tuple[list[float], int]:
    model.train()
    optimizer = build_optimizer(model, protocol, learning_rate)
    criterion = CrossEntropyLoss2d().to(device)
    iterator = iter(loader)
    microbatches_per_group = protocol.effective_train_batch // loader.batch_size
    timings: list[float] = []
    for group_index in range(warmup_groups + measured_groups):
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        group_samples = 0
        for _ in range(microbatches_per_group):
            images, labels = next(iterator)[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            functional.reset_net(model)
            loss = criterion(spike_counts(model, images, protocol.time_steps), labels)
            batch_samples = int(images.shape[0])
            (loss * (batch_samples / protocol.effective_train_batch)).backward()
            group_samples += batch_samples
            functional.reset_net(model)
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        if group_samples != protocol.effective_train_batch:
            raise RuntimeError("preflight did not benchmark a complete effective batch")
        if group_index >= warmup_groups:
            timings.append(elapsed)
    return timings, torch.cuda.max_memory_allocated(device)


@torch.no_grad()
def benchmark_validation_batches(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    protocol: ExperimentProtocol,
    device: torch.device,
    warmup_batches: int = 2,
    measured_batches: int = 5,
) -> list[float]:
    model.eval()
    iterator = iter(loader)
    timings: list[float] = []
    for batch_index in range(warmup_batches + measured_batches):
        images = next(iterator)[0].to(device, non_blocking=True)
        functional.reset_net(model)
        synchronize(device)
        started = time.perf_counter()
        spike_counts(model, images, protocol.time_steps)
        synchronize(device)
        elapsed = time.perf_counter() - started
        functional.reset_net(model)
        if batch_index >= warmup_batches:
            timings.append(elapsed)
    return timings


def benchmark_stage(
    *,
    model: torch.nn.Module,
    protocol: ExperimentProtocol,
    learning_rate: float,
    device: torch.device,
    batch_candidates: list[int],
) -> dict[str, Any]:
    last_error = ""
    for physical_batch in batch_candidates:
        try:
            seed_everything(protocol.seed)
            generator = torch.Generator().manual_seed(protocol.seed)
            train_loader, validation_loader = create_loaders(
                protocol,
                physical_batch=physical_batch,
                generator=generator,
            )
            model.to(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            train_times, peak = benchmark_training_groups(
                model=model,
                loader=train_loader,
                protocol=protocol,
                learning_rate=learning_rate,
                device=device,
            )
            validation_times = benchmark_validation_batches(
                model=model,
                loader=validation_loader,
                protocol=protocol,
                device=device,
            )
            train_median = statistics.median(train_times)
            validation_median = statistics.median(validation_times)
            updates = math.ceil(len(train_loader.dataset) / protocol.effective_train_batch)
            validation_batches = math.ceil(
                len(validation_loader.dataset) / protocol.validation_batch
            )
            return {
                "physical_batch": physical_batch,
                "gradient_accumulation_steps": protocol.effective_train_batch // physical_batch,
                "warmup_training_groups": 1,
                "measured_training_groups": len(train_times),
                "training_group_seconds": train_times,
                "median_training_group_seconds": train_median,
                "warmup_validation_batches": 2,
                "measured_validation_batches": len(validation_times),
                "validation_batch_seconds": validation_times,
                "median_validation_batch_seconds": validation_median,
                "optimizer_updates_per_epoch": updates,
                "validation_batches_per_epoch": validation_batches,
                "estimated_train_seconds_per_epoch": train_median * updates,
                "estimated_validation_seconds_per_epoch": validation_median * validation_batches,
                "estimated_total_seconds_per_epoch": (
                    train_median * updates + validation_median * validation_batches
                ),
                "peak_memory_bytes": peak,
            }
        except torch.cuda.OutOfMemoryError as error:
            last_error = str(error)
            print(f"OOM physical_batch={physical_batch}; trying smaller divisor", flush=True)
            functional.reset_net(model)
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"no physical batch fit GPU; last OOM: {last_error}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved experiment")
    protocol = ExperimentProtocol()
    output_root = prepare_new_directory(OUTPUT_ROOT)
    manifest = environment_manifest(protocol)
    atomic_write_json(output_root / "manifest.json", manifest)
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(protocol.seed)

    fp_model, fp_audit = build_fp32_lif_model(
        config=DEFAULT_CONFIG, classes=protocol.classes, seed=protocol.seed
    )
    fp_result = benchmark_stage(
        model=fp_model,
        protocol=protocol,
        learning_rate=protocol.fp32_learning_rate,
        device=device,
        batch_candidates=[64, 32, 16, 8, 4, 2, 1],
    )
    fp_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    squat_model, squat_audit = build_squat_from_fp32(
        fp_model, expected_weight_layers=protocol.expected_weight_layers
    )
    candidates = [value for value in [64, 32, 16, 8, 4, 2, 1] if value <= fp_result["physical_batch"]]
    squat_result = benchmark_stage(
        model=squat_model,
        protocol=protocol,
        learning_rate=protocol.squat_learning_rate,
        device=device,
        batch_candidates=candidates,
    )

    full_seconds = (
        protocol.fp32_max_epochs * fp_result["estimated_total_seconds_per_epoch"]
        + protocol.squat_max_epochs * squat_result["estimated_total_seconds_per_epoch"]
    )
    full_hours = full_seconds / 3600.0
    fp_epochs, squat_epochs, scale = scaled_epoch_budgets(full_hours, protocol)
    selected_hours = (
        fp_epochs * fp_result["estimated_total_seconds_per_epoch"]
        + squat_epochs * squat_result["estimated_total_seconds_per_epoch"]
    ) / 3600.0
    minimum_hours = (
        protocol.fp32_min_epochs * fp_result["estimated_total_seconds_per_epoch"]
        + protocol.squat_min_epochs * squat_result["estimated_total_seconds_per_epoch"]
    ) / 3600.0
    allowed = minimum_hours < 48.0
    result = {
        "reportable": False,
        "purpose": "correctness, memory, and wall-clock projection only",
        "fp32": fp_result,
        "squat": squat_result,
        "factory_audit": {
            "fp32": fp_audit.__dict__,
            "squat": squat_audit.__dict__,
        },
        "full_budget": {
            "fp32_epochs": protocol.fp32_max_epochs,
            "squat_epochs": protocol.squat_max_epochs,
            "estimated_hours": full_hours,
        },
        "budget_rule": {
            "triggered": full_hours >= 48.0,
            "scale_44h_over_full": scale,
            "selected_fp32_epochs": fp_epochs,
            "selected_squat_epochs": squat_epochs,
            "selected_estimated_hours": selected_hours,
            "minimum_100_plus_40_estimated_hours": minimum_hours,
            "formal_training_allowed": allowed,
            "stop_reason": (
                None
                if allowed
                else "minimum 100 FP32 + 40 SQUAT epochs is estimated >=48h"
            ),
        },
    }
    atomic_write_json(output_root / "preflight.json", result)
    print(json.dumps(result, indent=2, default=list), flush=True)
    if not allowed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

