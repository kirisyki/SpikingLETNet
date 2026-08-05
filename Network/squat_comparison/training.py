"""Shared deterministic train/validation engine for FP32-LIF and SQUAT."""

from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader

from dataset.udd import UDDPatchDataset
from utils.losses.loss import CrossEntropyLoss2d

from .audit import NeuronAuditCollector, OutputAudit
from .checkpointing import load_checkpoint, save_checkpoint
from .model_factory import seed_everything
from .protocol import ExperimentProtocol, atomic_write_json


UDD_ROOT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed")


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_datasets(protocol: ExperimentProtocol) -> tuple[UDDPatchDataset, UDDPatchDataset]:
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(protocol.input_height, protocol.input_width),
        ]
    )
    validation_transform = A.Compose([A.Resize(protocol.input_height, protocol.input_width)])
    train = UDDPatchDataset(
        txt_file=UDD_ROOT / "train_patches.txt", transforms=train_transform
    )
    validation = UDDPatchDataset(
        txt_file=UDD_ROOT / "val_patches.txt", transforms=validation_transform
    )
    if len(validation) != protocol.expected_validation_samples:
        raise RuntimeError(
            f"expected {protocol.expected_validation_samples} validation samples, "
            f"found {len(validation)}"
        )
    return train, validation


def create_loaders(
    protocol: ExperimentProtocol,
    *,
    physical_batch: int,
    generator: torch.Generator,
    workers: int | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    if physical_batch <= 0 or protocol.effective_train_batch % physical_batch:
        raise ValueError("physical batch must be a positive divisor of effective batch 64")
    train, validation = create_datasets(protocol)
    worker_count = protocol.workers if workers is None else workers
    common = {
        "num_workers": worker_count,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train,
        batch_size=physical_batch,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=protocol.validation_batch,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader


class DeviceConfusionMatrix:
    def __init__(self, classes: int, device: torch.device) -> None:
        self.classes = classes
        self.matrix = torch.zeros(classes, classes, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        labels = labels.flatten().long()
        predictions = predictions.flatten().long()
        valid = (labels >= 0) & (labels < self.classes)
        indices = self.classes * labels[valid] + predictions[valid]
        self.matrix += torch.bincount(
            indices, minlength=self.classes * self.classes
        ).reshape(self.classes, self.classes)

    def result(self) -> dict[str, Any]:
        matrix = self.matrix.double()
        denominator = matrix.sum(0) + matrix.sum(1) - matrix.diag()
        iou = matrix.diag() / denominator
        total = matrix.sum()
        return {
            "miou": float(torch.nanmean(iou).item()),
            "per_class_iou": [float(value) for value in iou.cpu().tolist()],
            "pixel_accuracy": float(matrix.diag().sum().item() / total.item()),
            "confusion_matrix": [[int(x) for x in row] for row in self.matrix.cpu().tolist()],
            "valid_pixels": int(total.item()),
        }


def repeat_static(images: torch.Tensor, time_steps: int) -> torch.Tensor:
    return images.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)


def spike_counts(model: nn.Module, images: torch.Tensor, time_steps: int) -> torch.Tensor:
    spikes = model(repeat_static(images, time_steps))
    if not isinstance(spikes, torch.Tensor) or spikes.ndim != 5:
        raise TypeError("direct SNN must return [T,B,C,H,W] tensor")
    return spikes.sum(dim=0)


def all_finite_gradients(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def all_finite_parameters(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


@dataclass(frozen=True)
class StageSettings:
    name: str
    epochs: int
    learning_rate: float
    physical_batch: int


def build_optimizer(
    model: nn.Module, protocol: ExperimentProtocol, learning_rate: float
) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(protocol.adam_beta1, protocol.adam_beta2),
        eps=protocol.adam_eps,
        weight_decay=protocol.weight_decay,
    )


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    fieldnames = list(row)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    protocol: ExperimentProtocol,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_interval: int,
    max_microbatches: int | None = None,
) -> tuple[dict[str, Any], int]:
    model.train()
    functional.reset_net(model)
    optimizer.zero_grad(set_to_none=True)
    dataset_samples = len(loader.dataset)
    processed = group_samples = 0
    group_target = min(protocol.effective_train_batch, dataset_samples)
    loss_sum = 0.0
    started = time.perf_counter()
    peak_memory = optimizer_steps = 0

    for microbatch, batch in enumerate(loader):
        if max_microbatches is not None and microbatch >= max_microbatches:
            break
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        batch_samples = int(images.shape[0])
        if group_samples == 0:
            group_target = min(protocol.effective_train_batch, dataset_samples - processed)
        functional.reset_net(model)
        counts = spike_counts(model, images, protocol.time_steps)
        loss = criterion(counts, labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at epoch={epoch} microbatch={microbatch}")
        (loss * (batch_samples / group_target)).backward()
        loss_sum += float(loss.item()) * batch_samples
        processed += batch_samples
        group_samples += batch_samples
        functional.reset_net(model)

        end_group = group_samples == group_target
        end_limited = max_microbatches is not None and microbatch + 1 == max_microbatches
        if end_group or (end_limited and group_samples > 0):
            if end_limited and not end_group:
                correction = group_target / group_samples
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if not all_finite_gradients(model):
                raise FloatingPointError(
                    f"non-finite gradient at epoch={epoch} global_step={global_step}"
                )
            optimizer.step()
            if not all_finite_parameters(model):
                raise FloatingPointError(
                    f"non-finite parameter at epoch={epoch} global_step={global_step}"
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            optimizer_steps += 1
            group_samples = 0

        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        if microbatch % max(log_interval, 1) == 0:
            print(
                f"train epoch={epoch + 1} micro={microbatch + 1}/{len(loader)} "
                f"samples={processed} loss={loss.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )

    if processed == 0:
        raise RuntimeError("training processed no samples")
    if group_samples:
        raise RuntimeError("training ended with an unstepped accumulation group")
    observed_microbatches = math.ceil(processed / loader.batch_size)
    return {
        "train_loss": loss_sum / processed,
        "train_samples": processed,
        "train_microbatches": observed_microbatches,
        "optimizer_steps": optimizer_steps,
        "train_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory,
    }, global_step


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    protocol: ExperimentProtocol,
    device: torch.device,
    collect_audit: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    functional.reset_net(model)
    confusion = DeviceConfusionMatrix(protocol.classes, device)
    output_audit = OutputAudit(protocol.classes) if collect_audit else None
    neuron_collector = NeuronAuditCollector(model) if collect_audit else None
    started = time.perf_counter()
    samples = batches = 0
    try:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)
            functional.reset_net(model)
            counts = spike_counts(model, images, protocol.time_steps)
            confusion.update(labels, counts.argmax(dim=1))
            if output_audit is not None:
                output_audit.update(counts)
            samples += int(images.shape[0])
            batches += 1
            functional.reset_net(model)
    finally:
        if neuron_collector is not None:
            neuron_collector.close()
    if batches == 0:
        raise RuntimeError("validation processed no batches")
    result = confusion.result()
    result.update(
        {
            "validation_samples": samples,
            "validation_batches": batches,
            "validation_seconds": time.perf_counter() - started,
        }
    )
    if collect_audit:
        assert output_audit is not None and neuron_collector is not None
        result["output_audit"] = output_audit.result()
        result["neuron_audit"] = neuron_collector.result()
    return result


def run_training_stage(
    *,
    model: nn.Module,
    protocol: ExperimentProtocol,
    settings: StageSettings,
    output_dir: Path,
    device: torch.device,
    resume: Path | None = None,
    workers: int | None = None,
    log_interval: int = 20,
) -> dict[str, Any]:
    seed_everything(protocol.seed)
    model.to(device)
    criterion = CrossEntropyLoss2d().to(device)
    loader_generator = torch.Generator().manual_seed(protocol.seed)
    train_loader, validation_loader = create_loaders(
        protocol,
        physical_batch=settings.physical_batch,
        generator=loader_generator,
        workers=workers,
    )
    updates_per_epoch = math.ceil(len(train_loader.dataset) / protocol.effective_train_batch)
    optimizer = build_optimizer(model, protocol, settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=settings.epochs * updates_per_epoch,
        eta_min=protocol.eta_min,
    )
    start_epoch = global_step = 0
    best_miou = float("-inf")
    if resume is not None:
        payload = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
            map_location=device,
        )
        if payload["protocol"] != protocol.to_dict():
            raise RuntimeError("resume protocol mismatch")
        start_epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        best_miou = float(payload["best_miou"])
        loader_generator.set_state(payload["loader_generator_state"])

    stage_started = time.perf_counter()
    peak_memory = 0
    for epoch in range(start_epoch, settings.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            protocol=protocol,
            device=device,
            epoch=epoch,
            global_step=global_step,
            log_interval=log_interval,
        )
        validation = validate(
            model=model,
            loader=validation_loader,
            protocol=protocol,
            device=device,
        )
        peak_memory = max(peak_memory, train_metrics["peak_memory_bytes"])
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["train_loss"],
            "miou": validation["miou"],
            "pixel_accuracy": validation["pixel_accuracy"],
            **{
                f"iou_class_{index}": value
                for index, value in enumerate(validation["per_class_iou"])
            },
            "lr": optimizer.param_groups[0]["lr"],
            "train_samples": train_metrics["train_samples"],
            "optimizer_steps": train_metrics["optimizer_steps"],
            "validation_samples": validation["validation_samples"],
            "train_seconds": train_metrics["train_seconds"],
            "validation_seconds": validation["validation_seconds"],
            "peak_memory_bytes": train_metrics["peak_memory_bytes"],
        }
        append_metrics(output_dir / "metrics.csv", row)
        improved = validation["miou"] > best_miou
        if improved:
            best_miou = validation["miou"]
        checkpoint_args = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_miou": best_miou,
            "protocol": protocol.to_dict(),
            "loader_generator_state": loader_generator.get_state(),
        }
        save_checkpoint(output_dir / "checkpoint_last.pth", **checkpoint_args)
        if improved:
            save_checkpoint(output_dir / "checkpoint_best.pth", **checkpoint_args)
        print(
            f"stage={settings.name} epoch={epoch + 1}/{settings.epochs} "
            f"miou={validation['miou']:.6f} best={best_miou:.6f}",
            flush=True,
        )

    summary = {
        "stage": settings.name,
        "epochs": settings.epochs,
        "physical_batch": settings.physical_batch,
        "effective_batch": protocol.effective_train_batch,
        "updates_per_epoch": updates_per_epoch,
        "best_miou": best_miou,
        "global_step": global_step,
        "elapsed_seconds": time.perf_counter() - stage_started,
        "peak_memory_bytes": peak_memory,
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    return summary

