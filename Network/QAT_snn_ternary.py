"""Independent ternary QAT entrypoint for SpikingLETNet_shallow_max.

This script is separate from the existing int4 training and quantization
implementation. It refuses to overwrite non-empty run directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from builders.dataset_builder import build_dataset_train
from builders.model_builder import build_model
from quantization.ternary_qat import (
    export_master_state_dict,
    iter_ternary_layers,
    quantize_ternary_model,
    ternary_model_stats,
)
from spikingjelly.activation_based import functional
from utils.losses.loss import CrossEntropyLoss2d


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_FP_CHECKPOINT = (
    REPO_ROOT
    / "checkpoint/udd/"
    "SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ternary_QAT_checkpoint"
THEORETICAL_TERNARY_BITS = math.log2(3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent W1.58 ternary QAT for SpikingLETNet_shallow_max"
    )
    parser.add_argument(
        "--activation-mode",
        choices=["a4", "ternary"],
        required=True,
        help="a4 produces W1.58/A4; ternary produces W1.58/A1.58",
    )
    parser.add_argument("--model", default="SpikingLETNet_shallow_max")
    parser.add_argument("--dataset", default="udd", choices=["udd"])
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--input-size", default="400,400")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FP_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--poly-exp", type=float, default=0.9)
    parser.add_argument("--kd-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--T", type=int, default=1, help="input repeat timesteps")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="exact run directory; must be new unless --resume is used",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint_last.pth from an identical ternary run",
    )
    parser.add_argument(
        "--max-train-iters-per-epoch",
        type=int,
        default=0,
        help="smoke-test limit; 0 uses the full loader",
    )
    parser.add_argument(
        "--max-val-iters",
        type=int,
        default=0,
        help="smoke-test limit; 0 uses the full validation loader",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable cuDNN benchmark and request deterministic algorithms",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.resume is not None:
        resume = args.resume.resolve()
        if not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")
        inferred = resume.parent
        if args.output_dir is not None and args.output_dir.resolve() != inferred:
            raise ValueError(
                "--output-dir must equal the resume checkpoint directory: "
                f"{inferred}"
            )
        return inferred

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        variant = "w1p58_a4" if args.activation_mode == "a4" else "w1p58_a1p58"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = args.run_name or (
            f"{args.model}_{variant}_bs{args.batch_size}_seed{args.seed}_{stamp}"
        )
        output_dir = (args.output_root / args.dataset / run_name).resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_fp_model(args: argparse.Namespace) -> nn.Module:
    model = build_model(args.model, num_classes=args.classes, config=str(args.config))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"Missing 'model' state dict in {args.checkpoint}")
    model.load_state_dict(checkpoint["model"], strict=True)
    functional.set_step_mode(model, step_mode="m")
    return model


def freeze_teacher(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def temporal_average(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() in (3, 5):
        return tensor.mean(0)
    return tensor


def repeat_time(images: torch.Tensor, time_steps: int) -> torch.Tensor:
    return images.unsqueeze(0).repeat(time_steps, 1, 1, 1, 1)


def current_lr(base_lr: float, progress: int, total: int, exponent: float) -> float:
    fraction = min(max(progress / max(total, 1), 0.0), 1.0)
    return base_lr * math.pow(1.0 - fraction, exponent)


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
        denominator = matrix.sum(1) + matrix.sum(0) - torch.diag(matrix)
        iou = torch.diag(matrix) / denominator
        return float(torch.nanmean(iou).item()), [
            float(value) for value in iou.cpu().tolist()
        ]


def create_loaders(
    args: argparse.Namespace,
) -> Tuple[Iterable[Any], Iterable[Any]]:
    height, width = (int(value) for value in args.input_size.split(","))
    _, train_loader, val_loader = build_dataset_train(
        args.dataset,
        (height, width),
        args.batch_size,
        "trainval",
        True,
        True,
        args.num_workers,
    )
    return train_loader, val_loader


def train_one_epoch(
    args: argparse.Namespace,
    teacher: nn.Module,
    student: nn.Module,
    loader: Iterable[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_train_batches: int,
) -> Dict[str, float]:
    teacher.eval()
    student.train()
    kd_criterion = nn.MSELoss()
    total_loss = total_task = total_kd = 0.0
    completed = 0
    started = time.monotonic()
    max_batches = (
        min(total_train_batches, args.max_train_iters_per_epoch)
        if args.max_train_iters_per_epoch > 0
        else total_train_batches
    )

    for iteration, batch in enumerate(loader):
        if iteration >= max_batches:
            break
        images, labels = batch
        images = repeat_time(images.to(device, non_blocking=True), args.T)
        labels = labels.long().to(device, non_blocking=True)

        global_step = epoch * total_train_batches + iteration
        total_steps = args.max_epochs * total_train_batches
        lr = current_lr(args.lr, global_step, total_steps, args.poly_exp)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.no_grad():
            teacher_output, teacher_features = teacher.forward_qat(images)
            teacher_output = temporal_average(teacher_output)
            teacher_features = [
                temporal_average(feature) for feature in teacher_features
            ]

        student_output, student_features = student.forward_qat(images)
        student_output = temporal_average(student_output)
        student_features = [
            temporal_average(feature) for feature in student_features
        ]

        kd_loss = student_output.new_zeros(())
        for teacher_feature, student_feature in zip(
            teacher_features, student_features
        ):
            kd_loss = kd_loss + kd_criterion(
                student_feature, teacher_feature.detach()
            )
        task_loss = criterion(student_output, labels)
        loss = task_loss + args.kd_weight * kd_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch} iteration={iteration}: {loss}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        functional.reset_net(teacher)
        functional.reset_net(student)

        total_loss += float(loss.item())
        total_task += float(task_loss.item())
        total_kd += float(kd_loss.item())
        completed += 1

        if (
            iteration == 0
            or (iteration + 1) % args.log_interval == 0
            or iteration + 1 == max_batches
        ):
            elapsed = time.monotonic() - started
            print(
                f"train epoch={epoch + 1}/{args.max_epochs} "
                f"iter={iteration + 1}/{max_batches} lr={lr:.8f} "
                f"loss={loss.item():.5f} task={task_loss.item():.5f} "
                f"kd={kd_loss.item():.5f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if completed == 0:
        raise RuntimeError("No training batches were completed")
    return {
        "train_loss": total_loss / completed,
        "task_loss": total_task / completed,
        "kd_loss": total_kd / completed,
        "lr": optimizer.param_groups[0]["lr"],
        "train_batches": completed,
        "train_seconds": time.monotonic() - started,
    }


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    confusion = ConfusionMatrix(args.classes, device)
    started = time.monotonic()
    total_batches = len(loader)  # type: ignore[arg-type]
    max_batches = (
        min(total_batches, args.max_val_iters)
        if args.max_val_iters > 0
        else total_batches
    )
    completed = 0
    for iteration, batch in enumerate(loader):
        if iteration >= max_batches:
            break
        images, labels = batch
        images = repeat_time(images.to(device, non_blocking=True), args.T)
        labels = labels.long().to(device, non_blocking=True)
        output = temporal_average(model(images))
        confusion.update(labels.flatten(), output.argmax(1).flatten())
        functional.reset_net(model)
        completed += 1
        if (
            iteration == 0
            or (iteration + 1) % args.log_interval == 0
            or iteration + 1 == max_batches
        ):
            print(
                f"val iter={iteration + 1}/{max_batches} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    if completed == 0:
        raise RuntimeError("No validation batches were completed")
    miou, per_class = confusion.compute()
    return {
        "val_miou": miou,
        "per_class_iou": per_class,
        "val_batches": completed,
        "val_seconds": time.monotonic() - started,
    }


def append_metrics(path: Path, row: Dict[str, Any]) -> None:
    scalar_row = {
        key: value
        for key, value in row.items()
        if not isinstance(value, (list, dict))
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_row))
        if not exists:
            writer.writeheader()
        writer.writerow(scalar_row)


def checkpoint_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "format_version": 1,
        "quantization": "ternary_weight_qat",
        "weight_codes": [-1, 0, 1],
        "weight_scale": "per_output_channel_absmean",
        "activation_mode": args.activation_mode,
        "activation_scale": "per_tensor_dynamic",
        "theoretical_weight_bits": THEORETICAL_TERNARY_BITS,
        "seed": args.seed,
        "T": args.T,
        "source_fp_checkpoint": str(args.checkpoint.resolve()),
    }


def save_last_checkpoint(
    path: Path,
    args: argparse.Namespace,
    epoch: int,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_miou: float,
    best_epoch: int,
    history: Sequence[Dict[str, Any]],
) -> None:
    payload = {
        "epoch": epoch,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
        "best_epoch": best_epoch,
        "history": list(history),
        "metadata": checkpoint_metadata(args),
    }
    atomic_torch_save(payload, path)


def validate_resume_metadata(
    args: argparse.Namespace, checkpoint: Dict[str, Any]
) -> None:
    metadata = checkpoint.get("metadata", {})
    expected = checkpoint_metadata(args)
    for key in ("quantization", "activation_mode", "seed", "T"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"Resume metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected[key]!r}"
            )


def run(args: argparse.Namespace) -> Path:
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if args.T <= 0:
        raise ValueError("--T must be positive")
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"FP checkpoint does not exist: {args.checkpoint}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ternary SpikingLETNet training")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    setup_seed(args.seed, args.deterministic)
    output_dir = resolve_output_dir(args)

    teacher = load_fp_model(args)
    student = quantize_ternary_model(
        teacher, activation_mode=args.activation_mode, inplace=False
    )
    freeze_teacher(teacher)
    functional.set_step_mode(student, step_mode="m")
    layer_names = [name for name, _ in iter_ternary_layers(student)]
    if not layer_names:
        raise RuntimeError("No ternary layers were created")
    print(
        f"built ternary student with {len(layer_names)} wrapped layers; "
        f"first={layer_names[0]} last={layer_names[-1]}",
        flush=True,
    )

    teacher = teacher.to(device)
    student = student.to(device)
    criterion = CrossEntropyLoss2d().to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in student.parameters() if parameter.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    train_loader, val_loader = create_loaders(args)
    total_train_batches = len(train_loader)  # type: ignore[arg-type]

    start_epoch = 0
    best_miou = -math.inf
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        validate_resume_metadata(args, resume_checkpoint)
        student.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint["epoch"])
        best_miou = float(resume_checkpoint["best_miou"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        history = list(resume_checkpoint.get("history", []))
        print(
            f"resumed {args.resume} at epoch={start_epoch} "
            f"best_miou={best_miou:.6f}",
            flush=True,
        )

    manifest = {
        "created_at": utc_now(),
        "status": "running",
        "output_dir": str(output_dir),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "metadata": checkpoint_metadata(args),
        "source_fp_checkpoint_sha256": sha256_file(args.checkpoint),
        "ternary_layer_count": len(layer_names),
        "ternary_layer_names": layer_names,
        "train_batches_per_epoch": total_train_batches,
        "val_batches": len(val_loader),  # type: ignore[arg-type]
    }
    atomic_json_dump(manifest, output_dir / "run_manifest.json")
    metrics_path = output_dir / "metrics.csv"
    last_path = output_dir / "checkpoint_last.pth"
    run_started = time.monotonic()

    try:
        for epoch in range(start_epoch, args.max_epochs):
            train_metrics = train_one_epoch(
                args,
                teacher,
                student,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                total_train_batches,
            )
            val_metrics = evaluate(args, student, val_loader, device)
            stats = ternary_model_stats(student)
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "weight_zero_fraction": stats.zero_fraction,
                "elapsed_seconds": time.monotonic() - run_started,
            }
            history.append(row)
            append_metrics(metrics_path, row)

            if val_metrics["val_miou"] >= best_miou:
                best_miou = float(val_metrics["val_miou"])
                best_epoch = epoch + 1
                metadata = checkpoint_metadata(args)
                metadata["ternary_stats"] = stats.as_dict()
                master_payload = {
                    "epoch": epoch + 1,
                    "model": export_master_state_dict(student),
                    "best_miou": best_miou,
                    "metadata": metadata,
                }
                quantized_payload = {
                    "epoch": epoch + 1,
                    "model": student.state_dict(),
                    "best_miou": best_miou,
                    "metadata": metadata,
                }
                atomic_torch_save(master_payload, output_dir / "model_best.pth")
                atomic_torch_save(
                    quantized_payload, output_dir / "model_q_best.pth"
                )
                atomic_json_dump(
                    {
                        "epoch": best_epoch,
                        "val_miou": best_miou,
                        "per_class_iou": val_metrics["per_class_iou"],
                        "ternary_stats": stats.as_dict(),
                    },
                    output_dir / "best_metrics.json",
                )

            save_last_checkpoint(
                last_path,
                args,
                epoch + 1,
                student,
                optimizer,
                best_miou,
                best_epoch,
                history,
            )
            print(
                f"epoch={epoch + 1} train_loss={train_metrics['train_loss']:.6f} "
                f"val_miou={val_metrics['val_miou']:.6f} "
                f"best={best_miou:.6f}@{best_epoch} "
                f"zero_fraction={stats.zero_fraction:.6f}",
                flush=True,
            )
    except BaseException:
        manifest.update(
            {
                "status": "interrupted",
                "updated_at": utc_now(),
                "completed_epochs": len(history),
                "best_miou": best_miou if math.isfinite(best_miou) else None,
                "best_epoch": best_epoch,
            }
        )
        atomic_json_dump(manifest, output_dir / "run_manifest.json")
        raise

    manifest.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "completed_epochs": len(history),
            "best_miou": best_miou,
            "best_epoch": best_epoch,
            "elapsed_seconds": time.monotonic() - run_started,
        }
    )
    atomic_json_dump(manifest, output_dir / "run_manifest.json")
    print(f"training complete: {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    run(parse_args())
