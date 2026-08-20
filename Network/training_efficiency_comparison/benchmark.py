#!/usr/bin/env python3
"""Benchmark QIF/QAD against recurrent-LIF QAT+SQUAT on one CUDA device.

The public entry point is ``run``. It launches every measured configuration in
an isolated subprocess, so an OOM or CUDA allocator history from one method
cannot contaminate another method's peak-memory measurement.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

NETWORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NETWORK_DIR.parent
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))
os.chdir(NETWORK_DIR)

import albumentations as A  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402

from builders.model_builder import build_model  # noqa: E402
from dataset.udd import UDDPatchDataset  # noqa: E402
from quantization.int4_selfbuild import QLayer, quantize_model  # noqa: E402
from squat_comparison.model_factory import (  # noqa: E402
    build_fp32_lif_model,
    build_squat_from_fp32,
)
from squat_comparison.protocol import DEFAULT_CONFIG  # noqa: E402
from utils.losses.loss import CrossEntropyLoss2d  # noqa: E402

from training_efficiency_comparison.analysis import summarize_runs  # noqa: E402


SEED = 1234
TRAIN_SAMPLES = 25_700
QAD_FP_CHECKPOINT = (
    REPO_ROOT
    / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth"
)
QAD_STUDENT_CHECKPOINT = (
    REPO_ROOT
    / "QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260612-135249/model_q_best.pth"
)
SQUAT_CHECKPOINT = (
    REPO_ROOT / "squat_experiment_outputs/seed1234/w4m4s1_squat/checkpoint_best.pth"
)
UDD_ROOT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed")


@dataclass
class BenchmarkBundle:
    method: str
    trainable: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    reset_modules: tuple[nn.Module, ...]
    teacher: nn.Module | None = None
    kd_criterion: nn.Module | None = None
    kd_weight: float = 0.1
    time_steps: int = 1

    def reset(self) -> None:
        for module in self.reset_modules:
            functional.reset_net(module)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def temporal_average(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(0) if tensor.dim() in (3, 5) else tensor


def _load_qad_source() -> nn.Module:
    source = build_model(
        "SpikingLETNet_shallow_max", num_classes=6, config=str(DEFAULT_CONFIG)
    )
    payload = torch.load(QAD_FP_CHECKPOINT, map_location="cpu", weights_only=False)
    source.load_state_dict(payload["model"], strict=True)
    functional.set_step_mode(source, step_mode="m")
    return source


def build_bundle(method: str, *, device: torch.device, time_steps: int) -> BenchmarkBundle:
    seed_everything()
    criterion = CrossEntropyLoss2d().to(device)
    if method in {"qad", "qif_ste"}:
        source = _load_qad_source()
        student = quantize_model(
            source,
            k=4,
            inplace=False,
            quant=True,
            activation_quant=True,
            quant_start_layer=0,
            activation_quant_mode="per_tensor",
        )
        payload = torch.load(
            QAD_STUDENT_CHECKPOINT, map_location="cpu", weights_only=False
        )
        student.load_state_dict(payload["model"], strict=True)
        functional.set_step_mode(student, step_mode="m")
        student.train().to(device)
        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        if method == "qad":
            source.eval().to(device)
            for parameter in source.parameters():
                parameter.requires_grad_(False)
            return BenchmarkBundle(
                method=method,
                trainable=student,
                optimizer=optimizer,
                criterion=criterion,
                reset_modules=(source, student),
                teacher=source,
                kd_criterion=nn.MSELoss().to(device),
                time_steps=1,
            )
        del source
        gc.collect()
        return BenchmarkBundle(
            method=method,
            trainable=student,
            optimizer=optimizer,
            criterion=criterion,
            reset_modules=(student,),
            time_steps=1,
        )

    if method == "squat":
        fp_model, _ = build_fp32_lif_model(
            config=DEFAULT_CONFIG, classes=6, seed=SEED
        )
        model, _ = build_squat_from_fp32(fp_model, expected_weight_layers=71)
        payload = torch.load(SQUAT_CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        del fp_model, payload
        gc.collect()
        model.train().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )
        return BenchmarkBundle(
            method=method,
            trainable=model,
            optimizer=optimizer,
            criterion=criterion,
            reset_modules=(model,),
            time_steps=time_steps,
        )
    raise ValueError(f"unsupported method: {method}")


class CudaPhaseRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def measure(self, name: str, operation: Callable[[], Any]) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        self.events.append((name, start, end))
        return result

    def seconds(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, start, end in self.events:
            result[name] = result.get(name, 0.0) + start.elapsed_time(end) / 1000.0
        return result


def _qad_forward(
    bundle: BenchmarkBundle,
    images: torch.Tensor,
    labels: torch.Tensor,
    recorder: CudaPhaseRecorder,
) -> torch.Tensor:
    assert bundle.teacher is not None and bundle.kd_criterion is not None
    sequence = images.unsqueeze(0)

    def teacher_forward() -> tuple[torch.Tensor, list[torch.Tensor]]:
        with torch.no_grad():
            output, features = bundle.teacher.forward_qat(sequence)
            return temporal_average(output), [temporal_average(x) for x in features]

    _, teacher_features = recorder.measure("teacher_forward", teacher_forward)

    def student_forward() -> tuple[torch.Tensor, list[torch.Tensor]]:
        output, features = bundle.trainable.forward_qat(sequence)
        return temporal_average(output), [temporal_average(x) for x in features]

    output, student_features = recorder.measure("student_forward", student_forward)

    def calculate_loss() -> torch.Tensor:
        kd_loss = output.new_tensor(0.0)
        for teacher_feature, student_feature in zip(
            teacher_features, student_features, strict=True
        ):
            kd_loss = kd_loss + bundle.kd_criterion(
                student_feature, teacher_feature.detach()
            )
        return bundle.criterion(output, labels) + bundle.kd_weight * kd_loss

    return recorder.measure("loss", calculate_loss)


def _qif_ste_forward(
    bundle: BenchmarkBundle,
    images: torch.Tensor,
    labels: torch.Tensor,
    recorder: CudaPhaseRecorder,
) -> torch.Tensor:
    sequence = images.unsqueeze(0)

    def student_forward() -> torch.Tensor:
        output, _ = bundle.trainable.forward_qat(sequence)
        return temporal_average(output)

    output = recorder.measure("student_forward", student_forward)
    return recorder.measure("loss", lambda: bundle.criterion(output, labels))


def _squat_forward(
    bundle: BenchmarkBundle,
    images: torch.Tensor,
    labels: torch.Tensor,
    recorder: CudaPhaseRecorder,
) -> torch.Tensor:
    def snn_forward() -> torch.Tensor:
        sequence = images.unsqueeze(0).repeat(bundle.time_steps, 1, 1, 1, 1)
        spikes = bundle.trainable(sequence)
        if spikes.ndim != 5:
            raise RuntimeError("SQUAT output must have shape [T,B,C,H,W]")
        return spikes.sum(0)

    counts = recorder.measure("snn_forward", snn_forward)
    return recorder.measure("loss", lambda: bundle.criterion(counts, labels))


def training_update(
    bundle: BenchmarkBundle,
    *,
    cpu_images: torch.Tensor,
    cpu_labels: torch.Tensor,
    gpu_images: torch.Tensor | None,
    gpu_labels: torch.Tensor | None,
    physical_batch: int,
    effective_batch: int,
    device: torch.device,
) -> tuple[float, dict[str, float], float]:
    accumulation = effective_batch // physical_batch
    if effective_batch % physical_batch:
        raise ValueError("physical batch must divide effective batch")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    bundle.optimizer.zero_grad(set_to_none=True)
    recorder = CudaPhaseRecorder()
    last_loss = float("nan")
    for _ in range(accumulation):
        bundle.reset()
        if gpu_images is None or gpu_labels is None:
            images, labels = recorder.measure(
                "host_to_device",
                lambda: (
                    cpu_images.to(device, non_blocking=True),
                    cpu_labels.to(device, non_blocking=True),
                ),
            )
        else:
            images, labels = gpu_images, gpu_labels
        if bundle.method == "qad":
            loss = _qad_forward(bundle, images, labels, recorder)
            backward_name = "backward"
        elif bundle.method == "qif_ste":
            loss = _qif_ste_forward(bundle, images, labels, recorder)
            backward_name = "backward"
        else:
            loss = _squat_forward(bundle, images, labels, recorder)
            backward_name = "bptt_backward"
        scaled_loss = loss / accumulation
        recorder.measure(backward_name, scaled_loss.backward)
        last_loss = float(loss.detach().item())
        bundle.reset()
        if gpu_images is None:
            del images, labels
    recorder.measure("optimizer_step", bundle.optimizer.step)
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    return wall_seconds, recorder.seconds(), last_loss


def _count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _count_qlayers(module: nn.Module) -> int:
    return sum(isinstance(child, QLayer) for child in module.modules())


def worker(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    result: dict[str, Any] = {
        "status": "started",
        "started_at": utc_now(),
        "method": args.method,
        "mode": args.mode,
        "round": args.round,
        "physical_batch": args.physical_batch,
        "effective_batch": args.effective_batch,
        "time_steps": args.time_steps,
        "input_residency": args.input_residency,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
    }
    if not torch.cuda.is_available():
        result.update(status="error", error="CUDA is unavailable")
        atomic_json(output, result)
        return 2
    if args.effective_batch % args.physical_batch:
        raise ValueError("physical batch must divide effective batch")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything()
    try:
        batch = torch.load(args.batch_cache, map_location="cpu", weights_only=True)
        cpu_images = batch["images"][: args.physical_batch].contiguous()
        cpu_labels = batch["labels"][: args.physical_batch].contiguous().long()
        if cpu_images.shape[0] != args.physical_batch:
            raise RuntimeError("batch cache is smaller than requested physical batch")
        if args.input_residency == "cpu":
            cpu_images = cpu_images.pin_memory()
            cpu_labels = cpu_labels.pin_memory()
            gpu_images = gpu_labels = None
        else:
            gpu_images = cpu_images.to(device)
            gpu_labels = cpu_labels.to(device)
        bundle = build_bundle(args.method, device=device, time_steps=args.time_steps)
        bundle.reset()
        torch.cuda.synchronize(device)
        result["model"] = {
            "trainable_parameters": _count_parameters(bundle.trainable),
            "teacher_parameters": _count_parameters(bundle.teacher)
            if bundle.teacher is not None
            else 0,
            "qlayers": _count_qlayers(bundle.trainable),
        }
        result["memory"] = {
            "model_and_input_allocated_bytes": torch.cuda.memory_allocated(device),
            "model_and_input_reserved_bytes": torch.cuda.memory_reserved(device),
        }

        for index in range(args.warmup_updates):
            wall, _, loss = training_update(
                bundle,
                cpu_images=cpu_images,
                cpu_labels=cpu_labels,
                gpu_images=gpu_images,
                gpu_labels=gpu_labels,
                physical_batch=args.physical_batch,
                effective_batch=args.effective_batch,
                device=device,
            )
            print(
                f"worker method={args.method} mode={args.mode} round={args.round} "
                f"warmup={index + 1}/{args.warmup_updates} wall={wall:.4f}s loss={loss:.5f}",
                flush=True,
            )
        bundle.optimizer.zero_grad(set_to_none=True)
        bundle.reset()
        gc.collect()
        torch.cuda.synchronize(device)
        result["memory"].update(
            steady_allocated_bytes=torch.cuda.memory_allocated(device),
            steady_reserved_bytes=torch.cuda.memory_reserved(device),
        )
        torch.cuda.reset_peak_memory_stats(device)

        trials = []
        for index in range(args.measured_updates):
            wall, phases, loss = training_update(
                bundle,
                cpu_images=cpu_images,
                cpu_labels=cpu_labels,
                gpu_images=gpu_images,
                gpu_labels=gpu_labels,
                physical_batch=args.physical_batch,
                effective_batch=args.effective_batch,
                device=device,
            )
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at measured update {index}")
            finite_gradients = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in bundle.trainable.parameters()
            )
            if not finite_gradients:
                raise FloatingPointError(f"non-finite gradient at measured update {index}")
            trials.append(
                {
                    "trial": index,
                    "wall_seconds": wall,
                    "images_per_second": args.effective_batch / wall,
                    "seconds_per_image": wall / args.effective_batch,
                    "phase_seconds": phases,
                    "loss": loss,
                }
            )
            if index == 0 or (index + 1) % 5 == 0 or index + 1 == args.measured_updates:
                print(
                    f"worker method={args.method} mode={args.mode} round={args.round} "
                    f"measured={index + 1}/{args.measured_updates} wall={wall:.4f}s "
                    f"throughput={args.effective_batch / wall:.3f} img/s",
                    flush=True,
                )
        result["memory"].update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
        result["trials"] = trials
        result["gradient_audit"] = {
            "parameters_with_grad": sum(
                parameter.grad is not None for parameter in bundle.trainable.parameters()
            ),
            "teacher_parameters_with_grad": sum(
                parameter.grad is not None for parameter in bundle.teacher.parameters()
            )
            if bundle.teacher is not None
            else 0,
        }
        result["status"] = "ok"
        result["completed_at"] = utc_now()
        atomic_json(output, result)
        return 0
    except torch.cuda.OutOfMemoryError as error:
        result.update(
            status="oom",
            error=str(error),
            completed_at=utc_now(),
            memory={
                "allocated_bytes_at_failure": torch.cuda.memory_allocated(device),
                "reserved_bytes_at_failure": torch.cuda.memory_reserved(device),
            },
        )
        atomic_json(output, result)
        print(
            f"worker method={args.method} physical_batch={args.physical_batch} OOM",
            flush=True,
        )
        return 0
    except Exception as error:
        result.update(
            status="error",
            error=f"{type(error).__name__}: {error}",
            completed_at=utc_now(),
        )
        atomic_json(output, result)
        raise


def make_batch_cache(path: Path, maximum_batch: int) -> None:
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload["images"].shape[0]) >= maximum_batch:
            return
    transform = A.Compose([A.Resize(400, 400)])
    dataset = UDDPatchDataset(
        txt_file=UDD_ROOT / "train_patches.txt", transforms=transform
    )
    images = []
    labels = []
    for index in range(maximum_batch):
        image, label = dataset[index]
        images.append(image)
        labels.append(label.long())
    payload = {
        "images": torch.stack(images),
        "labels": torch.stack(labels),
        "source": str(UDD_ROOT / "train_patches.txt"),
        "indices": list(range(maximum_batch)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def git_text(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return completed.stdout.strip()


def environment_manifest() -> dict[str, Any]:
    checkpoints = [QAD_FP_CHECKPOINT, QAD_STUDENT_CHECKPOINT, SQUAT_CHECKPOINT]
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("analysis.py").resolve(),
        NETWORK_DIR / "QAT_snn_STE.py",
        NETWORK_DIR / "quantization/int4_selfbuild.py",
        NETWORK_DIR / "squat_comparison/lif_neuron.py",
        NETWORK_DIR / "squat_comparison/state_quantizer.py",
        NETWORK_DIR / "squat_comparison/training.py",
    ]
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    ).stdout.strip()
    return {
        "created_at": utc_now(),
        "purpose": "reportable same-GPU training-efficiency comparison",
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_status_at_start": git_text("status", "--short"),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "amp": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32_matmul": True,
            "allow_tf32_cudnn": True,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_smi_start": nvidia_smi,
        }
        if torch.cuda.is_available()
        else None,
        "protocol": {
            "seed": SEED,
            "input_shape": [3, 400, 400],
            "classes": 6,
            "train_samples": TRAIN_SAMPLES,
            "qad": "QIF T=1 W4A4, frozen FP32 teacher, feature KD weight 0.1",
            "squat": "DirectLIF T=8 W4M4S1, full BPTT",
            "timing": "CUDA-synchronized wall clock; data loading excluded",
            "memory": "torch.cuda peak allocated and reserved bytes",
        },
        "checkpoint_sha256": {str(path): sha256_file(path) for path in checkpoints},
        "source_sha256": {str(path): sha256_file(path) for path in sources},
    }


def launch_worker(
    *,
    output_dir: Path,
    batch_cache: Path,
    method: str,
    mode: str,
    round_index: int,
    physical_batch: int,
    effective_batch: int,
    time_steps: int,
    input_residency: str,
    warmup_updates: int,
    measured_updates: int,
) -> dict[str, Any]:
    tag = (
        f"{mode}_{method}_t{time_steps}_p{physical_batch}_e{effective_batch}"
        f"_r{round_index}"
    )
    output = output_dir / "runs" / f"{tag}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--output",
        str(output),
        "--batch-cache",
        str(batch_cache),
        "--method",
        method,
        "--mode",
        mode,
        "--round",
        str(round_index),
        "--physical-batch",
        str(physical_batch),
        "--effective-batch",
        str(effective_batch),
        "--time-steps",
        str(time_steps),
        "--input-residency",
        input_residency,
        "--warmup-updates",
        str(warmup_updates),
        "--measured-updates",
        str(measured_updates),
    ]
    print(f"launching {tag}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if not output.is_file():
        raise RuntimeError(f"worker produced no result for {tag}; exit={completed.returncode}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if completed.returncode and result.get("status") != "oom":
        raise RuntimeError(f"worker failed for {tag}: {result.get('error')}")
    return result


def find_capacity(
    *,
    output_dir: Path,
    batch_cache: Path,
    method: str,
    candidates: list[int],
) -> tuple[int, list[dict[str, Any]]]:
    results = []
    for candidate in candidates:
        result = launch_worker(
            output_dir=output_dir,
            batch_cache=batch_cache,
            method=method,
            mode="probe",
            round_index=0,
            physical_batch=candidate,
            effective_batch=64,
            time_steps=8 if method == "squat" else 1,
            input_residency="cpu",
            warmup_updates=1,
            measured_updates=1,
        )
        results.append(result)
        if result["status"] == "ok":
            return candidate, results
    raise RuntimeError(f"no physical batch candidate fits for {method}")


def flatten_trials(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if run.get("status") != "ok":
            continue
        for trial in run["trials"]:
            row = {
                "mode": run["mode"],
                "method": run["method"],
                "round": run["round"],
                "trial": trial["trial"],
                "physical_batch": run["physical_batch"],
                "effective_batch": run["effective_batch"],
                "time_steps": run["time_steps"],
                "input_residency": run["input_residency"],
                "wall_seconds": trial["wall_seconds"],
                "images_per_second": trial["images_per_second"],
                "seconds_per_image": trial["seconds_per_image"],
                "loss": trial["loss"],
                "peak_allocated_bytes": run["memory"]["peak_allocated_bytes"],
                "peak_reserved_bytes": run["memory"]["peak_reserved_bytes"],
                "steady_allocated_bytes": run["memory"]["steady_allocated_bytes"],
            }
            for phase, seconds in trial["phase_seconds"].items():
                row[f"phase_{phase}_seconds"] = seconds
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preliminary_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Integer-LIF + QAD 与直接 QAT+SQUAT 训练效率对比",
        "",
        "> 此文件由 benchmark 自动生成；最终论文表述需经过统计与方法学验证。",
        "",
        f"- GPU：`{manifest['gpu']['name']}`",
        "- 输入：UDD 真实样本，400×400，AMP 关闭",
        "- 计时：CUDA 同步 wall clock；不含数据读取、验证和 checkpoint I/O",
        "",
        "## 汇总",
        "",
        "| 模式 | 方法 | physical/effective batch | T | 峰值显存 GiB | update 中位时间 s | 吞吐 img/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        lines.append(
            f"| {group['mode']} | {group['method']} | "
            f"{group['physical_batch']}/{group['effective_batch']} | "
            f"{group['time_steps']} | "
            f"{group['memory']['peak_allocated_bytes'] / 2**30:.3f} | "
            f"{group['wall_seconds']['median']:.6f} | "
            f"{group['images_per_second']['median']:.3f} |"
        )
    lines.extend(["", "## 主比较倍率", ""])
    for comparison in summary["comparisons"]:
        details = [
            f"### {comparison['mode']}",
            "",
            f"- QAD 吞吐加速：`{comparison['qad_throughput_speedup']:.3f}×`",
            f"- QAD update 时间降低：`{comparison['qad_time_reduction_fraction']:.2%}`",
        ]
        if comparison["memory_comparable"]:
            details.extend(
                [
                    f"- QAD 峰值显存降低：`{comparison['qad_memory_reduction_fraction']:.2%}`",
                    f"- SQUAT/QAD 显存倍率：`{comparison['squat_to_qad_memory_ratio']:.3f}×`",
                ]
            )
        else:
            details.append("- 峰值显存倍率：`不计算（physical batch 不同）`")
        details.append("")
        lines.extend(details)
    lines.extend(
        [
            "## 结论边界",
            "",
            "主比较同时改变 QIF/LIF、T=1/T=8、BPTT、状态/激活量化与教师蒸馏，",
            "因此只能解释为完整训练路线的效率差异，不能把全部差异因果归因于 QAD。",
            "PyTorch fake quantization 的训练耗时也不代表低比特硬件推理速度或能耗。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    requested_output = Path(args.output_dir)
    output_dir = (
        requested_output.resolve()
        if requested_output.is_absolute()
        else (REPO_ROOT / requested_output).resolve()
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir()
    manifest = environment_manifest()
    manifest["run_configuration"] = vars(args)
    atomic_json(output_dir / "manifest.json", manifest)
    batch_cache = output_dir / "fixed_real_batch.pt"
    make_batch_cache(batch_cache, 64)

    candidates = [64, 32, 16, 8, 4, 2, 1]
    capacities: dict[str, int] = {}
    runs: list[dict[str, Any]] = []
    for method in ("qad", "squat"):
        capacity, probe_runs = find_capacity(
            output_dir=output_dir,
            batch_cache=batch_cache,
            method=method,
            candidates=candidates,
        )
        capacities[method] = capacity
        runs.extend(probe_runs)
        atomic_json(
            output_dir / "progress.json",
            {"status": "probing", "capacities": capacities, "updated_at": utc_now()},
        )
    common_batch = min(capacities.values())
    protocol = {
        "capacities": capacities,
        "common_physical_batch": common_batch,
        "matched_effective_batch": common_batch,
        "capacity_effective_batch": 64,
        "rounds": args.rounds,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "ablation_rounds": args.ablation_rounds,
        "ablation_updates": args.ablation_updates,
    }
    atomic_json(output_dir / "protocol.json", protocol)

    formal_runs: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        method_order = ("qad", "squat") if round_index % 2 == 0 else ("squat", "qad")
        for mode in ("matched", "capacity"):
            for method in method_order:
                physical = common_batch if mode == "matched" else capacities[method]
                effective = common_batch if mode == "matched" else 64
                result = launch_worker(
                    output_dir=output_dir,
                    batch_cache=batch_cache,
                    method=method,
                    mode=mode,
                    round_index=round_index,
                    physical_batch=physical,
                    effective_batch=effective,
                    time_steps=8 if method == "squat" else 1,
                    input_residency="gpu" if mode == "matched" else "cpu",
                    warmup_updates=args.warmup_updates,
                    measured_updates=args.measured_updates,
                )
                if result["status"] != "ok":
                    raise RuntimeError(f"formal run failed: {result}")
                formal_runs.append(result)
                atomic_json(
                    output_dir / "progress.json",
                    {
                        "status": "formal",
                        "completed_formal_runs": len(formal_runs),
                        "expected_formal_runs": args.rounds * 4,
                        "updated_at": utc_now(),
                    },
                )

    ablation_runs: list[dict[str, Any]] = []
    mechanism_batch = min(common_batch, 4)
    for round_index in range(args.ablation_rounds):
        ablation_runs.append(
            launch_worker(
                output_dir=output_dir,
                batch_cache=batch_cache,
                method="qif_ste",
                mode="qad_ablation",
                round_index=round_index,
                physical_batch=common_batch,
                effective_batch=common_batch,
                time_steps=1,
                input_residency="gpu",
                warmup_updates=max(1, args.warmup_updates // 2),
                measured_updates=args.ablation_updates,
            )
        )
        for time_steps in (1, 2, 4, 8):
            ablation_runs.append(
                launch_worker(
                    output_dir=output_dir,
                    batch_cache=batch_cache,
                    method="squat",
                    mode="timestep_ablation",
                    round_index=round_index,
                    physical_batch=mechanism_batch,
                    effective_batch=mechanism_batch,
                    time_steps=time_steps,
                    input_residency="gpu",
                    warmup_updates=max(1, args.warmup_updates // 2),
                    measured_updates=args.ablation_updates,
                )
            )

    reportable_runs = formal_runs + ablation_runs
    summary = summarize_runs(reportable_runs)
    summary["capacities"] = capacities
    summary["common_physical_batch"] = common_batch
    summary["epoch_projection"] = []
    for group in summary["groups"]:
        if group["mode"] != "capacity" or group["method"] not in {"qad", "squat"}:
            continue
        updates = math.ceil(TRAIN_SAMPLES / int(group["effective_batch"]))
        seconds = float(group["wall_seconds"]["median"]) * updates
        summary["epoch_projection"].append(
            {
                "method": group["method"],
                "training_samples": TRAIN_SAMPLES,
                "effective_batch": group["effective_batch"],
                "optimizer_updates": updates,
                "projected_training_seconds": seconds,
                "projected_training_minutes": seconds / 60.0,
            }
        )
    atomic_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "raw_trials.csv", flatten_trials(reportable_runs))
    (output_dir / "report.md").write_text(
        preliminary_markdown(summary, manifest), encoding="utf-8"
    )
    atomic_json(
        output_dir / "progress.json",
        {"status": "complete", "completed_at": utc_now()},
    )
    batch_cache.unlink(missing_ok=True)
    print(f"benchmark complete: {output_dir}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--batch-cache", required=True)
    worker_parser.add_argument("--method", choices=["qad", "qif_ste", "squat"], required=True)
    worker_parser.add_argument("--mode", required=True)
    worker_parser.add_argument("--round", type=int, default=0)
    worker_parser.add_argument("--physical-batch", type=int, required=True)
    worker_parser.add_argument("--effective-batch", type=int, required=True)
    worker_parser.add_argument("--time-steps", type=int, required=True)
    worker_parser.add_argument("--input-residency", choices=["cpu", "gpu"], required=True)
    worker_parser.add_argument("--warmup-updates", type=int, default=1)
    worker_parser.add_argument("--measured-updates", type=int, default=1)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--rounds", type=int, default=5)
    run_parser.add_argument("--warmup-updates", type=int, default=10)
    run_parser.add_argument("--measured-updates", type=int, default=20)
    run_parser.add_argument("--ablation-rounds", type=int, default=3)
    run_parser.add_argument("--ablation-updates", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return worker(args) if args.command == "worker" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
