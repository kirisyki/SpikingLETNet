#!/usr/bin/env python3
"""Measure the T=1 QIF dense FP32 PyTorch baseline on an NVIDIA GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import threading
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pynvml
import torch
from PIL import Image
from torch import nn

from energy_accounting import cv_percent, git_commit, sha256_file, strict_json_dump


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = PROJECT_ROOT / "Network"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))


def install_optional_import_stubs() -> None:
    if "torchsummary" not in sys.modules:
        module = types.ModuleType("torchsummary")
        module.summary = lambda *args, **kwargs: None
        sys.modules["torchsummary"] = module
    if "seaborn" not in sys.modules:
        module = types.ModuleType("seaborn")
        module.heatmap = lambda *args, **kwargs: None
        sys.modules["seaborn"] = module


install_optional_import_stubs()

from model.SpikingLETNet_shallow_max import SpikingLETNet_shallow_max  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")


@dataclass
class PowerSample:
    trial: int
    phase: str
    rel_time_s: float
    wall_time_s: float
    power_w: float
    temperature_c: int
    gpu_util_pct: int
    sm_clock_mhz: int
    mem_clock_mhz: int


class NvmlPowerSampler:
    def __init__(self, handle: Any, interval_s: float, trial: int, phase: str) -> None:
        self.handle = handle
        self.interval_s = interval_s
        self.trial = trial
        self.phase = phase
        self.samples: List[PowerSample] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_perf = 0.0

    def __enter__(self) -> "NvmlPowerSampler":
        self._start_perf = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def now(self) -> float:
        return time.perf_counter() - self._start_perf

    def _run(self) -> None:
        while not self._stop.is_set():
            rel = self.now()
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.samples.append(
                    PowerSample(
                        trial=self.trial,
                        phase=self.phase,
                        rel_time_s=rel,
                        wall_time_s=time.time(),
                        power_w=float(pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0,
                        temperature_c=int(
                            pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                        ),
                        gpu_util_pct=int(util.gpu),
                        sm_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)),
                        mem_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)),
                    )
                )
            except pynvml.NVMLError as exc:
                self.errors.append(str(exc))
            self._stop.wait(self.interval_s)


def interpolate_power(samples: Sequence[PowerSample], target_s: float) -> float:
    if not samples:
        raise ValueError("power trace contains no samples")
    ordered = sorted(samples, key=lambda sample: sample.rel_time_s)
    if target_s <= ordered[0].rel_time_s:
        return ordered[0].power_w
    if target_s >= ordered[-1].rel_time_s:
        return ordered[-1].power_w
    for left, right in zip(ordered[:-1], ordered[1:]):
        if left.rel_time_s <= target_s <= right.rel_time_s:
            width = right.rel_time_s - left.rel_time_s
            if width <= 0:
                return right.power_w
            weight = (target_s - left.rel_time_s) / width
            return left.power_w + weight * (right.power_w - left.power_w)
    raise RuntimeError("failed to interpolate power sample")


def integrate_energy_window_j(samples: Sequence[PowerSample], start_s: float, end_s: float) -> float:
    if end_s <= start_s:
        raise ValueError(f"invalid integration window: {start_s}..{end_s}")
    ordered = sorted(samples, key=lambda sample: sample.rel_time_s)
    points = [(start_s, interpolate_power(ordered, start_s))]
    points.extend(
        (sample.rel_time_s, sample.power_w)
        for sample in ordered
        if start_s < sample.rel_time_s < end_s
    )
    points.append((end_s, interpolate_power(ordered, end_s)))
    return sum(0.5 * (left[1] + right[1]) * (right[0] - left[0]) for left, right in zip(points[:-1], points[1:]))


def stable_mean_power(samples: Sequence[PowerSample], duration_s: float, stable_seconds: float = 5.0) -> float:
    if not samples:
        raise ValueError("idle phase contains no power samples")
    lower = max(0.0, duration_s - min(stable_seconds, duration_s))
    stable = [sample.power_w for sample in samples if sample.rel_time_s >= lower]
    if not stable:
        raise ValueError("idle phase has no samples in the stable window")
    return statistics.fmean(stable)


def decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def normalize_uuid(value: Any) -> str:
    text = decode_text(value)
    return text if text.startswith("GPU-") or text.startswith("MIG-") else f"GPU-{text}"


def resolve_nvml_handle(cuda_index: int, explicit_nvml_index: Optional[int]) -> tuple[Any, str]:
    properties = torch.cuda.get_device_properties(cuda_index)
    cuda_uuid = getattr(properties, "uuid", None)
    if cuda_uuid:
        expected_uuid = normalize_uuid(cuda_uuid)
        handle = pynvml.nvmlDeviceGetHandleByUUID(expected_uuid)
        actual_uuid = normalize_uuid(pynvml.nvmlDeviceGetUUID(handle))
        if actual_uuid != expected_uuid:
            raise RuntimeError(f"CUDA/NVML UUID mismatch: {expected_uuid} != {actual_uuid}")
        return handle, "cuda_uuid"
    pci_bus_id = getattr(properties, "pci_bus_id", None)
    if pci_bus_id:
        handle = pynvml.nvmlDeviceGetHandleByPciBusId(decode_text(pci_bus_id))
        return handle, "cuda_pci_bus_id"
    if explicit_nvml_index is None:
        raise RuntimeError(
            "PyTorch did not expose a CUDA UUID/PCI bus id; pass --nvml-index explicitly and verify the device"
        )
    handle = pynvml.nvmlDeviceGetHandleByIndex(explicit_nvml_index)
    cuda_name = decode_text(properties.name)
    nvml_name = decode_text(pynvml.nvmlDeviceGetName(handle))
    if cuda_name != nvml_name:
        raise RuntimeError(f"explicit CUDA/NVML device name mismatch: {cuda_name!r} != {nvml_name!r}")
    return handle, "explicit_nvml_index_name_verified"


def read_total_energy_j(handle: Any) -> Optional[float]:
    try:
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)) / 1000.0
    except (AttributeError, pynvml.NVMLError_NotSupported):
        return None


def foreign_compute_pids(handle: Any) -> list[int]:
    try:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError_NotSupported:
        return []
    return sorted({int(process.pid) for process in processes if int(process.pid) != os.getpid()})


def gpu_info(handle: Any, binding_method: str) -> Dict[str, Any]:
    pci = pynvml.nvmlDeviceGetPciInfo(handle)
    return {
        "name": decode_text(pynvml.nvmlDeviceGetName(handle)),
        "uuid": normalize_uuid(pynvml.nvmlDeviceGetUUID(handle)),
        "pci_bus_id": decode_text(pci.busId),
        "driver": decode_text(pynvml.nvmlSystemGetDriverVersion()),
        "binding_method": binding_method,
        "power_limit_w": float(pynvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0,
        "temperature_c": int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
        "sm_clock_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
        "mem_clock_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
    }


def read_split(split_file: Path, limit: int) -> List[str]:
    paths: List[str] = []
    with split_file.open("r") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                paths.append(parts[0])
            if len(paths) >= limit:
                break
    if len(paths) < limit:
        raise ValueError(f"Split has {len(paths)} images, need {limit}: {split_file}")
    return paths


def load_image_tensor(path: str, input_size: Tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((input_size[1], input_size[0]), Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def parse_size(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("input size must be HEIGHT,WIDTH")
    height, width = map(int, parts)
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("input dimensions must be positive")
    return height, width


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = SpikingLETNet_shallow_max(classes=args.classes, config=str(args.config))
    functional.set_step_mode(model, "m")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_bn_missing = [key for key in missing if ".bn_prelu.bn." not in key]
    if non_bn_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={non_bn_missing[:10]} unexpected={unexpected[:10]}")
    model.to(device).eval()
    return model


def preload_images(args: argparse.Namespace, device: torch.device) -> List[torch.Tensor]:
    total_needed = max(args.measure_images, args.warmup_images)
    tensors = [
        load_image_tensor(path, args.input_size).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
        for path in read_split(args.split_file, total_needed)
    ]
    torch.cuda.synchronize(device)
    return tensors


def run_forward_loop(model: nn.Module, images: Sequence[torch.Tensor], count: int, device: torch.device) -> float:
    start = time.perf_counter()
    with torch.inference_mode():
        for index in range(count):
            model(images[index % len(images)])
            functional.reset_net(model)
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def idle_wait(duration_s: float, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        time.sleep(min(0.01, max(0.0, duration_s - (time.perf_counter() - start))))
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def measure_idle(
    handle: Any,
    device: torch.device,
    trial: int,
    phase: str,
    duration_s: float,
    interval_s: float,
) -> tuple[float, List[PowerSample]]:
    with NvmlPowerSampler(handle, interval_s, trial, phase) as sampler:
        elapsed = idle_wait(duration_s, device)
    if not sampler.samples:
        raise RuntimeError(f"NVML returned no samples for {phase}; errors={sampler.errors[:3]}")
    return elapsed, sampler.samples


def measure_active(
    handle: Any,
    model: nn.Module,
    images: Sequence[torch.Tensor],
    count: int,
    device: torch.device,
    trial: int,
    interval_s: float,
) -> tuple[dict[str, Any], List[PowerSample]]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with NvmlPowerSampler(handle, interval_s, trial, "active") as sampler:
        torch.cuda.synchronize(device)
        start_rel = sampler.now()
        energy_before = read_total_energy_j(handle)
        wall_start = time.perf_counter()
        start_event.record()
        with torch.inference_mode():
            for index in range(count):
                model(images[index % len(images)])
                functional.reset_net(model)
        end_event.record()
        torch.cuda.synchronize(device)
        wall_elapsed = time.perf_counter() - wall_start
        energy_after = read_total_energy_j(handle)
        end_rel = sampler.now()
    if not sampler.samples:
        raise RuntimeError(f"NVML returned no active samples; errors={sampler.errors[:3]}")
    integrated_j = integrate_energy_window_j(sampler.samples, start_rel, end_rel)
    if energy_before is not None and energy_after is not None and energy_after >= energy_before:
        gross_j = energy_after - energy_before
        source = "nvml_total_energy_counter"
    else:
        gross_j = integrated_j
        source = "bounded_power_integration"
    return (
        {
            "active_wall_s": wall_elapsed,
            "active_cuda_s": start_event.elapsed_time(end_event) / 1000.0,
            "active_window_start_s": start_rel,
            "active_window_end_s": end_rel,
            "gross_energy_j": gross_j,
            "integrated_power_energy_j": integrated_j,
            "energy_source": source,
        },
        sampler.samples,
    )


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "cv_pct": cv_percent(values),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# PRO6000 T=1 QIF Dense FP32 Energy",
        "",
        f"Status: **{result['status']}**. Gross board energy is the primary metric; net energy is secondary.",
        "",
        "| Metric | Median | CV |",
        "|---|---:|---:|",
        f"| throughput (images/s) | {summary['images_per_second']['median']:.6f} | {summary['images_per_second']['cv_pct']:.3f}% |",
        f"| gross J/image | {summary['gross_j_per_image']['median']:.9f} | {summary['gross_j_per_image']['cv_pct']:.3f}% |",
        f"| net J/image | {summary['net_j_per_image']['median']:.9f} | {summary['net_j_per_image']['cv_pct']:.3f}% |",
        "",
        "The measured execution keeps QIFNode enabled and uses one input step. Preprocessing and host-to-device transfer are excluded.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "energy/measurements/v2/pro6000_qif_t1_dense_fp32")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA-visible device ordinal.")
    parser.add_argument("--nvml-index", type=int, default=None, help="Explicit fallback only when CUDA UUID/PCI identity is unavailable.")
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--input-size", type=parse_size, default=(400, 400))
    parser.add_argument("--warmup-images", type=int, default=200)
    parser.add_argument("--measure-images", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--sample-interval-ms", type=float, default=100.0)
    parser.add_argument("--tf32", choices=["off", "on"], default="off")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.warmup_images <= 0 or args.measure_images <= 0 or args.trials <= 0:
        parser.error("gpu must be non-negative; image and trial counts must be positive")
    if args.idle_seconds <= 0 or args.sample_interval_ms <= 0:
        parser.error("idle-seconds and sample-interval-ms must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PRO6000 measurement; CPU-only hosts may run --help and unit tests")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    tf32_enabled = args.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled

    pynvml.nvmlInit()
    try:
        handle, binding_method = resolve_nvml_handle(args.gpu, args.nvml_index)
        model = load_model(args, device)
        images = preload_images(args, device)
        foreign = foreign_compute_pids(handle)
        if foreign and not args.allow_shared_gpu:
            raise RuntimeError(f"foreign GPU compute processes detected: {foreign}")
        print(f"Loaded {len(images)} images; warming up {args.warmup_images} forwards")
        run_forward_loop(model, images, args.warmup_images, device)

        interval_s = args.sample_interval_ms / 1000.0
        trials: List[Dict[str, Any]] = []
        trace: List[PowerSample] = []
        for trial_index in range(1, args.trials + 1):
            pre_elapsed, pre_samples = measure_idle(
                handle, device, trial_index, "idle_pre", args.idle_seconds, interval_s
            )
            active, active_samples = measure_active(
                handle, model, images, args.measure_images, device, trial_index, interval_s
            )
            post_elapsed, post_samples = measure_idle(
                handle, device, trial_index, "idle_post", args.idle_seconds, interval_s
            )
            baseline_w = statistics.fmean(
                [stable_mean_power(pre_samples, pre_elapsed), stable_mean_power(post_samples, post_elapsed)]
            )
            gross_j = float(active["gross_energy_j"])
            net_j = gross_j - baseline_w * float(active["active_wall_s"])
            row = {
                "trial": trial_index,
                "measure_images": args.measure_images,
                **active,
                "idle_pre_s": pre_elapsed,
                "idle_post_s": post_elapsed,
                "idle_baseline_w": baseline_w,
                "gross_j_per_image": gross_j / args.measure_images,
                "net_energy_j": net_j,
                "net_j_per_image": net_j / args.measure_images,
                "images_per_second": args.measure_images / float(active["active_wall_s"]),
                "active_sample_count": len(active_samples),
                "idle_pre_sample_count": len(pre_samples),
                "idle_post_sample_count": len(post_samples),
            }
            trials.append(row)
            trace.extend(pre_samples + active_samples + post_samples)
            print(
                f"trial {trial_index}: gross={row['gross_j_per_image']:.6f} J/image "
                f"net={row['net_j_per_image']:.6f} J/image throughput={row['images_per_second']:.3f}"
            )

        metric_summary = {
            name: summarize_values([float(row[name]) for row in trials])
            for name in ("gross_j_per_image", "net_j_per_image", "images_per_second")
        }
        stable = (
            metric_summary["images_per_second"]["cv_pct"] <= 2.0
            and metric_summary["gross_j_per_image"]["cv_pct"] <= 5.0
            and metric_summary["net_j_per_image"]["cv_pct"] <= 10.0
        )
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        result = {
            "schema_version": 2,
            "measurement": "pro6000_qif_t1_dense_fp32",
            "model": "SpikingLETNet_shallow_max",
            "mode": "T=1 QIF-enabled dense PyTorch FP32 tensor execution",
            "primary_metric": "gross_j_per_image",
            "secondary_metric": "net_j_per_image",
            "status": "ready" if stable else "unstable",
            "thresholds": {"throughput_cv_pct": 2.0, "gross_cv_pct": 5.0, "net_cv_pct": 10.0},
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "split_file": str(args.split_file),
            "split_sha256": sha256_file(args.split_file),
            "input_size": list(args.input_size),
            "batch_size": 1,
            "warmup_images": args.warmup_images,
            "measure_images": args.measure_images,
            "trials": args.trials,
            "idle_seconds": args.idle_seconds,
            "sample_interval_ms": args.sample_interval_ms,
            "tf32": args.tf32,
            "gpu": gpu_info(handle, binding_method),
            "foreign_compute_pids": foreign,
            "summary": metric_summary,
            "trial_results": trials,
            "notes": [
                "Gross board energy is primary; net energy uses stable pre/post-idle baseline power.",
                "Data loading, preprocessing, host-to-device transfer, and argmax are excluded.",
                "QIFNode remains enabled; this is not a continuous-activation ANN.",
            ],
        }
        manifest = {
            "schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(sys.argv),
            "git_commit": git_commit(PROJECT_ROOT),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "numpy": np.__version__,
                "pynvml": getattr(pynvml, "__version__", None),
            },
            "result_status": result["status"],
        }
        strict_json_dump(result, run_dir / "result.json")
        strict_json_dump(manifest, run_dir / "manifest.json")
        write_csv(run_dir / "trials.csv", trials)
        write_csv(run_dir / "power_trace.csv", [asdict(sample) for sample in trace])
        write_markdown(run_dir / "result.md", result)
        print(f"wrote measurement run to {run_dir}")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
