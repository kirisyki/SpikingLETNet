#!/usr/bin/env python3
"""Measure ANN-mode FP32 SpikingLETNet_shallow_max inference energy on PRO6000."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pynvml
import torch
from PIL import Image
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = PROJECT_ROOT / "Network"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))


def install_optional_import_stubs() -> None:
    if "torchsummary" not in sys.modules:
        torchsummary_stub = types.ModuleType("torchsummary")
        torchsummary_stub.summary = lambda *args, **kwargs: None
        sys.modules["torchsummary"] = torchsummary_stub
    if "seaborn" not in sys.modules:
        seaborn_stub = types.ModuleType("seaborn")
        seaborn_stub.heatmap = lambda *args, **kwargs: None
        sys.modules["seaborn"] = seaborn_stub


install_optional_import_stubs()

from model.SpikingLETNet_shallow_max import SpikingLETNet_shallow_max  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")


@dataclass
class PowerSample:
    rel_time_s: float
    wall_time_s: float
    power_w: float
    temperature_c: int
    gpu_util_pct: int
    sm_clock_mhz: int
    mem_clock_mhz: int
    phase: str


class NvmlPowerSampler:
    def __init__(self, gpu_index: int, interval_s: float, phase: str) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.phase = phase
        self.samples: List[PowerSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_wall = 0.0

    def __enter__(self) -> "NvmlPowerSampler":
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._start_wall = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        pynvml.nvmlShutdown()

    def _run(self) -> None:
        while not self._stop.is_set():
            wall = time.perf_counter()
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                sample = PowerSample(
                    rel_time_s=wall - self._start_wall,
                    wall_time_s=time.time(),
                    power_w=pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
                    temperature_c=pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU),
                    gpu_util_pct=int(util.gpu),
                    sm_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)),
                    mem_clock_mhz=int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)),
                    phase=self.phase,
                )
                self.samples.append(sample)
            except pynvml.NVMLError:
                pass
            self._stop.wait(self.interval_s)


def integrate_energy_j(samples: List[PowerSample], elapsed_s: float) -> float:
    if not samples:
        return float("nan")
    if len(samples) == 1:
        return samples[0].power_w * elapsed_s
    energy = 0.0
    for left, right in zip(samples[:-1], samples[1:]):
        dt = max(0.0, right.rel_time_s - left.rel_time_s)
        energy += 0.5 * (left.power_w + right.power_w) * dt
    covered = samples[-1].rel_time_s - samples[0].rel_time_s
    if elapsed_s > covered:
        tail = elapsed_s - covered
        energy += samples[-1].power_w * tail
    return energy


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
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def parse_size(value: str) -> Tuple[int, int]:
    height, width = value.split(",")
    return int(height), int(width)


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
    model.to(device)
    model.eval()
    return model


def preload_images(args: argparse.Namespace, device: torch.device) -> List[torch.Tensor]:
    total_needed = max(args.measure_images, args.warmup_images)
    image_paths = read_split(args.split_file, total_needed)
    tensors = []
    for path in image_paths:
        tensor = load_image_tensor(path, args.input_size).unsqueeze(0).unsqueeze(0)
        tensors.append(tensor.to(device, non_blocking=True))
    torch.cuda.synchronize(device)
    return tensors


def run_forward_loop(model: nn.Module, images: List[torch.Tensor], count: int, device: torch.device) -> float:
    start = time.perf_counter()
    with torch.inference_mode():
        for index in range(count):
            output = model(images[index % len(images)])
            _ = output.argmax(dim=2)
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


def sample_stats(samples: List[PowerSample]) -> Dict[str, float]:
    powers = [sample.power_w for sample in samples]
    if not powers:
        return {"mean_w": float("nan"), "median_w": float("nan"), "min_w": float("nan"), "max_w": float("nan")}
    return {
        "mean_w": statistics.fmean(powers),
        "median_w": statistics.median(powers),
        "min_w": min(powers),
        "max_w": max(powers),
    }


def gpu_info(gpu_index: int) -> Dict[str, object]:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = {
        "name": pynvml.nvmlDeviceGetName(handle),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0,
        "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
    }
    pynvml.nvmlShutdown()
    return info


def write_power_trace(path: Path, samples: List[PowerSample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PowerSample.__dataclass_fields__.keys()))
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def write_markdown(path: Path, result: Dict[str, object]) -> None:
    lines = [
        "# PRO6000 ANN FP32 Inference Energy",
        "",
        "Measurement target: `SpikingLETNet_shallow_max`, FP32 checkpoint, ANN-mode dense forward (`T=1`, no time-step expansion).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| measured images | {result['measure_images']} |",
        f"| active elapsed (s) | {result['active_elapsed_s']:.6f} |",
        f"| idle elapsed (s) | {result['idle_elapsed_s']:.6f} |",
        f"| active mean power (W) | {result['active_power']['mean_w']:.6f} |",
        f"| idle mean power (W) | {result['idle_power']['mean_w']:.6f} |",
        f"| gross J/image | {result['gross_j_per_image']:.9f} |",
        f"| net J/image | {result['net_j_per_image']:.9f} |",
        f"| images/s | {result['images_per_second']:.6f} |",
        "",
        "Gross energy includes the full GPU power during active forward. Net energy subtracts an equal-duration idle baseline.",
        "Data loading and preprocessing are excluded; images are preloaded to GPU before measurement.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "measurements/pro6000_ann_fp32_max")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--input-size", type=parse_size, default=(400, 400))
    parser.add_argument("--warmup-images", type=int, default=50)
    parser.add_argument("--measure-images", type=int, default=500)
    parser.add_argument("--sample-interval-ms", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    before_info = gpu_info(args.gpu)
    model = load_model(args, device)
    images = preload_images(args, device)

    print(f"Loaded {len(images)} images to {device}")
    print("Warmup...")
    run_forward_loop(model, images, args.warmup_images, device)

    print("Measuring active forward...")
    interval_s = args.sample_interval_ms / 1000.0
    with NvmlPowerSampler(args.gpu, interval_s, "active") as sampler:
        active_elapsed_s = run_forward_loop(model, images, args.measure_images, device)
    active_samples = sampler.samples
    active_energy_j = integrate_energy_j(active_samples, active_elapsed_s)

    print("Measuring idle baseline...")
    with NvmlPowerSampler(args.gpu, interval_s, "idle") as sampler:
        idle_elapsed_s = idle_wait(active_elapsed_s, device)
    idle_samples = sampler.samples
    idle_energy_j = integrate_energy_j(idle_samples, active_elapsed_s)

    after_info = gpu_info(args.gpu)

    gross_j_per_image = active_energy_j / args.measure_images
    net_j_per_image = (active_energy_j - idle_energy_j) / args.measure_images
    result = {
        "measurement": "pro6000_ann_fp32_max",
        "model": "SpikingLETNet_shallow_max",
        "mode": "ANN dense FP32 forward",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split_file": str(args.split_file),
        "input_size": list(args.input_size),
        "warmup_images": args.warmup_images,
        "measure_images": args.measure_images,
        "sample_interval_ms": args.sample_interval_ms,
        "gpu_before": before_info,
        "gpu_after": after_info,
        "active_elapsed_s": active_elapsed_s,
        "idle_elapsed_s": idle_elapsed_s,
        "active_energy_j": active_energy_j,
        "idle_energy_j_same_duration": idle_energy_j,
        "gross_j_per_image": gross_j_per_image,
        "net_j_per_image": net_j_per_image,
        "images_per_second": args.measure_images / active_elapsed_s,
        "active_power": sample_stats(active_samples),
        "idle_power": sample_stats(idle_samples),
        "active_sample_count": len(active_samples),
        "idle_sample_count": len(idle_samples),
        "notes": [
            "Data loading and preprocessing are excluded.",
            "Images are preloaded to GPU before timing and power sampling.",
            "Net energy subtracts an equal-duration idle GPU baseline.",
            "No GPU global settings such as clocks or power limits were changed.",
        ],
    }

    result_json = args.output_dir / "result.json"
    result_md = args.output_dir / "result.md"
    trace_csv = args.output_dir / "power_trace.csv"
    with result_json.open("w") as handle:
        json.dump(result, handle, indent=2)
    write_markdown(result_md, result)
    write_power_trace(trace_csv, active_samples + idle_samples)

    print(f"wrote {result_json}")
    print(f"wrote {result_md}")
    print(f"wrote {trace_csv}")
    print(json.dumps({k: result[k] for k in ["gross_j_per_image", "net_j_per_image", "images_per_second"]}, indent=2))


if __name__ == "__main__":
    main()
