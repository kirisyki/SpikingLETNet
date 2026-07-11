#!/usr/bin/env python3
"""Measure T=1-reference temporal reuse in quantized SpikingLETNet convolutions.

The target hardware always computes timestep 1.  For timesteps 2..T, each
input-channel convolution window is compared only with the timestep-1 window.
An equal window reuses the timestep-1 partial convolution result; a different
window is recomputed.

The QAT model is evaluated with a single software time frame.  This script
wraps QLayer.quantize_input so that statistics use the exact integer activation
codes produced by the checkpoint's forward pass, including inputs that are
requantized after residual additions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = PROJECT_ROOT / "Network"
TOOLS_DIR = PROJECT_ROOT / "tools"
for directory in (NETWORK_DIR, TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def _install_optional_import_stubs() -> None:
    if "torchsummary" not in sys.modules:
        module = types.ModuleType("torchsummary")
        module.summary = lambda *args, **kwargs: None
        sys.modules["torchsummary"] = module
    if "seaborn" not in sys.modules:
        module = types.ModuleType("seaborn")
        module.heatmap = lambda *args, **kwargs: None
        sys.modules["seaborn"] = module


_install_optional_import_stubs()

from energy_accounting import LayerModePolicy, git_commit, sha256_file, stable_sample_ids  # noqa: E402
from model.SpikingLETNet_shallow_max import SpikingLETNet_shallow_max  # noqa: E402
from quantization.int4_selfbuild import QLayer  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260613-170955/"
    "model_q_best_complete.pt"
)
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
DEFAULT_POLICY = PROJECT_ROOT / "tools/energy_layer_policy_v2.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "temporal_reuse_analysis/results/qat_max_val500"


def pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    values = tuple(map(int, value))
    if len(values) != 2:
        raise ValueError(f"expected a pair, got {value!r}")
    return values


def json_dump(value: Any, path: Path) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UDDImageDataset(Dataset[torch.Tensor]):
    def __init__(self, split_file: Path, max_images: int, input_size: tuple[int, int]) -> None:
        samples: list[str] = []
        with split_file.open() as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 2:
                    samples.append(parts[0])
                if len(samples) >= max_images:
                    break
        if len(samples) < max_images:
            raise ValueError(
                f"requested {max_images} images, but {split_file} contains only {len(samples)} usable rows"
            )
        self.samples = samples
        self.input_size = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.samples[index]).convert("RGB")
        image = image.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


def conv_positive_minima(codes: torch.Tensor, layer: nn.Conv2d, timestep: int) -> torch.Tensor:
    """Return the minimum positive code in every per-channel Conv2d window.

    Zero-valued elements and zero padding use sentinel T+1 because they remain
    zero at every hardware timestep and therefore do not cause a window change.
    """

    if codes.dim() != 4:
        raise ValueError(f"Conv2d activation codes must be [B,C,H,W], got {tuple(codes.shape)}")
    sentinel = float(timestep + 1)
    positive = torch.where(codes > 0, codes, torch.full_like(codes, sentinel))
    pad_h, pad_w = pair(layer.padding)
    if pad_h or pad_w:
        positive = F.pad(positive, (pad_w, pad_w, pad_h, pad_h), value=sentinel)
    return -F.max_pool2d(
        -positive,
        kernel_size=pair(layer.kernel_size),
        stride=pair(layer.stride),
        padding=0,
        dilation=pair(layer.dilation),
    )


def conv_transpose_positive_minima(
    codes: torch.Tensor, layer: nn.ConvTranspose2d, timestep: int
) -> torch.Tensor:
    """Return window minima after ConvTranspose2d zero insertion and padding."""

    if codes.dim() != 4:
        raise ValueError(
            f"ConvTranspose2d activation codes must be [B,C,H,W], got {tuple(codes.shape)}"
        )
    stride_h, stride_w = pair(layer.stride)
    batch, channels, height, width = map(int, codes.shape)
    spaced_height = height + (height - 1) * (stride_h - 1)
    spaced_width = width + (width - 1) * (stride_w - 1)
    sentinel = float(timestep + 1)
    positive = torch.full(
        (batch, channels, spaced_height, spaced_width),
        sentinel,
        dtype=codes.dtype,
        device=codes.device,
    )
    positive[:, :, ::stride_h, ::stride_w] = torch.where(
        codes > 0, codes, torch.full_like(codes, sentinel)
    )

    kernel_h, kernel_w = pair(layer.kernel_size)
    dilation_h, dilation_w = pair(layer.dilation)
    padding_h, padding_w = pair(layer.padding)
    output_padding_h, output_padding_w = pair(layer.output_padding)
    pad_top = dilation_h * (kernel_h - 1) - padding_h
    pad_left = dilation_w * (kernel_w - 1) - padding_w
    pad_bottom = pad_top + output_padding_h
    pad_right = pad_left + output_padding_w
    if min(pad_top, pad_bottom, pad_left, pad_right) < 0:
        raise ValueError(
            f"unsupported negative converted padding for ConvTranspose2d: "
            f"{(pad_left, pad_right, pad_top, pad_bottom)}"
        )
    positive = F.pad(
        positive,
        (pad_left, pad_right, pad_top, pad_bottom),
        value=sentinel,
    )
    return -F.max_pool2d(
        -positive,
        kernel_size=(kernel_h, kernel_w),
        stride=1,
        padding=0,
        dilation=(dilation_h, dilation_w),
    )


def compute_step_histogram(
    codes: torch.Tensor, layer: nn.Conv2d | nn.ConvTranspose2d, timestep: int
) -> torch.Tensor:
    """Count windows requiring 1..T computations under T=1-only reuse.

    A positive integer code q expands to q leading ones.  Let m be the minimum
    positive q in a window.  Timesteps 1..m match timestep 1, while m+1..T do
    not.  Because timestep 1 is always computed, that window requires T+1-m
    computations.  An all-zero window uses one computation.
    """

    if isinstance(layer, nn.ConvTranspose2d):
        minima = conv_transpose_positive_minima(codes, layer, timestep)
    elif isinstance(layer, nn.Conv2d):
        minima = conv_positive_minima(codes, layer, timestep)
    else:
        raise TypeError(f"unsupported layer type: {type(layer).__name__}")

    minima_hist = torch.bincount(minima.round().to(torch.int64).reshape(-1), minlength=timestep + 2)
    step_hist = torch.zeros(timestep + 1, dtype=torch.int64, device=codes.device)
    for minimum in range(1, timestep):
        step_hist[timestep + 1 - minimum] = minima_hist[minimum]
    # Minimum q=T and the all-zero sentinel T+1 both match timestep 1 forever.
    step_hist[1] = minima_hist[timestep] + minima_hist[timestep + 1]
    return step_hist


def op_type(layer: nn.Module) -> str:
    if isinstance(layer, nn.ConvTranspose2d):
        return "conv_transpose2d"
    if isinstance(layer, nn.Conv2d):
        return "conv2d"
    raise TypeError(type(layer).__name__)


@dataclass
class LayerAccumulator:
    name: str
    layer: nn.Conv2d | nn.ConvTranspose2d
    policy_rule: str
    timestep: int
    histogram: torch.Tensor | None = None
    calls: int = 0
    quantization_scale_one_calls: int = 0
    code_min: torch.Tensor | None = None
    code_max: torch.Tensor | None = None
    max_integer_error: torch.Tensor | None = None

    def observe(self, quantized_input: torch.Tensor, scale: float | torch.Tensor) -> None:
        codes = quantized_input.detach() / scale
        rounded = codes.round()
        integer_error = torch.max(torch.abs(codes - rounded))
        if rounded.dim() != 4:
            raise ValueError(
                f"spiking convolution {self.name} expected 4D quantized input after the QLayer "
                f"time flatten, got {tuple(rounded.shape)}"
            )
        histogram = compute_step_histogram(rounded, self.layer, self.timestep)
        self.histogram = histogram if self.histogram is None else self.histogram + histogram
        current_min = rounded.min()
        current_max = rounded.max()
        self.code_min = current_min if self.code_min is None else torch.minimum(self.code_min, current_min)
        self.code_max = current_max if self.code_max is None else torch.maximum(self.code_max, current_max)
        self.max_integer_error = (
            integer_error
            if self.max_integer_error is None
            else torch.maximum(self.max_integer_error, integer_error)
        )
        if isinstance(scale, torch.Tensor):
            scale_is_one = bool(torch.all(scale == 1).item())
        else:
            scale_is_one = float(scale) == 1.0
        self.quantization_scale_one_calls += int(scale_is_one)
        self.calls += 1

    def as_row(self, processed_images: int) -> dict[str, Any]:
        if self.calls == 0 or self.histogram is None:
            raise RuntimeError(f"layer {self.name} was not observed")
        histogram = [int(value) for value in self.histogram.detach().cpu().tolist()]
        code_min = float(self.code_min.item()) if self.code_min is not None else math.nan
        code_max = float(self.code_max.item()) if self.code_max is not None else math.nan
        max_integer_error = (
            float(self.max_integer_error.item()) if self.max_integer_error is not None else math.nan
        )
        if code_min < 0 or code_max > self.timestep:
            raise ValueError(
                f"semantic spiking layer {self.name} produced codes outside [0,{self.timestep}]: "
                f"min={code_min}, max={code_max}"
            )
        if max_integer_error > 1e-4:
            raise ValueError(
                f"layer {self.name} quantized input is not integer-like: max error={max_integer_error}"
            )

        window_sites = sum(histogram[1:])
        compute_step_sum = sum(step * histogram[step] for step in range(1, self.timestep + 1))
        mean_steps = compute_step_sum / window_sites
        kernel_h, kernel_w = pair(self.layer.kernel_size)
        output_channels_per_input = int(self.layer.out_channels) // int(self.layer.groups)
        macs_per_window = output_channels_per_input * kernel_h * kernel_w
        one_step_macs = window_sites * macs_per_window
        executed_macs = compute_step_sum * macs_per_window
        dense_t_macs = one_step_macs * self.timestep
        row: dict[str, Any] = {
            "layer": self.name,
            "op_type": op_type(self.layer),
            "policy_rule": self.policy_rule,
            "kernel_size": f"{kernel_h}x{kernel_w}",
            "stride": "x".join(map(str, pair(self.layer.stride))),
            "padding": "x".join(map(str, pair(self.layer.padding))),
            "dilation": "x".join(map(str, pair(self.layer.dilation))),
            "groups": int(self.layer.groups),
            "in_channels": int(self.layer.in_channels),
            "out_channels": int(self.layer.out_channels),
            "calls": self.calls,
            "processed_images": processed_images,
            "scale_one_calls": self.quantization_scale_one_calls,
            "requantized_calls": self.calls - self.quantization_scale_one_calls,
            "code_min": code_min,
            "code_max": code_max,
            "max_integer_error": max_integer_error,
            "window_sites": window_sites,
            "compute_step_sum": compute_step_sum,
            "mean_compute_timesteps": mean_steps,
            "one_step_macs": one_step_macs,
            "executed_macs": executed_macs,
            "dense_t8_macs": dense_t_macs,
            "mac_skip_ratio_vs_t8": 1.0 - executed_macs / dense_t_macs,
        }
        for step in range(1, self.timestep + 1):
            row[f"windows_{step}_steps"] = histogram[step]
            row[f"ratio_{step}_steps"] = histogram[step] / window_sites
        return row


class QuantizedInputCollector:
    def __init__(self, model: nn.Module, policy: LayerModePolicy, timestep: int) -> None:
        self.model = model
        self.policy = policy
        self.timestep = timestep
        self.accumulators: dict[str, LayerAccumulator] = {}
        self.original_methods: dict[str, tuple[QLayer, Callable[..., Any], bool]] = {}
        self._install()

    def _install(self) -> None:
        for name, module in self.model.named_modules():
            if not isinstance(module, QLayer) or not isinstance(
                module.layer, (nn.Conv2d, nn.ConvTranspose2d)
            ):
                continue
            mode, rule = self.policy.classify(name, "max", op_type(module.layer))
            if mode != "spike":
                continue
            if not bool(module.quant) or not bool(module.activation_quant):
                raise ValueError(f"spiking QLayer {name} does not have weight and activation quantization enabled")
            accumulator = LayerAccumulator(name, module.layer, rule, self.timestep)
            self.accumulators[name] = accumulator
            original = module.quantize_input
            had_instance_override = "quantize_input" in module.__dict__

            def wrapped(
                x: torch.Tensor,
                *,
                _original: Callable[..., Any] = original,
                _accumulator: LayerAccumulator = accumulator,
            ) -> tuple[torch.Tensor, float | torch.Tensor]:
                quantized_input, scale = _original(x)
                _accumulator.observe(quantized_input, scale)
                return quantized_input, scale

            self.original_methods[name] = (module, original, had_instance_override)
            module.quantize_input = wrapped  # type: ignore[method-assign]
        if not self.accumulators:
            raise RuntimeError("no quantized spiking convolution layers matched the semantic policy")

    def remove(self) -> None:
        for module, original, had_instance_override in self.original_methods.values():
            if had_instance_override:
                module.quantize_input = original  # type: ignore[method-assign]
            else:
                del module.quantize_input
        self.original_methods.clear()

    def rows(self, processed_images: int) -> list[dict[str, Any]]:
        missing = [name for name, stats in self.accumulators.items() if stats.calls == 0]
        # PA3 exists in the module tree but is disabled in this model's forward.
        unexpected_missing = [name for name in missing if name != "PA3.conv"]
        if unexpected_missing:
            raise RuntimeError(f"selected spiking layers were not executed: {unexpected_missing}")
        return [
            stats.as_row(processed_images)
            for stats in self.accumulators.values()
            if stats.calls > 0
        ]


def load_qat_model(checkpoint: Path, device: torch.device) -> nn.Module:
    # This repository documents the complete QAT checkpoint as a trusted local pickle.
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if model.__class__.__name__ != SpikingLETNet_shallow_max.__name__:
        raise RuntimeError(
            f"checkpoint class mismatch: expected {SpikingLETNet_shallow_max.__name__}, "
            f"got {model.__class__.__name__}"
        )
    functional.set_step_mode(model, "m")
    if int(getattr(model, "T", -1)) != 8:
        raise ValueError(f"checkpoint model.T must be 8, got {getattr(model, 'T', None)}")
    model.to(device).eval()
    return model


def summarize(rows: Sequence[dict[str, Any]], timestep: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty layer rows")
    total_windows = sum(int(row["window_sites"]) for row in rows)
    total_compute_steps = sum(int(row["compute_step_sum"]) for row in rows)
    total_one_step_macs = sum(int(row["one_step_macs"]) for row in rows)
    total_executed_macs = sum(int(row["executed_macs"]) for row in rows)
    total_dense_t_macs = sum(int(row["dense_t8_macs"]) for row in rows)
    global_histogram = {
        str(step): sum(int(row[f"windows_{step}_steps"]) for row in rows)
        for step in range(1, timestep + 1)
    }
    return {
        "included_layers": len(rows),
        "timestep": timestep,
        "layer_equal_mean_compute_timesteps": sum(
            float(row["mean_compute_timesteps"]) for row in rows
        )
        / len(rows),
        "window_weighted_mean_compute_timesteps": total_compute_steps / total_windows,
        "mac_weighted_mean_compute_timesteps": total_executed_macs / total_one_step_macs,
        "mac_skip_ratio_vs_fixed_t8": 1.0 - total_executed_macs / total_dense_t_macs,
        "mac_reduction_factor_vs_fixed_t8": total_dense_t_macs / total_executed_macs,
        "total_window_sites": total_windows,
        "total_compute_step_sum": total_compute_steps,
        "total_one_step_macs": total_one_step_macs,
        "total_executed_macs": total_executed_macs,
        "total_fixed_t8_macs": total_dense_t_macs,
        "window_compute_step_histogram": global_histogram,
        "window_compute_step_ratios": {
            step: count / total_windows for step, count in global_histogram.items()
        },
    }


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict[str, Any], rows: Sequence[dict[str, Any]], path: Path) -> None:
    ranked = sorted(rows, key=lambda row: float(row["mean_compute_timesteps"]), reverse=True)
    lines = [
        "# SpikingLETNet QAT Temporal Reuse Result",
        "",
        f"- Images: `{summary['processed_images']}` (UDD6 validation, fixed order, batch size 1)",
        f"- Included quantized spiking convolution layers: `{summary['included_layers']}`",
        f"- Hardware timesteps: `{summary['timestep']}`",
        f"- **MAC-weighted mean computed timesteps: `{summary['mac_weighted_mean_compute_timesteps']:.6f}`**",
        f"- Layer-equal mean computed timesteps: `{summary['layer_equal_mean_compute_timesteps']:.6f}`",
        f"- Window-weighted mean computed timesteps: `{summary['window_weighted_mean_compute_timesteps']:.6f}`",
        f"- MAC skip ratio versus fixed T=8: `{summary['mac_skip_ratio_vs_fixed_t8']:.4%}`",
        f"- MAC reduction factor versus fixed T=8: `{summary['mac_reduction_factor_vs_fixed_t8']:.4f}x`",
        "",
        "The main result is MAC-weighted. Timestep 1 is always computed. Each later timestep is",
        "compared only with timestep 1 at per-input-channel convolution-window granularity.",
        "",
        "## Per-layer result",
        "",
        "| Layer | Op | Kernel | Mean steps | MAC skip vs T=8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| `{row['layer']}` | {row['op_type']} | {row['kernel_size']} | "
            f"{float(row['mean_compute_timesteps']):.6f} | "
            f"{float(row['mac_skip_ratio_vs_t8']):.4%} |"
        )
    lines.extend(
        [
            "",
            "Full integer counts, 1–8-step histograms, quantized-code ranges, layer geometry,",
            "and MAC totals are available in `layer_statistics.csv`; provenance is in `summary.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.max_images <= 0:
        raise ValueError("max-images must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    dataset = UDDImageDataset(args.split_file, args.max_images, tuple(args.input_size))
    loader: Iterable[torch.Tensor] = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    policy = LayerModePolicy.load(args.layer_policy)
    model = load_qat_model(args.checkpoint, device)
    collector = QuantizedInputCollector(model, policy, args.timestep)
    processed_images = 0
    started = datetime.now(timezone.utc)
    try:
        with torch.inference_mode():
            for images in loader:
                images = images.to(device, non_blocking=True)
                # One software frame; QIF T=8 integer activations are expanded only by the collector.
                model(images.unsqueeze(0))
                functional.reset_net(model)
                processed_images += int(images.shape[0])
                if processed_images == 1 or processed_images % args.progress_every == 0:
                    print(f"processed {processed_images}/{args.max_images} images", flush=True)
    finally:
        collector.remove()
    if processed_images != args.max_images:
        raise RuntimeError(f"processed {processed_images} images, expected {args.max_images}")

    rows = collector.rows(processed_images)
    aggregate = summarize(rows, args.timestep)
    finished = datetime.now(timezone.utc)
    summary = {
        "method": "t1_reference_per_input_channel_window_reuse",
        "model": "SpikingLETNet_shallow_max",
        "checkpoint_kind": "qat_int4",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "dataset_split": str(args.split_file.resolve()),
        "dataset_split_sha256": sha256_file(args.split_file),
        "processed_images": processed_images,
        "batch_size": 1,
        "sample_ids": stable_sample_ids(dataset.samples),
        "input_size": list(args.input_size),
        "single_software_frame": True,
        "integer_encoding": "q leading ones followed by T-q zeros",
        "comparison_reference": "timestep_1_only",
        "comparison_granularity": "input_channel_x_output_position_x_receptive_field",
        "timestep_1_always_computed": True,
        "conv_transpose_method": "zero_insert_then_convolution",
        "layer_selection": "QLayer Conv2d/ConvTranspose2d classified spike by semantic policy",
        "layer_policy": str(args.layer_policy.resolve()),
        "layer_policy_sha256": policy.sha256,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": git_commit(PROJECT_ROOT),
        "seed": args.seed,
        **aggregate,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "layer_statistics.csv")
    json_dump(summary, args.output_dir / "summary.json")
    write_report(summary, rows, args.output_dir / "report.md")
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--layer-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--input-size", type=int, nargs=2, default=(400, 400), metavar=("H", "W"))
    parser.add_argument("--timestep", type=int, default=8, choices=[8])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    _, result = run(parse_args())
    print(json.dumps({
        "processed_images": result["processed_images"],
        "included_layers": result["included_layers"],
        "mac_weighted_mean_compute_timesteps": result["mac_weighted_mean_compute_timesteps"],
        "mac_skip_ratio_vs_fixed_t8": result["mac_skip_ratio_vs_fixed_t8"],
    }, indent=2))

