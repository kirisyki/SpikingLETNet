#!/usr/bin/env python3
"""Estimate mixed MAC/SOP energy for SpikingLETNet shallow variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = PROJECT_ROOT / "Network"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))


def install_optional_import_stubs() -> None:
    """Avoid unrelated optional imports blocking the three target models."""
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
from model.SpikingLETNet_shallow_middle import SpikingLETNet_shallow_middle  # noqa: E402
from model.SpikingLETNet_shallow_small import SpikingLETNet_shallow_small  # noqa: E402
from quantization.int4_selfbuild import QLayer  # noqa: E402
from spikingjelly.activation_based import functional, layer  # noqa: E402


VARIANT_CLASSES = {
    "small": SpikingLETNet_shallow_small,
    "middle": SpikingLETNet_shallow_middle,
    "max": SpikingLETNet_shallow_max,
}

DEFAULT_CONFIG = PROJECT_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")

DEFAULT_FP_CHECKPOINTS = {
    "small": PROJECT_ROOT
    / "checkpoint/udd/SpikingLETNet_shallow_smallbs64gpu1_trainval20260611-234750/model_best.pth",
    "middle": PROJECT_ROOT
    / "checkpoint/udd/SpikingLETNet_shallow_middlebs64gpu1_trainval20260611-222604/model_best.pth",
    "max": PROJECT_ROOT
    / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth",
}

DEFAULT_QAT_CHECKPOINTS = {
    "small": PROJECT_ROOT
    / "QAT_checkpoint/udd/SpikingLETNet_shallow_smallbs64gpu1_trainval20260613-200147/model_q_best_complete.pt",
    "middle": PROJECT_ROOT
    / "QAT_checkpoint/udd/SpikingLETNet_shallow_middlebs96gpu1_trainval20260613-155726/model_q_best_complete.pt",
    "max": PROJECT_ROOT
    / "QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260613-170955/model_q_best_complete.pt",
}

DEFAULT_ENERGY = {
    "fp": {
        "mac": 1.0,
        "ac": 1.0,
        "mem_read": 0.0,
        "mem_write": 0.0,
        "neuron_update": 0.0,
    },
    "int4": {
        "mac": 1.0,
        "ac": 1.0,
        "mem_read": 0.0,
        "mem_write": 0.0,
        "neuron_update": 0.0,
    },
}


class UDDImageDataset(Dataset):
    def __init__(self, split_file: Path, input_size: Tuple[int, int]) -> None:
        self.samples: List[Tuple[str, str]] = []
        with split_file.open("r") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], parts[1]))
        if not self.samples:
            raise ValueError(f"No samples found in split file: {split_file}")

        self.input_size = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path, _mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image, dtype="float32") / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1)
        return (image_tensor - self.mean) / self.std


@dataclass
class LayerStats:
    layer: str
    op_type: str
    precision: str
    calls: int = 0
    input_elems: int = 0
    input_sum: float = 0.0
    input_nonzero: int = 0
    dense_macs_total: float = 0.0
    dense_macs_charged: float = 0.0
    sop_total: float = 0.0
    ac_charged: float = 0.0
    core_energy_pj: float = 0.0
    mem_energy_pj: float = 0.0
    neuron_energy_pj: float = 0.0
    classifications: Dict[str, int] = field(default_factory=dict)

    def add_classification(self, name: str) -> None:
        self.classifications[name] = self.classifications.get(name, 0) + 1

    @property
    def total_energy_pj(self) -> float:
        return self.core_energy_pj + self.mem_energy_pj + self.neuron_energy_pj

    @property
    def mean_spikes_per_input(self) -> float:
        if self.input_elems == 0:
            return float("nan")
        return self.input_sum / self.input_elems

    @property
    def nonzero_ratio(self) -> float:
        if self.input_elems == 0:
            return float("nan")
        return self.input_nonzero / self.input_elems

    def as_row(self, variant: str, checkpoint_kind: str) -> Dict[str, Any]:
        return {
            "variant": variant,
            "checkpoint_kind": checkpoint_kind,
            "precision": self.precision,
            "layer": self.layer,
            "op_type": self.op_type,
            "calls": self.calls,
            "classifications": json.dumps(self.classifications, sort_keys=True),
            "input_elems": self.input_elems,
            "input_sum": self.input_sum,
            "input_nonzero": self.input_nonzero,
            "mean_spikes_per_input": self.mean_spikes_per_input,
            "nonzero_ratio": self.nonzero_ratio,
            "dense_macs_total": self.dense_macs_total,
            "dense_macs_charged": self.dense_macs_charged,
            "sop_total": self.sop_total,
            "ac_charged": self.ac_charged,
            "core_energy_pj": self.core_energy_pj,
            "mem_energy_pj": self.mem_energy_pj,
            "neuron_energy_pj": self.neuron_energy_pj,
            "total_energy_pj": self.total_energy_pj,
        }


def parse_input_size(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("input size must be HEIGHT,WIDTH")
    return int(parts[0]), int(parts[1])


def load_energy_config(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    energy = json.loads(json.dumps(DEFAULT_ENERGY))
    if path is None:
        return energy

    with path.open("r") as handle:
        if path.suffix.lower() == ".json":
            loaded = json.load(handle)
        else:
            import yaml

            loaded = yaml.safe_load(handle)

    if "energy_pj" in loaded:
        loaded = loaded["energy_pj"]

    for precision, values in loaded.items():
        if precision not in energy:
            energy[precision] = {}
        for key, value in values.items():
            energy[precision][key] = float(value)
    return energy


def first_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def tensor_batch_time_numel(tensor: torch.Tensor) -> int:
    return tensor.numel()


def conv2d_macs(module: nn.Conv2d, output: torch.Tensor) -> int:
    output_elems = output.numel()
    kernel_ops = (module.in_channels // module.groups) * product(module.kernel_size)
    return int(output_elems * kernel_ops)


def conv_transpose2d_macs(module: nn.ConvTranspose2d, input_tensor: torch.Tensor) -> int:
    input_elems = input_tensor.numel()
    kernel_ops = (module.out_channels // module.groups) * product(module.kernel_size)
    return int(input_elems * kernel_ops)


def linear_macs(module: nn.Linear, output: torch.Tensor) -> int:
    return int(output.numel() * module.in_features)


def module_op_type(module: nn.Module) -> str:
    if isinstance(module, nn.ConvTranspose2d):
        return "conv_transpose2d"
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    if isinstance(module, nn.Linear):
        return "linear"
    return module.__class__.__name__


def module_macs(module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor) -> int:
    if isinstance(module, nn.ConvTranspose2d):
        return conv_transpose2d_macs(module, input_tensor)
    if isinstance(module, nn.Conv2d):
        return conv2d_macs(module, output)
    if isinstance(module, nn.Linear):
        return linear_macs(module, output)
    return 0


def is_supported_compute_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, QLayer))


def unwrap_compute_module(module: nn.Module) -> nn.Module:
    if isinstance(module, QLayer):
        return module.layer
    return module


def is_integer_like(tensor: torch.Tensor, eps: float) -> bool:
    if tensor.numel() == 0:
        return False
    if not torch.is_floating_point(tensor):
        return True
    return bool(torch.max(torch.abs(tensor - torch.round(tensor))).item() <= eps)


def classify_input(
    layer_name: str,
    module: nn.Module,
    input_tensor: torch.Tensor,
    timestep: int,
    integer_eps: float,
) -> str:
    raw_module = unwrap_compute_module(module)
    if isinstance(raw_module, nn.Linear):
        return "dense"
    if "transformer" in layer_name:
        return "dense"

    if input_tensor.numel() == 0:
        return "dense"

    detached = input_tensor.detach()
    min_value = float(detached.min().item())
    max_value = float(detached.max().item())
    if min_value >= -integer_eps and max_value <= timestep + integer_eps and is_integer_like(detached, integer_eps):
        return "spike"
    return "dense"


class EnergyHookCollector:
    def __init__(
        self,
        model: nn.Module,
        precision: str,
        energy: Mapping[str, Mapping[str, float]],
        timestep: int,
        integer_eps: float,
        include_extended: bool,
    ) -> None:
        self.precision = precision
        self.energy = energy
        self.timestep = timestep
        self.integer_eps = integer_eps
        self.include_extended = include_extended
        self.stats: Dict[str, LayerStats] = {}
        self.handles: List[Any] = []
        self.register(model)

    def register(self, model: nn.Module) -> None:
        qlayer_child_names = {
            f"{name}.layer"
            for name, module in model.named_modules()
            if name and isinstance(module, QLayer)
        }
        for name, module in model.named_modules():
            if not name or not is_supported_compute_module(module):
                continue
            if name in qlayer_child_names:
                continue
            if isinstance(module, QLayer):
                raw_module = module.layer
            else:
                raw_module = module
            if not isinstance(raw_module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                continue

            self.stats[name] = LayerStats(
                layer=name,
                op_type=module_op_type(raw_module),
                precision=self.precision,
            )
            self.handles.append(module.register_forward_hook(self.make_hook(name, module)))

    def make_hook(self, name: str, module: nn.Module):
        def hook(_module: nn.Module, inputs: Any, output: Any) -> None:
            input_tensor = first_tensor(inputs)
            output_tensor = first_tensor(output)
            if input_tensor is None or output_tensor is None:
                return

            raw_module = unwrap_compute_module(module)
            dense_macs = float(module_macs(raw_module, input_tensor, output_tensor))
            input_detached = input_tensor.detach()
            input_elems = tensor_batch_time_numel(input_detached)
            input_sum = float(input_detached.clamp(min=0).sum().item())
            input_nonzero = int(torch.count_nonzero(input_detached).item())
            classification = classify_input(name, module, input_detached, self.timestep, self.integer_eps)

            stats = self.stats[name]
            stats.calls += 1
            stats.input_elems += input_elems
            stats.input_sum += input_sum
            stats.input_nonzero += input_nonzero
            stats.dense_macs_total += dense_macs
            stats.add_classification(classification)

            precision_energy = self.energy[self.precision]
            if classification == "spike":
                mean_spikes = input_sum / input_elems if input_elems else 0.0
                # dense_macs already includes the leading time dimension for SNN tensors.
                # QIF integer activations encode cumulative spikes across time, so divide
                # by the actual time length before scaling by mean_spikes.
                actual_timesteps = input_tensor.shape[0] if input_tensor.dim() in (3, 5) else 1
                sop = (dense_macs / max(int(actual_timesteps), 1)) * mean_spikes
                stats.sop_total += sop
                stats.ac_charged += sop
                stats.core_energy_pj += sop * precision_energy.get("ac", 0.0)
            else:
                stats.dense_macs_charged += dense_macs
                stats.core_energy_pj += dense_macs * precision_energy.get("mac", 0.0)

            if self.include_extended:
                reads = float(input_elems)
                writes = float(output_tensor.numel())
                stats.mem_energy_pj += (
                    reads * precision_energy.get("mem_read", 0.0)
                    + writes * precision_energy.get("mem_write", 0.0)
                )
                stats.neuron_energy_pj += writes * precision_energy.get("neuron_update", 0.0)

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_fp_model(variant: str, config: Path, checkpoint: Path, classes: int, device: torch.device) -> nn.Module:
    model = VARIANT_CLASSES[variant](classes=classes, config=str(config))
    functional.set_step_mode(model, "m")
    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_obj.get("model", checkpoint_obj) if isinstance(checkpoint_obj, dict) else checkpoint_obj
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_bn_missing = [key for key in missing if ".bn_prelu.bn." not in key]
    if non_bn_missing or unexpected:
        print(
            f"warning: checkpoint mismatch for {variant} fp: "
            f"missing_non_bn={len(non_bn_missing)} unexpected={len(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model


def load_qat_model(checkpoint: Path, device: torch.device) -> nn.Module:
    # The complete QAT files are trusted local pickle model objects.
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    functional.set_step_mode(model, "m")
    model.to(device)
    model.eval()
    return model


def make_loader(args: argparse.Namespace) -> Iterable[torch.Tensor]:
    if args.synthetic:
        height, width = args.input_size
        total = args.max_batches if args.max_batches > 0 else 1
        return [torch.randn(args.batch_size, 3, height, width) for _ in range(total)]

    dataset = UDDImageDataset(args.split_file, args.input_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )


def repeat_timesteps(images: torch.Tensor, timestep: int) -> torch.Tensor:
    return images.unsqueeze(0).repeat(timestep, 1, 1, 1, 1)


def run_one(
    variant: str,
    checkpoint_kind: str,
    checkpoint: Path,
    args: argparse.Namespace,
    energy: Mapping[str, Mapping[str, float]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(args.device)
    precision = "fp" if checkpoint_kind == "fp" else "int4"
    if checkpoint_kind == "fp":
        model = load_fp_model(variant, args.config, checkpoint, args.classes, device)
    else:
        model = load_qat_model(checkpoint, device)

    collector = EnergyHookCollector(
        model=model,
        precision=precision,
        energy=energy,
        timestep=args.timestep,
        integer_eps=args.integer_eps,
        include_extended=args.include_extended,
    )
    loader = make_loader(args)

    processed = 0
    try:
        with torch.no_grad():
            for batch_index, images in enumerate(loader):
                if args.max_batches > 0 and batch_index >= args.max_batches:
                    break
                images = images.to(device)
                model_input = repeat_timesteps(images, args.timestep)
                model(model_input)
                functional.reset_net(model)
                processed += 1
                print(f"{variant}/{checkpoint_kind}: processed batch {processed}")
    finally:
        collector.remove()

    rows = [stats.as_row(variant, checkpoint_kind) for stats in collector.stats.values()]
    summary = summarize_rows(variant, checkpoint_kind, precision, checkpoint, processed, rows, args, energy)
    return rows, summary


def summarize_rows(
    variant: str,
    checkpoint_kind: str,
    precision: str,
    checkpoint: Path,
    processed_batches: int,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    energy: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    totals = {
        "dense_macs_total": sum(float(row["dense_macs_total"]) for row in rows),
        "dense_macs_charged": sum(float(row["dense_macs_charged"]) for row in rows),
        "sop_total": sum(float(row["sop_total"]) for row in rows),
        "ac_charged": sum(float(row["ac_charged"]) for row in rows),
        "core_energy_pj": sum(float(row["core_energy_pj"]) for row in rows),
        "mem_energy_pj": sum(float(row["mem_energy_pj"]) for row in rows),
        "neuron_energy_pj": sum(float(row["neuron_energy_pj"]) for row in rows),
        "total_energy_pj": sum(float(row["total_energy_pj"]) for row in rows),
    }
    total_inputs = sum(int(row["input_elems"]) for row in rows)
    total_spikes = sum(float(row["input_sum"]) for row in rows)
    return {
        "variant": variant,
        "checkpoint_kind": checkpoint_kind,
        "precision": precision,
        "checkpoint": str(checkpoint),
        "config": str(args.config),
        "processed_batches": processed_batches,
        "batch_size": args.batch_size,
        "timestep": args.timestep,
        "input_size": list(args.input_size),
        "energy_pj": energy[precision],
        "mean_spikes_per_input": total_spikes / total_inputs if total_inputs else float("nan"),
        **totals,
    }


def checkpoint_for(args: argparse.Namespace, variant: str, checkpoint_kind: str) -> Path:
    override = getattr(args, f"{variant}_{checkpoint_kind}_checkpoint")
    if override:
        return override
    if checkpoint_kind == "fp":
        return DEFAULT_FP_CHECKPOINTS[variant]
    return DEFAULT_QAT_CHECKPOINTS[variant]


def write_outputs(output_dir: Path, layer_rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_csv = output_dir / "layer_energy.csv"
    summary_csv = output_dir / "summary_energy.csv"
    summary_json = output_dir / "summary_energy.json"

    if layer_rows:
        with layer_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_rows[0].keys()))
            writer.writeheader()
            writer.writerows(layer_rows)

    if summaries:
        with summary_csv.open("w", newline="") as handle:
            fieldnames = [
                key
                for key in summaries[0].keys()
                if key != "energy_pj"
            ] + ["energy_pj"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    with summary_json.open("w") as handle:
        json.dump({"summaries": summaries}, handle, indent=2)

    print(f"wrote {layer_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_json}")


def add_checkpoint_override_args(parser: argparse.ArgumentParser) -> None:
    for variant in ("small", "middle", "max"):
        parser.add_argument(f"--{variant}-fp-checkpoint", type=Path, default=None)
        parser.add_argument(f"--{variant}-qat-checkpoint", type=Path, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["small", "middle", "max"], choices=list(VARIANT_CLASSES))
    parser.add_argument("--checkpoint-kinds", nargs="+", default=["fp", "qat"], choices=["fp", "qat"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "energy_estimates")
    parser.add_argument("--energy-config", type=Path, default=None)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20, help="Use 0 to process the full split.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-size", type=parse_input_size, default=(400, 400))
    parser.add_argument("--timestep", type=int, default=8)
    parser.add_argument("--integer-eps", type=float, default=1e-4)
    parser.add_argument("--include-extended", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use random images for smoke tests.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    add_checkpoint_override_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    energy = load_energy_config(args.energy_config)

    all_layer_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for variant in args.variants:
        for checkpoint_kind in args.checkpoint_kinds:
            checkpoint = checkpoint_for(args, variant, checkpoint_kind)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            rows, summary = run_one(variant, checkpoint_kind, checkpoint, args, energy)
            all_layer_rows.extend(rows)
            summaries.append(summary)

    summaries.sort(key=lambda item: (item["checkpoint_kind"], item["total_energy_pj"]))
    write_outputs(args.output_dir, all_layer_rows, summaries)


if __name__ == "__main__":
    main()
