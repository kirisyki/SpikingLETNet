#!/usr/bin/env python3
"""Estimate versioned core arithmetic energy for SpikingLETNet shallow variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from energy_accounting import (
    LayerModePolicy,
    git_commit,
    sha256_file,
    stable_sample_ids,
    strict_json_dump,
    validate_finite_nonnegative,
)


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
from model.SpikingLETNet_shallow_middle import SpikingLETNet_shallow_middle  # noqa: E402
from model.SpikingLETNet_shallow_small import SpikingLETNet_shallow_small  # noqa: E402
from quantization.int4_selfbuild import QLayer  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


VARIANT_CLASSES = {
    "small": SpikingLETNet_shallow_small,
    "middle": SpikingLETNet_shallow_middle,
    "max": SpikingLETNet_shallow_max,
}

DEFAULT_CONFIG = PROJECT_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
DEFAULT_POLICY = PROJECT_ROOT / "tools/energy_layer_policy_v2.yaml"

DEFAULT_FP_CHECKPOINTS = {
    "small": PROJECT_ROOT / "checkpoint/udd/SpikingLETNet_shallow_smallbs64gpu1_trainval20260611-234750/model_best.pth",
    "middle": PROJECT_ROOT / "checkpoint/udd/SpikingLETNet_shallow_middlebs64gpu1_trainval20260611-222604/model_best.pth",
    "max": PROJECT_ROOT / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth",
}

DEFAULT_QAT_CHECKPOINTS = {
    "small": PROJECT_ROOT / "QAT_checkpoint/udd/SpikingLETNet_shallow_smallbs64gpu1_trainval20260613-200147/model_q_best_complete.pt",
    "middle": PROJECT_ROOT / "QAT_checkpoint/udd/SpikingLETNet_shallow_middlebs96gpu1_trainval20260613-155726/model_q_best_complete.pt",
    "max": PROJECT_ROOT / "QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260613-170955/model_q_best_complete.pt",
}

DEFAULT_ENERGY = {
    "fp": {"mac": 1.0, "ac": 1.0, "mem_read": 0.0, "mem_write": 0.0, "neuron_update": 0.0},
    "int4": {"mac": 1.0, "ac": 1.0, "mem_read": 0.0, "mem_write": 0.0, "neuron_update": 0.0},
}

EXCLUDED_OPS = [
    "bias",
    "batch_norm",
    "layer_norm",
    "softmax",
    "pooling",
    "qif_update_reset",
    "shuffle_concat_reshape_repeat",
    "bilinear_interpolation",
]


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
        image_path, _ = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        image_tensor = torch.from_numpy(np.array(image, dtype="float32") / 255.0).permute(2, 0, 1)
        return (image_tensor - self.mean) / self.std


@dataclass
class LayerStats:
    layer: str
    op_type: str
    precision: str
    mode: str
    classification_source: str
    calls: int = 0
    input_elems: int = 0
    input_neuron_sites: float = 0.0
    input_firing_rate_sum: float = 0.0
    input_positive_sum: float = 0.0
    input_nonzero: int = 0
    input_min: Optional[float] = None
    input_max: Optional[float] = None
    integer_like_calls: int = 0
    dense_macs_total: float = 0.0
    dense_macs_one_step_equivalent: float = 0.0
    dense_macs_charged: float = 0.0
    sop_total: float = 0.0
    ac_charged: float = 0.0
    core_energy_pj: float = 0.0
    mem_energy_pj: float = 0.0
    neuron_energy_pj: float = 0.0
    classifications: Dict[str, int] = field(default_factory=dict)
    observed_input_shapes: List[List[int]] = field(default_factory=list)
    observed_output_shapes: List[List[int]] = field(default_factory=list)
    observed_time_slices: List[int] = field(default_factory=list)

    @property
    def total_energy_pj(self) -> float:
        return self.core_energy_pj + self.mem_energy_pj + self.neuron_energy_pj

    def add_classification(self, name: str) -> None:
        self.classifications[name] = self.classifications.get(name, 0) + 1

    def observe_shapes(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
        input_shape = list(map(int, input_tensor.shape))
        output_shape = list(map(int, output_tensor.shape))
        if input_shape not in self.observed_input_shapes:
            self.observed_input_shapes.append(input_shape)
        if output_shape not in self.observed_output_shapes:
            self.observed_output_shapes.append(output_shape)

    def observe_input(
        self, tensor: torch.Tensor, integer_eps: float, time_slices: int
    ) -> tuple[float, int, bool]:
        if time_slices <= 0:
            raise ValueError(f"time_slices must be positive, got {time_slices}")
        detached = tensor.detach()
        min_value = float(detached.min().item()) if detached.numel() else 0.0
        max_value = float(detached.max().item()) if detached.numel() else 0.0
        integer_like = is_integer_like(detached, integer_eps)
        self.input_min = min_value if self.input_min is None else min(self.input_min, min_value)
        self.input_max = max_value if self.input_max is None else max(self.input_max, max_value)
        self.integer_like_calls += int(integer_like)
        self.input_elems += detached.numel()
        self.input_neuron_sites += detached.numel() / time_slices
        if time_slices not in self.observed_time_slices:
            self.observed_time_slices.append(time_slices)
        positive_sum = float(detached.clamp(min=0).sum().item())
        self.input_positive_sum += positive_sum
        self.input_firing_rate_sum += positive_sum / time_slices
        self.input_nonzero += int(torch.count_nonzero(detached).item())
        return min_value, max_value, integer_like

    def as_row(self, variant: str, checkpoint_kind: str, processed_images: int, timestep: int) -> Dict[str, Any]:
        if processed_images <= 0:
            raise ValueError("processed_images must be positive")
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        mean_spike_count = self.input_positive_sum / self.input_elems if self.input_elems else None
        configured_firing_rate = mean_spike_count / timestep if mean_spike_count is not None else None
        sop_density = (
            self.input_firing_rate_sum / self.input_elems if self.input_elems else None
        )
        nonzero_ratio = self.input_nonzero / self.input_elems if self.input_elems else None
        row = {
            "variant": variant,
            "checkpoint_kind": checkpoint_kind,
            "precision": self.precision,
            "layer": self.layer,
            "op_type": self.op_type,
            "mode": self.mode,
            "classification_source": self.classification_source,
            "calls": self.calls,
            "classifications": dict(sorted(self.classifications.items())),
            "observed_input_shapes": self.observed_input_shapes,
            "observed_output_shapes": self.observed_output_shapes,
            "observed_time_slices": self.observed_time_slices,
            "configured_timestep": timestep,
            "input_elems": self.input_elems,
            "input_neuron_sites": self.input_neuron_sites,
            "input_firing_rate_sum": self.input_firing_rate_sum,
            "input_positive_sum": self.input_positive_sum,
            "input_nonzero": self.input_nonzero,
            "input_min": self.input_min,
            "input_max": self.input_max,
            "integer_like_calls": self.integer_like_calls,
            "mean_spike_count_per_input": mean_spike_count if self.mode == "spike" else None,
            "configured_firing_rate": configured_firing_rate if self.mode == "spike" else None,
            "sop_density_per_executed_mac": sop_density if self.mode == "spike" else None,
            "nonzero_ratio": nonzero_ratio,
            "dense_macs_total": self.dense_macs_total,
            "dense_macs_executed": self.dense_macs_total,
            "dense_macs_one_step_equivalent": self.dense_macs_one_step_equivalent,
            "dense_macs_total_per_image": self.dense_macs_total / processed_images,
            "dense_macs_executed_per_image": self.dense_macs_total / processed_images,
            "dense_macs_one_step_equivalent_per_image": self.dense_macs_one_step_equivalent / processed_images,
            "dense_macs_charged": self.dense_macs_charged,
            "dense_macs_charged_per_image": self.dense_macs_charged / processed_images,
            "sop_total": self.sop_total,
            "sop_per_image": self.sop_total / processed_images,
            "ac_charged": self.ac_charged,
            "core_energy_pj": self.core_energy_pj,
            "core_energy_pj_per_image": self.core_energy_pj / processed_images,
            "mem_energy_pj": self.mem_energy_pj,
            "neuron_energy_pj": self.neuron_energy_pj,
            "total_energy_pj": self.total_energy_pj,
            "total_energy_pj_per_image": self.total_energy_pj / processed_images,
        }
        return row


def parse_input_size(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("input size must be HEIGHT,WIDTH")
    height, width = map(int, parts)
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("input dimensions must be positive")
    return height, width


def load_energy_config(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    energy = json.loads(json.dumps(DEFAULT_ENERGY))
    if path is not None:
        loaded = json.loads(path.read_text()) if path.suffix.lower() == ".json" else __import__("yaml").safe_load(path.read_text())
        if "energy_pj" in loaded:
            loaded = loaded["energy_pj"]
        for precision, values in loaded.items():
            energy.setdefault(str(precision), {})
            for key, value in values.items():
                energy[str(precision)][str(key)] = float(value)
    for precision in ("fp", "int4"):
        missing = set(DEFAULT_ENERGY[precision]) - set(energy.get(precision, {}))
        if missing:
            raise ValueError(f"energy config missing {precision} keys: {sorted(missing)}")
    validate_finite_nonnegative(energy)
    return energy


def first_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def conv_macs(module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor) -> int:
    if isinstance(module, nn.ConvTranspose2d):
        return int(input_tensor.numel() * (module.out_channels // module.groups) * product(module.kernel_size))
    if isinstance(module, nn.Conv2d):
        return int(output.numel() * (module.in_channels // module.groups) * product(module.kernel_size))
    if isinstance(module, nn.Conv1d):
        return int(output.numel() * (module.in_channels // module.groups) * product(module.kernel_size))
    if isinstance(module, nn.Linear):
        return int(output.numel() * module.in_features)
    raise TypeError(f"unsupported compute module: {type(module)}")


def module_op_type(module: nn.Module) -> str:
    if isinstance(module, nn.ConvTranspose2d):
        return "conv_transpose2d"
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    if isinstance(module, nn.Conv1d):
        return "conv1d"
    if isinstance(module, nn.Linear):
        return "linear"
    return module.__class__.__name__.lower()


def unwrap_compute_module(module: nn.Module) -> nn.Module:
    return module.layer if isinstance(module, QLayer) else module


def observed_time_slices(module: nn.Module, input_tensor: torch.Tensor) -> int:
    """Return an explicit leading time axis; folded batch axes remain one slice."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and input_tensor.dim() == 5:
        return int(input_tensor.shape[0])
    if isinstance(module, nn.Conv1d) and input_tensor.dim() == 4:
        return int(input_tensor.shape[0])
    return 1


def supported_compute_module(module: nn.Module) -> bool:
    if isinstance(module, QLayer):
        return isinstance(module.layer, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear))


def precision_for_module(module: nn.Module) -> str:
    if isinstance(module, QLayer) and bool(module.quant) and bool(module.activation_quant):
        if int(module.k) != 4:
            raise ValueError(f"no energy parameter configured for QLayer bit width {module.k}: {module.name}")
        return "int4"
    return "fp"


def is_integer_like(tensor: torch.Tensor, eps: float) -> bool:
    if tensor.numel() == 0:
        return False
    if not torch.is_floating_point(tensor):
        return True
    return bool(torch.max(torch.abs(tensor - torch.round(tensor))).item() <= eps)


def legacy_classify_input(layer_name: str, module: nn.Module, tensor: torch.Tensor, timestep: int, eps: float) -> str:
    raw_module = unwrap_compute_module(module)
    if isinstance(raw_module, (nn.Linear, nn.Conv1d)) or "transformer" in layer_name or tensor.numel() == 0:
        return "dense"
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    return "spike" if min_value >= -eps and max_value <= timestep + eps and is_integer_like(tensor, eps) else "dense"


def attention_matmul_macs(module: nn.Module, input_tensor: torch.Tensor) -> int:
    if input_tensor.dim() != 3:
        raise ValueError(f"EffAttention input must be [B,N,D], got {tuple(input_tensor.shape)}")
    batch, tokens, _ = map(int, input_tensor.shape)
    reduce_module = unwrap_compute_module(module.reduce)
    if not isinstance(reduce_module, nn.Linear):
        raise TypeError(
            f"EffAttention.reduce must resolve to Linear, got {type(reduce_module).__name__}"
        )
    reduced_dim = int(reduce_module.out_features)
    heads = int(module.num_heads)
    if reduced_dim % heads:
        raise ValueError(f"attention reduced dimension {reduced_dim} is not divisible by {heads} heads")
    chunk = math.ceil(tokens // 4)
    if chunk <= 0:
        raise ValueError(f"attention token count must be at least 4, got {tokens}")
    sizes = [min(chunk, tokens - offset) for offset in range(0, tokens, chunk)]
    head_dim = reduced_dim // heads
    return int(2 * batch * heads * head_dim * sum(size * size for size in sizes))


class EnergyHookCollector:
    def __init__(
        self,
        model: nn.Module,
        variant: str,
        energy: Mapping[str, Mapping[str, float]],
        timestep: int,
        integer_eps: float,
        include_extended: bool,
        accounting_mode: str,
        classification_mode: str,
        policy: Optional[LayerModePolicy],
    ) -> None:
        self.variant = variant
        self.energy = energy
        self.timestep = timestep
        self.integer_eps = integer_eps
        self.include_extended = include_extended
        self.accounting_mode = accounting_mode
        self.classification_mode = classification_mode
        self.policy = policy
        self.stats: Dict[str, LayerStats] = {}
        self.handles: List[Any] = []
        self.register(model)

    def configured_mode(self, name: str, op_type: str) -> tuple[str, str]:
        if self.accounting_mode == "all-dense":
            return "dense", "accounting_mode:all-dense"
        if self.classification_mode == "legacy-auto":
            return "dynamic", "legacy-auto"
        if self.policy is None:
            raise ValueError("semantic-policy classification requires a layer policy")
        mode, rule_id = self.policy.classify(name, self.variant, op_type)
        return mode, f"policy:{rule_id}"

    def register(self, model: nn.Module) -> None:
        qlayer_child_names = {
            f"{name}.layer" for name, module in model.named_modules() if name and isinstance(module, QLayer)
        }
        for name, module in model.named_modules():
            if not name or not supported_compute_module(module) or name in qlayer_child_names:
                continue
            raw_module = unwrap_compute_module(module)
            op_type = module_op_type(raw_module)
            mode, source = self.configured_mode(name, op_type)
            self.stats[name] = LayerStats(
                layer=name,
                op_type=op_type,
                precision=precision_for_module(module),
                mode=mode,
                classification_source=source,
            )
            self.handles.append(module.register_forward_hook(self.make_compute_hook(name, module)))

        for name, module in model.named_modules():
            if name and module.__class__.__name__ == "EffAttention":
                stats_name = f"{name}.matmul"
                self.stats[stats_name] = LayerStats(
                    layer=stats_name,
                    op_type="attention_matmul",
                    precision="fp",
                    mode="dense",
                    classification_source="special:EffAttention",
                )
                self.handles.append(module.register_forward_hook(self.make_attention_hook(stats_name, module)))

    def make_compute_hook(self, name: str, module: nn.Module):
        def hook(_module: nn.Module, inputs: Any, output: Any) -> None:
            input_tensor = first_tensor(inputs)
            output_tensor = first_tensor(output)
            if input_tensor is None or output_tensor is None:
                return
            stats = self.stats[name]
            raw_module = unwrap_compute_module(module)
            stats.observe_shapes(input_tensor, output_tensor)
            time_slices = observed_time_slices(raw_module, input_tensor)
            dense_macs = float(conv_macs(raw_module, input_tensor, output_tensor))
            min_value, max_value, integer_like = stats.observe_input(input_tensor, self.integer_eps, time_slices)
            mode = stats.mode
            if mode == "dynamic":
                mode = legacy_classify_input(name, module, input_tensor.detach(), self.timestep, self.integer_eps)
            elif mode == "spike" and (min_value < -self.integer_eps or not integer_like):
                raise ValueError(
                    f"spike-count policy violation at {self.variant}:{name}: "
                    f"min={min_value}, max={max_value}, integer_like={integer_like}"
                )
            stats.calls += 1
            stats.add_classification(mode)
            stats.dense_macs_total += dense_macs
            stats.dense_macs_one_step_equivalent += dense_macs / time_slices
            params = self.energy[stats.precision]
            if mode == "spike":
                mean_spike_count = (
                    float(input_tensor.detach().clamp(min=0).sum().item()) / input_tensor.numel()
                )
                dense_macs_one_step = dense_macs / time_slices
                firing_rate_for_executed_tensor = mean_spike_count / time_slices
                sop_from_rate = dense_macs * firing_rate_for_executed_tensor
                sop_from_count = dense_macs_one_step * mean_spike_count
                if not math.isclose(sop_from_rate, sop_from_count, rel_tol=1e-12, abs_tol=1e-6):
                    raise AssertionError(
                        f"inconsistent SOP definitions at {self.variant}:{name}: "
                        f"rate={sop_from_rate} count={sop_from_count}"
                    )
                sop = sop_from_count
                stats.sop_total += sop
                stats.ac_charged += sop
                stats.core_energy_pj += sop * params["ac"]
            else:
                stats.dense_macs_charged += dense_macs
                stats.core_energy_pj += dense_macs * params["mac"]
            if self.include_extended:
                reads = float(input_tensor.numel())
                writes = float(output_tensor.numel())
                stats.mem_energy_pj += reads * params["mem_read"] + writes * params["mem_write"]
                stats.neuron_energy_pj += writes * params["neuron_update"]

        return hook

    def make_attention_hook(self, name: str, module: nn.Module):
        def hook(_module: nn.Module, inputs: Any, output: Any) -> None:
            input_tensor = first_tensor(inputs)
            output_tensor = first_tensor(output)
            if input_tensor is None or output_tensor is None:
                return
            stats = self.stats[name]
            stats.observe_shapes(input_tensor, output_tensor)
            dense_macs = float(attention_matmul_macs(module, input_tensor))
            stats.observe_input(input_tensor, self.integer_eps, time_slices=1)
            stats.calls += 1
            stats.add_classification("dense")
            stats.dense_macs_total += dense_macs
            stats.dense_macs_one_step_equivalent += dense_macs
            stats.dense_macs_charged += dense_macs
            stats.core_energy_pj += dense_macs * self.energy["fp"]["mac"]

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
        raise RuntimeError(
            f"checkpoint mismatch for {variant} fp: missing_non_bn={non_bn_missing[:10]} "
            f"unexpected={unexpected[:10]}"
        )
    model.to(device).eval()
    return model


def load_qat_model(variant: str, checkpoint: Path, device: torch.device) -> nn.Module:
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = VARIANT_CLASSES[variant].__name__
    if model.__class__.__name__ != expected:
        raise RuntimeError(f"QAT checkpoint class mismatch: expected {expected}, got {model.__class__.__name__}")
    functional.set_step_mode(model, "m")
    model.to(device).eval()
    return model


def quantization_inventory(model: nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, QLayer):
            rows.append(
                {
                    "layer": name,
                    "bits": int(module.k),
                    "weight_quant": bool(module.quant),
                    "activation_quant": bool(module.activation_quant),
                    "activation_quant_mode": str(module.activation_quant_mode),
                }
            )
    return rows


def make_loader(args: argparse.Namespace) -> tuple[Iterable[torch.Tensor], list[str]]:
    if args.synthetic:
        height, width = args.input_size
        batches = args.max_batches if args.max_batches > 0 else 1
        tensors = [torch.randn(args.batch_size, 3, height, width) for _ in range(batches)]
        ids = [f"synthetic:{batch}:{sample}" for batch in range(batches) for sample in range(args.batch_size)]
        return tensors, ids
    dataset = UDDImageDataset(args.split_file, args.input_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    return loader, [sample[0] for sample in dataset.samples]


def repeat_timesteps(images: torch.Tensor, timestep: int) -> torch.Tensor:
    return images.unsqueeze(0).repeat(timestep, 1, 1, 1, 1)


def run_one(
    variant: str,
    checkpoint_kind: str,
    checkpoint: Path,
    args: argparse.Namespace,
    energy: Mapping[str, Mapping[str, float]],
    policy: Optional[LayerModePolicy],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.device)
    model = (
        load_fp_model(variant, args.config, checkpoint, args.classes, device)
        if checkpoint_kind == "fp"
        else load_qat_model(variant, checkpoint, device)
    )
    q_inventory = quantization_inventory(model)
    collector = EnergyHookCollector(
        model=model,
        variant=variant,
        energy=energy,
        timestep=args.timestep,
        integer_eps=args.integer_eps,
        include_extended=args.include_extended,
        accounting_mode=args.accounting_mode,
        classification_mode=args.classification_mode,
        policy=policy,
    )
    loader, sample_paths = make_loader(args)
    processed_batches = 0
    processed_images = 0
    try:
        with torch.no_grad():
            for batch_index, images in enumerate(loader):
                if args.max_batches > 0 and batch_index >= args.max_batches:
                    break
                images = images.to(device)
                model(repeat_timesteps(images, args.timestep))
                functional.reset_net(model)
                processed_batches += 1
                processed_images += int(images.shape[0])
                print(f"{variant}/{checkpoint_kind}: processed batch {processed_batches}, images={processed_images}")
    finally:
        collector.remove()
    if processed_images <= 0:
        raise RuntimeError(f"no images processed for {variant}/{checkpoint_kind}")
    rows = [
        stats.as_row(variant, checkpoint_kind, processed_images, args.timestep)
        for stats in collector.stats.values()
    ]
    summary = summarize_rows(
        variant,
        checkpoint_kind,
        checkpoint,
        processed_batches,
        processed_images,
        rows,
        sample_paths[:processed_images],
        q_inventory,
        args,
        energy,
        policy,
    )
    return rows, summary


def summarize_rows(
    variant: str,
    checkpoint_kind: str,
    checkpoint: Path,
    processed_batches: int,
    processed_images: int,
    rows: Sequence[Mapping[str, Any]],
    sample_paths: Sequence[str],
    q_inventory: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    energy: Mapping[str, Mapping[str, float]],
    policy: Optional[LayerModePolicy],
) -> dict[str, Any]:
    metric_names = [
        "dense_macs_total",
        "dense_macs_charged",
        "sop_total",
        "ac_charged",
        "core_energy_pj",
        "mem_energy_pj",
        "neuron_energy_pj",
        "total_energy_pj",
    ]
    totals = {name: sum(float(row[name]) for row in rows) for name in metric_names}
    per_image = {f"{name}_per_image": value / processed_images for name, value in totals.items()}
    spike_rows = [row for row in rows if row["mode"] == "spike" and int(row["input_elems"]) > 0]
    spike_inputs = sum(int(row["input_elems"]) for row in spike_rows)
    spike_sum = sum(float(row["input_positive_sum"]) for row in spike_rows)
    spike_firing_rate_sum = sum(float(row["input_firing_rate_sum"]) for row in spike_rows)
    precision_counts: Dict[str, int] = {}
    for row in rows:
        if int(row["calls"]) > 0:
            precision_counts[str(row["precision"])] = precision_counts.get(str(row["precision"]), 0) + 1
    return {
        "schema_version": 2,
        "variant": variant,
        "checkpoint_kind": checkpoint_kind,
        "precision": "fp" if checkpoint_kind == "fp" else "mixed_fp_int4",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "split_file": None if args.synthetic else str(args.split_file),
        "split_sha256": None if args.synthetic else sha256_file(args.split_file),
        "synthetic": bool(args.synthetic),
        "processed_batches": processed_batches,
        "processed_images": processed_images,
        "batch_size": args.batch_size,
        "timestep": args.timestep,
        "input_size": list(args.input_size),
        "accounting_mode": args.accounting_mode,
        "classification_mode": args.classification_mode,
        "sop_method": "qif_cumulative_spike_count_times_one_step_dense_macs_v2",
        "spike_value_semantics": "qif_cumulative_count_0_to_T",
        "layer_policy_version": policy.version if policy is not None else None,
        "layer_policy_sha256": policy.sha256 if policy is not None else None,
        "count_convention": "standard_dense_kernel_ops_including_padding_convention",
        "energy_pj": energy,
        "precision_layer_counts": precision_counts,
        "quantization_inventory": list(q_inventory),
        "sample_ids": stable_sample_ids(sample_paths),
        "mean_spike_count_per_input_across_spike_layers": (
            spike_sum / spike_inputs if spike_inputs else None
        ),
        "configured_firing_rate_across_spike_layers": (
            spike_sum / spike_inputs / args.timestep if spike_inputs else None
        ),
        "sop_density_per_executed_mac_across_spike_layers": (
            spike_firing_rate_sum / spike_inputs if spike_inputs else None
        ),
        "excluded_ops": EXCLUDED_OPS,
        **totals,
        **per_image,
    }


def csv_safe(value: Any) -> Any:
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key)) for key in fieldnames})


def build_manifest(args: argparse.Namespace, summaries: Sequence[Mapping[str, Any]], policy: Optional[LayerModePolicy]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "git_commit": git_commit(PROJECT_ROOT),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "device": args.device,
        "accounting_mode": args.accounting_mode,
        "classification_mode": args.classification_mode,
        "sop_method": "qif_cumulative_spike_count_times_one_step_dense_macs_v2",
        "spike_value_semantics": "qif_cumulative_count_0_to_T",
        "layer_policy": None
        if policy is None
        else {"path": str(policy.source), "version": policy.version, "sha256": policy.sha256},
        "energy_config": None
        if args.energy_config is None
        else {"path": str(args.energy_config), "sha256": sha256_file(args.energy_config)},
        "summary_keys": [f"{item['variant']}:{item['checkpoint_kind']}" for item in summaries],
        "excluded_ops": EXCLUDED_OPS,
    }


def write_outputs(
    output_dir: Path,
    layer_rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "layer_energy.csv", layer_rows)
    write_csv(output_dir / "summary_energy.csv", summaries)
    strict_json_dump({"schema_version": 2, "summaries": summaries}, output_dir / "summary_energy.json")
    strict_json_dump(manifest, output_dir / "run_manifest.json")
    for name in ("layer_energy.csv", "summary_energy.csv", "summary_energy.json", "run_manifest.json"):
        print(f"wrote {output_dir / name}")


def checkpoint_for(args: argparse.Namespace, variant: str, checkpoint_kind: str) -> Path:
    override = getattr(args, f"{variant}_{checkpoint_kind}_checkpoint")
    if override:
        return override
    return DEFAULT_FP_CHECKPOINTS[variant] if checkpoint_kind == "fp" else DEFAULT_QAT_CHECKPOINTS[variant]


def add_checkpoint_override_args(parser: argparse.ArgumentParser) -> None:
    for variant in VARIANT_CLASSES:
        parser.add_argument(f"--{variant}-fp-checkpoint", type=Path, default=None)
        parser.add_argument(f"--{variant}-qat-checkpoint", type=Path, default=None)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_CLASSES), choices=list(VARIANT_CLASSES))
    parser.add_argument("--checkpoint-kinds", nargs="+", default=["fp", "qat"], choices=["fp", "qat"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "energy/theoretical/v2/snn_t8_20_images")
    parser.add_argument("--energy-config", type=Path, default=None)
    parser.add_argument("--layer-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--accounting-mode", choices=["mixed", "all-dense"], default="mixed")
    parser.add_argument("--classification-mode", choices=["semantic-policy", "legacy-auto"], default="semantic-policy")
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=20, help="Use 0 to process the full split.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-size", type=parse_input_size, default=(400, 400))
    parser.add_argument("--timestep", type=int, default=8)
    parser.add_argument("--integer-eps", type=float, default=1e-4)
    parser.add_argument("--include-extended", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    add_checkpoint_override_args(parser)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_batches < 0 or args.timestep <= 0 or args.integer_eps < 0:
        parser.error("batch-size and timestep must be positive; max-batches and integer-eps must be non-negative")
    if args.accounting_mode == "all-dense" and args.timestep != 1:
        parser.error("all-dense accounting is the T=1 QIF baseline and requires --timestep 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    energy = load_energy_config(args.energy_config)
    policy = LayerModePolicy.load(args.layer_policy) if args.classification_mode == "semantic-policy" else None
    all_layer_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for variant in args.variants:
        for checkpoint_kind in args.checkpoint_kinds:
            checkpoint = checkpoint_for(args, variant, checkpoint_kind)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            rows, summary = run_one(variant, checkpoint_kind, checkpoint, args, energy, policy)
            all_layer_rows.extend(rows)
            summaries.append(summary)
    summaries.sort(key=lambda item: (str(item["checkpoint_kind"]), str(item["variant"])))
    write_outputs(args.output_dir, all_layer_rows, summaries, build_manifest(args, summaries, policy))


if __name__ == "__main__":
    main()
