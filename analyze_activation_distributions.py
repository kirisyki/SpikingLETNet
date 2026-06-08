#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
import types
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.join(PROJECT_ROOT, "Network")
if NETWORK_DIR not in sys.path:
    sys.path.insert(0, NETWORK_DIR)

if "torchsummary" not in sys.modules:
    torchsummary_stub = types.ModuleType("torchsummary")
    torchsummary_stub.summary = lambda *args, **kwargs: None
    sys.modules["torchsummary"] = torchsummary_stub

from builders.model_builder import build_model  # noqa: E402
from quantization.int4_selfbuild import QLayer, quantize_model  # noqa: E402
from spikingjelly.activation_based import functional, layer  # noqa: E402
from model.module.neuron import QIFNode  # noqa: E402


DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260606-171257/model_best.pth",
)
DEFAULT_CONFIG = os.path.join(
    PROJECT_ROOT,
    "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260606-171257/1.3M.yaml",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "act_distributions")
DEFAULT_UDD_ROOT = "/root/autodl-tmp/UDD/UDD6/preprocessed"


class UDDActivationDataset(Dataset):
    def __init__(self, txt_file, input_size=(400, 400)):
        self.samples = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], parts[1]))
        self.input_size = input_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        image = image.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        mask = mask.resize((self.input_size[1], self.input_size[0]), Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = (image - self.mean) / self.std
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


class RunningActivationStats:
    def __init__(self, max_samples=200000, samples_per_batch=4096, zero_eps=1e-8):
        self.count = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.min_value = None
        self.max_value = None
        self.zero_count = 0
        self.max_samples = max_samples
        self.samples_per_batch = samples_per_batch
        self.zero_eps = zero_eps
        self.samples = []

    def update(self, tensor):
        if tensor.numel() == 0:
            return

        x = tensor.detach()
        if not torch.is_floating_point(x):
            x = x.float()

        flat = x.reshape(-1)
        flat64 = flat.double()
        numel = flat.numel()

        self.count += numel
        self.sum += flat64.sum().item()
        self.sq_sum += flat64.square().sum().item()
        self.zero_count += torch.count_nonzero(flat.abs() <= self.zero_eps).item()

        current_min = flat.min().item()
        current_max = flat.max().item()
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

        remaining = self.max_samples - self.sample_count
        if remaining <= 0:
            return

        take = min(self.samples_per_batch, remaining, numel)
        if take <= 0:
            return

        if numel <= take:
            sampled = flat.detach().float().cpu()
        else:
            indices = torch.randint(0, numel, (take,), device=flat.device)
            sampled = flat[indices].detach().float().cpu()
        self.samples.append(sampled)

    @property
    def sample_count(self):
        return sum(s.numel() for s in self.samples)

    def as_dict(self):
        mean = self.sum / self.count if self.count else float("nan")
        variance = self.sq_sum / self.count - mean * mean if self.count else float("nan")
        variance = max(variance, 0.0) if self.count else variance
        return {
            "count": self.count,
            "mean": mean,
            "variance": variance,
            "std": variance ** 0.5 if self.count else float("nan"),
            "min": self.min_value,
            "max": self.max_value,
            "zero_ratio": self.zero_count / self.count if self.count else float("nan"),
            "sample_count": self.sample_count,
        }

    def sample_tensor(self):
        if not self.samples:
            return torch.empty(0)
        return torch.cat(self.samples)


class ActivationCollector:
    def __init__(
        self,
        model,
        scope,
        hook_target,
        include,
        exclude,
        max_samples,
        samples_per_batch,
        zero_eps,
        global_stats=None,
    ):
        self.stats = OrderedDict()
        self.handles = []
        self.scope = scope
        self.hook_target = hook_target
        self.include = include or []
        self.exclude = exclude or []
        self.max_samples = max_samples
        self.samples_per_batch = samples_per_batch
        self.zero_eps = zero_eps
        self.global_stats = global_stats
        self.register(model)

    def register(self, model):
        for name, module in model.named_modules():
            if not name or not self.should_track(name, module):
                continue
            self.stats[name] = RunningActivationStats(
                max_samples=self.max_samples,
                samples_per_batch=self.samples_per_batch,
                zero_eps=self.zero_eps,
            )
            self.handles.append(module.register_forward_hook(self.make_hook(name)))

    def should_track(self, name, module):
        if self.include and not any(re.search(pattern, name) for pattern in self.include):
            return False
        if self.exclude and any(re.search(pattern, name) for pattern in self.exclude):
            return False

        if self.scope == "leaf":
            return len(list(module.children())) == 0
        if self.scope == "qlayer":
            return isinstance(module, QLayer)
        if self.scope == "qif":
            return isinstance(module, QIFNode)
        if self.scope == "conv":
            return isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, layer.Conv2d, layer.ConvTranspose2d))
        if self.scope == "conv_bn_qif":
            return isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.ConvTranspose2d,
                    nn.BatchNorm2d,
                    layer.Conv2d,
                    layer.ConvTranspose2d,
                    layer.BatchNorm2d,
                    QIFNode,
                ),
            )
        if self.scope == "all":
            return True
        raise ValueError(f"Unknown activation scope: {self.scope}")

    def make_hook(self, name):
        def hook(_module, inputs, output):
            tensors = inputs if self.hook_target == "input" else output
            for tensor in iter_tensors(tensors):
                self.stats[name].update(tensor)
                if self.global_stats is not None:
                    self.global_stats.update(tensor)
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")


def parse_input_size(value):
    height, width = value.split(",")
    return int(height), int(width)


def build_loader(args):
    if args.dataset != "udd":
        raise NotImplementedError("This standalone script currently supports --dataset udd.")

    split_file = args.split_file or os.path.join(args.udd_root, "val_patches.txt")
    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"UDD split file not found: {split_file}")

    dataset = UDDActivationDataset(split_file, input_size=parse_input_size(args.input_size))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def write_summary(output_dir, collector, global_stats, args):
    rows = []
    for name, stats in collector.stats.items():
        row = {"layer": name}
        row.update(stats.as_dict())
        rows.append(row)

    csv_path = os.path.join(output_dir, "activation_summary.csv")
    fieldnames = ["layer", "count", "mean", "variance", "std", "min", "max", "zero_ratio", "sample_count"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "model": args.model,
        "dataset": args.dataset,
        "scope": args.scope,
        "hook_target": args.hook_target,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "timestep": args.T,
        "global": global_stats.as_dict(),
        "layers": rows,
    }
    json_path = os.path.join(output_dir, "activation_summary.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(output_dir, "activation_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"scope: {args.scope}\n")
        f.write(f"hook_target: {args.hook_target}\n")
        f.write(f"processed_batches: {args.processed_batches}\n")
        f.write("\n[global]\n")
        for key, value in global_stats.as_dict().items():
            f.write(f"{key}: {value}\n")
        f.write("\n[layer mean and variance]\n")
        for row in rows:
            f.write(f"{row['layer']}: mean={row['mean']}, variance={row['variance']}\n")

    return csv_path, json_path, txt_path


def plot_histogram(samples, title, save_path, bins, stats, value_label="activation"):
    if samples.numel() == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(samples.numpy(), bins=bins, density=True, color="#2563eb", alpha=0.78)
    plt.xlabel(f"{value_label} value")
    plt.ylabel("density")
    plt.title(f"{title}\nmean={stats['mean']:.6g}, var={stats['variance']:.6g}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_all(output_dir, collector, global_stats, args):
    hist_dir = os.path.join(output_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    for filename in os.listdir(hist_dir):
        if filename.endswith(".png"):
            os.remove(os.path.join(hist_dir, filename))

    plot_histogram(
        global_stats.sample_tensor(),
        f"all tracked {args.hook_target}s",
        os.path.join(output_dir, "global_activation_distribution.png"),
        args.bins,
        global_stats.as_dict(),
        value_label=args.hook_target,
    )

    for index, (name, stats) in enumerate(collector.stats.items()):
        if stats.count == 0:
            continue
        filename = f"{index:03d}_{safe_name(name)}.png"
        plot_histogram(
            stats.sample_tensor(),
            name,
            os.path.join(hist_dir, filename),
            args.bins,
            stats.as_dict(),
            value_label=args.hook_target,
        )

    names = []
    means = []
    variances = []
    for name, stats in collector.stats.items():
        summary = stats.as_dict()
        if stats.count == 0:
            continue
        names.append(name)
        means.append(summary["mean"])
        variances.append(summary["variance"])

    if not names:
        return

    x = range(len(names))
    plt.figure(figsize=(max(10, len(names) * 0.22), 5))
    plt.plot(x, means, marker=".", linewidth=1)
    plt.xticks(x, names, rotation=90, fontsize=6)
    plt.ylabel("mean")
    plt.title(f"QLayer {args.hook_target.capitalize()} Mean by Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_activation_means.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(max(10, len(names) * 0.22), 5))
    plt.plot(x, variances, marker=".", linewidth=1, color="#dc2626")
    plt.xticks(x, names, rotation=90, fontsize=6)
    plt.ylabel("variance")
    plt.title(f"QLayer {args.hook_target.capitalize()} Variance by Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_activation_variances.png"), dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and visualize model activation distributions.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="SpikingLETNet_shallow_max")
    parser.add_argument("--dataset", default="udd")
    parser.add_argument("--udd_root", default=DEFAULT_UDD_ROOT)
    parser.add_argument("--split_file", default=None, help="Optional path to train_patches.txt or val_patches.txt.")
    parser.add_argument("--input_size", default="400,400", help="Image resize as height,width.")
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=20, help="Set <=0 to process the whole loader.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--scope",
        default="qlayer",
        choices=["qlayer", "leaf", "qif", "conv", "conv_bn_qif", "all"],
        help="Which modules to hook for activation statistics.",
    )
    parser.add_argument(
        "--hook_target",
        default="input",
        choices=["input", "output"],
        help="Collect values before or after the hooked module computes.",
    )
    parser.add_argument("--quantize_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quant_bits", type=int, default=4)
    parser.add_argument("--quant_start_layer", type=int, default=0)
    parser.add_argument(
        "--activation_quant_mode",
        default="per_tensor",
        choices=["per_tensor", "per_image", "per_channel"],
    )
    parser.add_argument("--include", nargs="*", default=None, help="Optional regex filters for layer names.")
    parser.add_argument("--exclude", nargs="*", default=None, help="Optional regex filters for layer names.")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=200000)
    parser.add_argument("--samples_per_batch", type=int, default=4096)
    parser.add_argument("--zero_eps", type=float, default=1e-8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    model = build_model(args.model, num_classes=args.classes, config=args.config)
    functional.set_step_mode(model, step_mode="m")
    load_checkpoint(model, args.checkpoint, device="cpu")
    if args.quantize_model:
        model = quantize_model(
            model,
            k=args.quant_bits,
            inplace=False,
            quant=True,
            activation_quant=True,
            quant_start_layer=args.quant_start_layer,
            activation_quant_mode=args.activation_quant_mode,
        )
        functional.set_step_mode(model, step_mode="m")
    model.to(device)
    model.eval()

    loader = build_loader(args)
    global_stats = RunningActivationStats(
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        zero_eps=args.zero_eps,
    )
    collector = ActivationCollector(
        model,
        scope=args.scope,
        hook_target=args.hook_target,
        include=args.include,
        exclude=args.exclude,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        zero_eps=args.zero_eps,
        global_stats=global_stats,
    )
    print(f"Registered hooks: {len(collector.handles)}")
    print(f"Output directory: {args.output_dir}")

    processed = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if args.max_batches > 0 and batch_index >= args.max_batches:
                    break

                images = batch[0].to(device)
                images = images.repeat(args.T, 1, 1, 1, 1)
                model(images)
                functional.reset_net(model)

                processed += 1
                print(f"processed batch {processed}/{len(loader)}")
    finally:
        collector.remove()

    args.processed_batches = processed
    csv_path, json_path, txt_path = write_summary(args.output_dir, collector, global_stats, args)
    plot_all(args.output_dir, collector, global_stats, args)

    print(f"Processed batches: {processed}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Summary TXT: {txt_path}")
    print(f"Histogram directory: {os.path.join(args.output_dir, 'histograms')}")


if __name__ == "__main__":
    main()
