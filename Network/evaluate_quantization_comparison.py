#!/usr/bin/env python3
"""Unified full-validation evaluation for QAD and three W4A4 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn


NETWORK_DIR = Path(__file__).resolve().parent
os.chdir(NETWORK_DIR)

from builders.dataset_builder import build_dataset_test  # noqa: E402
from builders.model_builder import build_model  # noqa: E402
from quantization.int4_selfbuild import QLayer, quantize_model  # noqa: E402
from quantization_comparison.checkpointing import (  # noqa: E402
    CHECKPOINT_FORMAT,
)
from quantization_comparison.layer_adapter import (  # noqa: E402
    repeat_time,
    temporal_average,
)
from quantization_comparison.model_factory import (  # noqa: E402
    build_quantized_model,
    quantized_layer_names,
)
from quantization_comparison.protocol import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_CONFIG,
    DEFAULT_FP_CHECKPOINT,
    DEFAULT_QAD_CHECKPOINT,
    DEFAULT_RESULT_DIR,
    ExperimentProtocol,
    sha256_file,
    validate_protected_hashes,
)
from spikingjelly.activation_based import functional  # noqa: E402


CLASS_NAMES = ("background", "facade", "road", "vegetation", "vehicle", "roof")
EXPECTED_VALIDATION_SAMPLES = 8478
HISTORICAL_QAD_MIOU = 0.619547
HISTORICAL_QAD_TOLERANCE = 1e-5
BASELINE_PATHS = {
    "ste": DEFAULT_CHECKPOINT_ROOT / "ste_qat_w4a4/checkpoint_best.pth",
    "lsq": DEFAULT_CHECKPOINT_ROOT / "lsq_qat_w4a4/checkpoint_best.pth",
    "ewgs": DEFAULT_CHECKPOINT_ROOT / "ewgs_qat_w4a4/checkpoint_best.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--ste-checkpoint", type=Path, default=BASELINE_PATHS["ste"])
    parser.add_argument("--lsq-checkpoint", type=Path, default=BASELINE_PATHS["lsq"])
    parser.add_argument("--ewgs-checkpoint", type=Path, default=BASELINE_PATHS["ewgs"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-val-iters", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class DeviceConfusionMatrix:
    def __init__(self, classes: int, device: torch.device):
        self.classes = classes
        self.matrix = torch.zeros(
            classes, classes, dtype=torch.int64, device=device
        )

    def update(self, labels: torch.Tensor, predictions: torch.Tensor) -> None:
        labels = labels.flatten().long()
        predictions = predictions.flatten().long()
        valid = (labels >= 0) & (labels < self.classes)
        index = self.classes * labels[valid] + predictions[valid]
        self.matrix += torch.bincount(
            index, minlength=self.classes * self.classes
        ).reshape(self.classes, self.classes)

    def result(self) -> tuple[float, list[float]]:
        matrix = self.matrix.double()
        union = matrix.sum(0) + matrix.sum(1) - matrix.diag()
        iou = matrix.diag() / union
        return float(torch.nanmean(iou).item()), [
            float(value) for value in iou.cpu().tolist()
        ]


def base_model() -> nn.Module:
    model = build_model(
        "SpikingLETNet_shallow_max",
        num_classes=6,
        config=str(DEFAULT_CONFIG),
    )
    functional.set_step_mode(model, step_mode="m")
    return model


def load_historical_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise KeyError(f"historical checkpoint has no 'model': {path}")
    return payload["model"], {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "epoch": payload.get("epoch"),
    }


def build_fp() -> tuple[nn.Module, dict[str, Any]]:
    model = base_model()
    state, metadata = load_historical_state(DEFAULT_FP_CHECKPOINT)
    model.load_state_dict(state, strict=True)
    return model, metadata


def build_qad() -> tuple[nn.Module, dict[str, Any]]:
    model = quantize_model(
        base_model(),
        k=4,
        inplace=False,
        quant=True,
        activation_quant=True,
        quant_start_layer=0,
        activation_quant_mode="per_tensor",
    )
    functional.set_step_mode(model, step_mode="m")
    state, metadata = load_historical_state(DEFAULT_QAD_CHECKPOINT)
    model.load_state_dict(state, strict=True)
    return model, metadata


def build_baseline(
    method: str, checkpoint_path: Path
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid comparison checkpoint: {checkpoint_path}")
    expected_protocol = ExperimentProtocol(method).to_dict()
    if payload.get("protocol") != expected_protocol:
        raise ValueError(f"protocol mismatch in {checkpoint_path}")
    model = build_quantized_model(method, base_model())
    model.load_state_dict(payload["model_state"], strict=True)
    functional.set_step_mode(model, step_mode="m")
    return model, {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "epoch": payload.get("epoch"),
        "best_miou_recorded": payload.get("best_miou"),
    }


class IntegerBypassAudit:
    """Observe the historical QLayer integer/+8 bypass without changing it."""

    def __init__(self, model: nn.Module):
        self.records: dict[str, dict[str, int]] = {}
        self.handles: list[Any] = []
        for module_name, module in model.named_modules():
            if not isinstance(module, QLayer):
                continue
            self.records[module_name] = {
                "calls": 0,
                "integer_bypass_calls": 0,
                "plus8_bypass_calls": 0,
                "plus8_values": 0,
            }

            def hook(
                target: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                name: str = module_name,
            ) -> None:
                del target
                tensor = inputs[0]
                if tensor.dim() in (3, 5):
                    tensor = tensor.flatten(0, 1)
                record = self.records[name]
                record["calls"] += 1
                integer_bypass = bool(
                    torch.all(tensor == tensor.round())
                    and torch.all(tensor >= -8)
                    and torch.all(tensor <= 8)
                )
                if integer_bypass:
                    record["integer_bypass_calls"] += 1
                    plus8_count = int((tensor == 8).sum().item())
                    if plus8_count:
                        record["plus8_bypass_calls"] += 1
                        record["plus8_values"] += plus8_count

            self.handles.append(module.register_forward_pre_hook(hook))

    def close(self) -> dict[str, Any]:
        for handle in self.handles:
            handle.remove()
        totals = {
            key: sum(record[key] for record in self.records.values())
            for key in (
                "calls",
                "integer_bypass_calls",
                "plus8_bypass_calls",
                "plus8_values",
            )
        }
        return {"totals": totals, "layers": self.records}


@torch.no_grad()
def evaluate_model(
    *,
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    limit: int | None,
    audit_bypass: bool,
) -> dict[str, Any]:
    model.to(device).eval()
    confusion = DeviceConfusionMatrix(6, device)
    audit = IntegerBypassAudit(model) if audit_bypass else None
    started = time.perf_counter()
    completed = 0
    for iteration, batch in enumerate(loader):
        if limit is not None and iteration >= limit:
            break
        images, labels = batch[:2]
        images = repeat_time(images.to(device, non_blocking=True), 1)
        labels = labels.long().to(device, non_blocking=True)
        output = temporal_average(model(images))
        confusion.update(labels, output.argmax(1))
        functional.reset_net(model)
        completed += 1
        if iteration == 0 or completed % 20 == 0:
            print(
                f"eval {completed}/{limit or len(loader)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    if completed == 0:
        raise RuntimeError("no validation batches evaluated")
    miou, per_class = confusion.result()
    result = {
        "miou": miou,
        "per_class_iou": {
            name: value for name, value in zip(CLASS_NAMES, per_class)
        },
        "evaluated_batches": completed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if audit is not None:
        result["integer_bypass_audit"] = audit.close()
    return result


def preregistered_decision(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qad = results["qad"]["miou"]
    ste = results["ste"]["miou"]
    advanced_best_name = max(("lsq", "ewgs"), key=lambda name: results[name]["miou"])
    advanced_best = results[advanced_best_name]["miou"]
    qad_minus_ste_points = (qad - ste) * 100.0
    qad_minus_advanced_points = (qad - advanced_best) * 100.0
    return {
        "qad_minus_ste_miou_points": qad_minus_ste_points,
        "effective_threshold_points": 1.0,
        "qad_effective": qad_minus_ste_points >= 1.0,
        "best_advanced_baseline": advanced_best_name,
        "qad_minus_best_advanced_miou_points": qad_minus_advanced_points,
        "competitive_tolerance_points": 0.5,
        "qad_competitive": qad_minus_advanced_points >= -0.5,
        "superiority_threshold_points": 0.5,
        "qad_superior": qad_minus_advanced_points >= 0.5,
    }


def write_csv(results: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "miou",
        *[f"iou_{name}" for name in CLASS_NAMES],
        "evaluated_batches",
        "elapsed_seconds",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in results.items():
            writer.writerow(
                {
                    "method": method,
                    "miou": result["miou"],
                    **{
                        f"iou_{name}": result["per_class_iou"][name]
                        for name in CLASS_NAMES
                    },
                    "evaluated_batches": result["evaluated_batches"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "checkpoint": result["checkpoint"]["path"],
                }
            )


def write_markdown(
    results: dict[str, dict[str, Any]],
    decision: dict[str, Any],
    path: Path,
) -> None:
    display_names = {
        "fp32": "FP32",
        "qad": "QAD W4A4",
        "ste": "STE-QAT W4A4",
        "lsq": "LSQ-QAT W4A4",
        "ewgs": "EWGS-QAT W4A4",
    }
    lines = [
        "# SpikingLETNet_shallow_max W4A4 量化训练对比结果",
        "",
        "所有模型使用同一 UDD `val_patches.txt`、400×400 预处理、"
        "T=1、batch size=20 和同一混淆矩阵实现重新评估。",
        "",
        "重要限制：历史 QAD 训练 127 epochs，而三个对比基线按用户缩短后的预算训练 16 epochs；",
        "因此结果不能排除训练预算差异的影响。",
        "",
        "| 方法 | mIoU | " + " | ".join(CLASS_NAMES) + " |",
        "|---|---:|" + "|".join(["---:"] * len(CLASS_NAMES)) + "|",
    ]
    for method, result in results.items():
        values = " | ".join(
            f"{result['per_class_iou'][name]:.6f}" for name in CLASS_NAMES
        )
        lines.append(
            f"| {display_names[method]} | {result['miou']:.6f} | {values} |"
        )
    lines.extend(
        [
            "",
            "## 预注册判据",
            "",
            f"- QAD − STE-QAT："
            f"{decision['qad_minus_ste_miou_points']:.4f} 个绝对 mIoU 点；"
            f"有效性判据（≥1.0）：`{decision['qad_effective']}`。",
            f"- 最佳高级基线：`{decision['best_advanced_baseline']}`；"
            f"QAD − 最佳高级基线："
            f"{decision['qad_minus_best_advanced_miou_points']:.4f} 点。",
            f"- 竞争性判据（不落后超过 0.5 点）："
            f"`{decision['qad_competitive']}`。",
            f"- 优越性判据（领先至少 0.5 点）："
            f"`{decision['qad_superior']}`。",
            "",
            "## 可复核信息",
            "",
            "- `comparison.json` 保存完整 checkpoint 哈希、逐类结果、"
            "历史整数激活旁路与 +8 命中审计。",
            "- 这里的结论只对应预注册的 seed=1234 单种子实验。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_protected_hashes()
    if args.smoke:
        if args.max_val_iters <= 0:
            raise ValueError("--max-val-iters must be positive in smoke mode")
    elif args.max_val_iters != 1:
        raise ValueError("--max-val-iters is smoke-only")

    checkpoint_paths = {
        "ste": args.ste_checkpoint.resolve(),
        "lsq": args.lsq_checkpoint.resolve(),
        "ewgs": args.ewgs_checkpoint.resolve(),
    }
    required = [
        DEFAULT_CONFIG,
        DEFAULT_FP_CHECKPOINT,
        DEFAULT_QAD_CHECKPOINT,
        *checkpoint_paths.values(),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for unified evaluation")
    device = torch.device("cuda")
    _, loader = build_dataset_test(
        "udd",
        num_workers=6,
        none_gt=False,
        batch_size=20,
    )
    if len(loader.dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(
            f"expected {EXPECTED_VALIDATION_SAMPLES} validation samples, "
            f"observed {len(loader.dataset)}"
        )

    builders: list[
        tuple[str, Callable[[], tuple[nn.Module, dict[str, Any]]], bool]
    ] = [
        ("fp32", build_fp, False),
        ("qad", build_qad, True),
        ("ste", lambda: build_baseline("ste", checkpoint_paths["ste"]), True),
        ("lsq", lambda: build_baseline("lsq", checkpoint_paths["lsq"]), False),
        ("ewgs", lambda: build_baseline("ewgs", checkpoint_paths["ewgs"]), False),
    ]
    results: dict[str, dict[str, Any]] = {}
    limit = args.max_val_iters if args.smoke else None
    for name, builder, audit_bypass in builders:
        print(f"evaluating {name}", flush=True)
        model, checkpoint = builder()
        if name != "fp32" and len(quantized_layer_names(model)) != 71:
            raise RuntimeError(f"{name} does not contain 71 quantized layers")
        metrics = evaluate_model(
            model=model,
            loader=loader,
            device=device,
            limit=limit,
            audit_bypass=audit_bypass,
        )
        results[name] = {**metrics, "checkpoint": checkpoint}
        del model
        torch.cuda.empty_cache()

    if not args.smoke:
        qad_error = abs(results["qad"]["miou"] - HISTORICAL_QAD_MIOU)
        if qad_error > HISTORICAL_QAD_TOLERANCE:
            raise RuntimeError(
                "historical QAD reproduction failed: "
                f"observed={results['qad']['miou']:.9f}, "
                f"expected={HISTORICAL_QAD_MIOU:.9f}, error={qad_error:.9f}"
            )
    decision = preregistered_decision(results)
    payload = {
        "protocol": {
            "dataset": "UDD val_patches.txt",
            "validation_samples": EXPECTED_VALIDATION_SAMPLES,
            "input_size": [400, 400],
            "time_steps": 1,
            "batch_size": 20,
            "seed": 1234,
            "qad_training_epochs_observed": 127,
            "baseline_training_epochs": 16,
            "learning_rate_schedule_epochs": 150,
            "smoke": args.smoke,
            "evaluated_batches_limit": limit,
            "historical_qad_expected_miou": HISTORICAL_QAD_MIOU,
            "historical_qad_tolerance": HISTORICAL_QAD_TOLERANCE,
        },
        "results": results,
        "decision": decision,
        "protected_files": validate_protected_hashes(),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(results, output_dir / "comparison.csv")
    write_markdown(results, decision, output_dir / "report.md")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
