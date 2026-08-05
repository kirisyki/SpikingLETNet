#!/usr/bin/env python3
"""Run the preregistered frozen-checkpoint Integer-LIF distortion probe."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
os.chdir(NETWORK_DIR)

from builders.dataset_builder import build_dataset_test  # noqa: E402
from builders.model_builder import build_model  # noqa: E402
from quantization_distortion_probe.factorized_quantization import (  # noqa: E402
    MODES,
    build_probe_model,
    bypass_audit,
)
from quantization_distortion_probe.qif_capture import (  # noqa: E402
    QIFCapture,
    compare_qif_captures,
)
from quantization_distortion_probe.reporting import (  # noqa: E402
    write_json,
    write_report,
    write_summary_csv,
)
from quantization_distortion_probe.statistics import (  # noqa: E402
    CLASS_NAMES,
    VARIANT_MODES,
    summarize_probe,
)
from spikingjelly.activation_based import functional  # noqa: E402


DEFAULT_CONFIG = NETWORK_DIR / "configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/"
    "model_best.pth"
)
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
DEFAULT_OUTPUT = REPO_ROOT / "quantization_distortion_results/frozen_fp_qif_probe"
EXPECTED_VALIDATION_PATCHES = 8478
RAW_LAYER_FIELDS = (
    "mode",
    "patch_index",
    "source_id",
    "image_path",
    "layer",
    "invocations",
    "count",
    "disagreement_count",
    "abs_error_sum",
    "signed_error_sum",
    "reference_zero_count",
    "variant_zero_count",
    "reference_saturation_count",
    "variant_saturation_count",
)
RAW_TASK_FIELDS = (
    "mode",
    "patch_index",
    "source_id",
    "image_path",
    "valid_pixel_count",
    "prediction_flip_count_vs_fp_qif",
)
COUNT_FIELDS = (
    "count",
    "disagreement_count",
    "abs_error_sum",
    "signed_error_sum",
    "reference_zero_count",
    "variant_zero_count",
    "reference_saturation_count",
    "variant_saturation_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Zero runs the complete validation loader.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270805)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id_from_path(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_img"):
        stem = stem[: -len("_img")]
    pieces = stem.split("_")
    if len(pieces) < 3:
        raise ValueError(f"cannot derive source image from {path}")
    return "_".join(pieces[:-2])


def temporal_average(output: torch.Tensor) -> torch.Tensor:
    return output.mean(0) if output.dim() in (3, 5) else output


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> np.ndarray:
    labels = labels.flatten().long()
    predictions = predictions.flatten().long()
    valid = (labels >= 0) & (labels < len(CLASS_NAMES))
    indices = len(CLASS_NAMES) * labels[valid] + predictions[valid]
    matrix = torch.bincount(
        indices, minlength=len(CLASS_NAMES) ** 2
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    return matrix.cpu().numpy().astype(np.int64, copy=False)


def add_counts(target: dict[str, int], row: dict[str, Any], sample: int) -> None:
    for field in COUNT_FIELDS:
        value = row[field]
        if not torch.is_tensor(value):
            raise TypeError(f"expected tensor count for {field}")
        target[field] += int(value[sample].item())


def build_source_model(checkpoint: Path, config: Path) -> tuple[torch.nn.Module, int | None]:
    model = build_model(
        "SpikingLETNet_shallow_max",
        num_classes=len(CLASS_NAMES),
        config=str(config),
    )
    functional.set_step_mode(model, step_mode="m")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint does not contain a model state: {checkpoint}")
    model.load_state_dict(payload["model"], strict=True)
    return model, payload.get("epoch")


def git_state() -> dict[str, str]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short"),
    }


def source_layer_json(
    aggregates: dict[tuple[str, str, str], dict[str, int]]
) -> list[dict[str, Any]]:
    rows = []
    for (mode, source, layer), counts in sorted(aggregates.items()):
        rows.append({"mode": mode, "source_id": source, "layer": layer, **counts})
    return rows


def source_confusion_json(
    aggregates: dict[tuple[str, str], np.ndarray]
) -> list[dict[str, Any]]:
    return [
        {"mode": mode, "source_id": source, "matrix": matrix.tolist()}
        for (mode, source), matrix in sorted(aggregates.items())
    ]


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    if args.batch_size != 20:
        raise ValueError("the preregistered historical validation batch size is 20")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    for path in (args.checkpoint, args.config, args.split_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty result directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"

    def log(message: str) -> None:
        timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(timestamped, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("the approved execution protocol requires an explicit CUDA index")
    torch.cuda.set_device(device.index)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    source_model, checkpoint_epoch = build_source_model(args.checkpoint, args.config)
    models = {
        mode: build_probe_model(source_model, mode).to(device).eval() for mode in MODES
    }
    del source_model
    for model in models.values():
        functional.set_step_mode(model, step_mode="m")
        functional.reset_net(model)
    captures = {mode: QIFCapture(model) for mode, model in models.items()}

    _, loader = build_dataset_test(
        "udd", args.workers, none_gt=False, batch_size=args.batch_size
    )
    if len(loader.dataset) != EXPECTED_VALIDATION_PATCHES:
        raise RuntimeError(
            f"expected {EXPECTED_VALIDATION_PATCHES} validation patches, "
            f"observed {len(loader.dataset)}"
        )
    dataset_paths = [sample[0] for sample in loader.dataset.samples]
    split_paths = [line.split()[0] for line in args.split_file.read_text().splitlines()]
    if dataset_paths != split_paths:
        raise RuntimeError("canonical loader order differs from the frozen split file")

    code_files = [
        Path(__file__),
        NETWORK_DIR / "quantization_distortion_probe/factorized_quantization.py",
        NETWORK_DIR / "quantization_distortion_probe/qif_capture.py",
        NETWORK_DIR / "quantization_distortion_probe/statistics.py",
        NETWORK_DIR / "quantization_distortion_probe/reporting.py",
    ]
    manifest: dict[str, Any] = {
        "format": "integer-lif-distortion-probe-v1",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "design": "frozen-checkpoint 2x2 operator quantization probe",
            "modes": list(MODES),
            "model": "SpikingLETNet_shallow_max",
            "dataset": "UDD6 validation",
            "expected_validation_patches": EXPECTED_VALIDATION_PATCHES,
            "batch_size": args.batch_size,
            "time_steps": 1,
            "qif_code_range": [0, 8],
            "weight_code_range": [-8, 7],
            "noninteger_operator_input_code_range": [-8, 7],
            "integer_operator_input_bypass_range": [-8, 8],
            "activation_scale": "dynamic per-tensor per forward",
            "bias_precision": "FP32",
            "primary_metric": "source-balanced macro QIF code disagreement",
            "statistical_unit": "35 original UDD source images",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "max_batches": args.max_batches,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
            "epoch": checkpoint_epoch,
        },
        "config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
        },
        "split": {
            "path": str(args.split_file.resolve()),
            "sha256": sha256_file(args.split_file),
        },
        "code_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in code_files},
        "git": git_state(),
        "device": {
            "requested": args.device,
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)

    source_layer: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    source_confusion: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    )
    source_prediction_flip: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sources: set[str] = set()
    active_layers: list[str] | None = None
    inactive_layers: list[str] | None = None
    completed_patches = 0
    completed_batches = 0
    started = time.perf_counter()

    raw_layer_path = args.output_dir / "per_patch_layer_counts.csv.gz"
    raw_task_path = args.output_dir / "per_patch_task_counts.csv.gz"
    with gzip.open(raw_layer_path, "wt", newline="", encoding="utf-8") as layer_handle, gzip.open(
        raw_task_path, "wt", newline="", encoding="utf-8"
    ) as task_handle:
        layer_writer = csv.DictWriter(layer_handle, fieldnames=RAW_LAYER_FIELDS)
        task_writer = csv.DictWriter(task_handle, fieldnames=RAW_TASK_FIELDS)
        layer_writer.writeheader()
        task_writer.writeheader()

        with torch.inference_mode():
            for batch_index, (images_cpu, labels_cpu) in enumerate(loader):
                if args.max_batches > 0 and batch_index >= args.max_batches:
                    break
                batch_size = images_cpu.shape[0]
                patch_indices = list(
                    range(completed_patches, completed_patches + batch_size)
                )
                image_paths = [dataset_paths[index] for index in patch_indices]
                batch_sources = [source_id_from_path(path) for path in image_paths]
                sources.update(batch_sources)
                images = images_cpu.to(device, non_blocking=True).unsqueeze(0)
                labels = labels_cpu.long().to(device, non_blocking=True)

                reference_capture = captures["fp_qif"]
                reference_capture.clear()
                functional.reset_net(models["fp_qif"])
                reference_output = temporal_average(models["fp_qif"](images))
                reference_prediction = reference_output.argmax(1)
                functional.reset_net(models["fp_qif"])

                if active_layers is None:
                    active_layers = reference_capture.active_names
                    inactive_layers = reference_capture.inactive_names
                    if not active_layers:
                        raise RuntimeError("no QIF nodes executed during the reference forward")
                    log(
                        f"declared_qif={len(reference_capture.declared_names)} "
                        f"active_qif={len(active_layers)} inactive_qif={len(inactive_layers)}"
                    )
                elif active_layers != reference_capture.active_names:
                    raise RuntimeError("active reference QIF set changed between batches")

                for sample, (patch_index, source, image_path) in enumerate(
                    zip(patch_indices, batch_sources, image_paths)
                ):
                    valid = (labels[sample] >= 0) & (labels[sample] < len(CLASS_NAMES))
                    valid_count = int(valid.sum().item())
                    source_confusion[("fp_qif", source)] += confusion_matrix(
                        labels[sample], reference_prediction[sample]
                    )
                    task_writer.writerow(
                        {
                            "mode": "fp_qif",
                            "patch_index": patch_index,
                            "source_id": source,
                            "image_path": image_path,
                            "valid_pixel_count": valid_count,
                            "prediction_flip_count_vs_fp_qif": 0,
                        }
                    )

                for mode in VARIANT_MODES:
                    capture = captures[mode]
                    capture.clear()
                    functional.reset_net(models[mode])
                    output = temporal_average(models[mode](images))
                    prediction = output.argmax(1)
                    functional.reset_net(models[mode])
                    comparisons = compare_qif_captures(
                        reference_capture, capture, batch_size=batch_size
                    )
                    if [row["layer"] for row in comparisons] != active_layers:
                        raise RuntimeError(f"active QIF order changed for {mode}")

                    for sample, (patch_index, source, image_path) in enumerate(
                        zip(patch_indices, batch_sources, image_paths)
                    ):
                        valid = (labels[sample] >= 0) & (
                            labels[sample] < len(CLASS_NAMES)
                        )
                        valid_count = int(valid.sum().item())
                        flip_count = int(
                            ((prediction[sample] != reference_prediction[sample]) & valid)
                            .sum()
                            .item()
                        )
                        source_prediction_flip[(mode, source)]["flip_count"] += flip_count
                        source_prediction_flip[(mode, source)]["valid_count"] += valid_count
                        source_confusion[(mode, source)] += confusion_matrix(
                            labels[sample], prediction[sample]
                        )
                        task_writer.writerow(
                            {
                                "mode": mode,
                                "patch_index": patch_index,
                                "source_id": source,
                                "image_path": image_path,
                                "valid_pixel_count": valid_count,
                                "prediction_flip_count_vs_fp_qif": flip_count,
                            }
                        )

                    for comparison in comparisons:
                        layer = str(comparison["layer"])
                        invocations = int(comparison["invocations"])
                        cpu_counts = {
                            field: comparison[field].cpu().tolist() for field in COUNT_FIELDS
                        }
                        rows = []
                        for sample, (patch_index, source, image_path) in enumerate(
                            zip(patch_indices, batch_sources, image_paths)
                        ):
                            row = {
                                "mode": mode,
                                "patch_index": patch_index,
                                "source_id": source,
                                "image_path": image_path,
                                "layer": layer,
                                "invocations": invocations,
                                **{field: int(cpu_counts[field][sample]) for field in COUNT_FIELDS},
                            }
                            rows.append(row)
                            add_counts(source_layer[(mode, source, layer)], comparison, sample)
                        layer_writer.writerows(rows)

                    capture.clear()
                    del output, prediction, comparisons

                reference_capture.clear()
                del images, labels, reference_output, reference_prediction
                completed_patches += batch_size
                completed_batches += 1
                if completed_batches == 1 or completed_batches % 10 == 0:
                    log(
                        f"batches={completed_batches}/{len(loader)} "
                        f"patches={completed_patches} "
                        f"elapsed_seconds={time.perf_counter() - started:.1f}"
                    )

    if completed_patches == 0 or active_layers is None or inactive_layers is None:
        raise RuntimeError("probe did not process any validation data")

    summary = summarize_probe(
        source_layer=source_layer,
        source_confusion=source_confusion,
        source_prediction_flip=source_prediction_flip,
        sources=sources,
        active_layers=active_layers,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    audits = {mode: bypass_audit(models[mode]) for mode in VARIANT_MODES}
    for capture in captures.values():
        capture.close()

    manifest.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_batches": completed_batches,
            "completed_patches": completed_patches,
            "complete_validation": completed_patches == EXPECTED_VALIDATION_PATCHES,
            "elapsed_seconds": time.perf_counter() - started,
            "declared_qif_count": len(captures["fp_qif"].declared_names),
            "active_qif_names": active_layers,
            "inactive_qif_names": inactive_layers,
            "integer_bypass_audit": audits,
            "artifacts": {
                "raw_layer_counts": raw_layer_path.name,
                "raw_task_counts": raw_task_path.name,
                "source_layer_aggregates": "source_layer_aggregates.json",
                "source_confusion": "source_confusion.json",
                "summary": "summary.json",
                "summary_csv": "summary.csv",
                "report": "report.md",
            },
        }
    )
    write_json(args.output_dir / "source_layer_aggregates.json", source_layer_json(source_layer))
    write_json(args.output_dir / "source_confusion.json", source_confusion_json(source_confusion))
    write_json(args.output_dir / "summary.json", summary)
    write_summary_csv(args.output_dir / "summary.csv", summary)
    write_report(args.output_dir / "report.md", summary, manifest)
    write_json(args.output_dir / "manifest.json", manifest)
    log(
        "complete: observable_w4a4_distortion="
        f"{summary['preregistered_decision']['observable_w4a4_distortion']} "
        "coupled_superadditive_distortion="
        f"{summary['preregistered_decision']['coupled_superadditive_distortion']}"
    )
    log(f"report={args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
