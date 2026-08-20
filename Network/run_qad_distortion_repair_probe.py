#!/usr/bin/env python3
"""Compare trained QAD and STE-QAT representations against the QAD FP teacher."""

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

import evaluate_quantization_comparison as common  # noqa: E402
from quantization_comparison.layer_adapter import temporal_average  # noqa: E402
from quantization_distortion_probe.qif_capture import (  # noqa: E402
    QIFCapture,
    compare_qif_captures,
)
from quantization_distortion_probe.statistics import (  # noqa: E402
    CLASS_NAMES,
)
from qad_distortion_repair.statistics import summarize_repair  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


DEFAULT_STE_CHECKPOINT = (
    REPO_ROOT
    / "quantization_comparison_checkpoint/udd/seed1234/ste_qat_w4a4/"
    "checkpoint_best.pth"
)
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
DEFAULT_OUTPUT = REPO_ROOT / "qad_distortion_repair_results/qad_vs_ste16_full_20260806"
EXPECTED_VALIDATION_PATCHES = 8478
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
RAW_QIF_FIELDS = (
    "method",
    "patch_index",
    "source_id",
    "image_path",
    "layer",
    "invocations",
    *COUNT_FIELDS,
)
FEATURE_SUM_FIELDS = (
    "count",
    "sq_error_sum",
    "abs_error_sum",
    "signed_error_sum",
    "reference_sq_sum",
    "variant_sq_sum",
    "dot_sum",
)
RAW_FEATURE_FIELDS = (
    "method",
    "patch_index",
    "source_id",
    "image_path",
    "feature_index",
    *FEATURE_SUM_FIELDS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ste-checkpoint", type=Path, default=DEFAULT_STE_CHECKPOINT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270806)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def source_id_from_path(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_img"):
        stem = stem[: -len("_img")]
    pieces = stem.split("_")
    if len(pieces) < 3:
        raise ValueError(f"cannot derive source image from {path}")
    return "_".join(pieces[:-2])


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> np.ndarray:
    labels = labels.flatten().long()
    predictions = predictions.flatten().long()
    valid = (labels >= 0) & (labels < len(CLASS_NAMES))
    indices = len(CLASS_NAMES) * labels[valid] + predictions[valid]
    return (
        torch.bincount(indices, minlength=len(CLASS_NAMES) ** 2)
        .reshape(len(CLASS_NAMES), len(CLASS_NAMES))
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )


def feature_counts(
    reference: torch.Tensor, variant: torch.Tensor, batch_size: int
) -> dict[str, torch.Tensor | int]:
    if reference.shape != variant.shape or reference.shape[0] != batch_size:
        raise ValueError(
            f"feature shape mismatch: reference={tuple(reference.shape)} "
            f"variant={tuple(variant.shape)} batch={batch_size}"
        )
    reference = reference.reshape(batch_size, -1).float()
    variant = variant.reshape(batch_size, -1).float()
    delta = variant - reference
    return {
        "count": reference.shape[1],
        "sq_error_sum": delta.square().sum(1),
        "abs_error_sum": delta.abs().sum(1),
        "signed_error_sum": delta.sum(1),
        "reference_sq_sum": reference.square().sum(1),
        "variant_sq_sum": variant.square().sum(1),
        "dot_sum": (reference * variant).sum(1),
    }


def add_tensor_counts(
    target: dict[str, int | float],
    counts: dict[str, torch.Tensor | int],
    sample: int,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        value = counts[field]
        if torch.is_tensor(value):
            target[field] += float(value[sample].item())
        else:
            target[field] += int(value)


def rows_from_aggregates(
    aggregates: dict[tuple[Any, ...], dict[str, int | float]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = []
    for key, values in sorted(aggregates.items()):
        rows.append({**dict(zip(key_fields, key)), **values})
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    flattened = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, list):
                        flat[f"{key}_{nested_key}"] = json.dumps(nested_value)
                    elif not isinstance(nested_value, dict):
                        flat[f"{key}_{nested_key}"] = nested_value
            else:
                flat[key] = value
        flattened.append(flat)
        for key in flat:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size != 20:
        raise ValueError("use the historical validation batch size of 20")
    if not args.ste_checkpoint.is_file() or not args.split_file.is_file():
        raise FileNotFoundError("required checkpoint or split file is missing")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("an explicit CUDA device index is required")
    torch.cuda.set_device(device.index)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    models_with_metadata = {
        "fp": common.build_fp(),
        "qad": common.build_qad(),
        "ste": common.build_baseline("ste", args.ste_checkpoint),
    }
    models = {
        method: model.to(device).eval()
        for method, (model, _) in models_with_metadata.items()
    }
    captures = {method: QIFCapture(model) for method, model in models.items()}
    for model in models.values():
        functional.set_step_mode(model, step_mode="m")
        functional.reset_net(model)

    _, loader = common.build_dataset_test(
        "udd", num_workers=args.workers, none_gt=False, batch_size=args.batch_size
    )
    if len(loader.dataset) != EXPECTED_VALIDATION_PATCHES:
        raise RuntimeError(f"unexpected validation size: {len(loader.dataset)}")
    dataset_paths = [sample[0] for sample in loader.dataset.samples]
    split_paths = [line.split()[0] for line in args.split_file.read_text().splitlines()]
    if dataset_paths != split_paths:
        raise RuntimeError("canonical loader order differs from split file")

    manifest = {
        "format": "qad-distortion-repair-probe-v1",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "design": "paired trained-student representation audit against common FP-QIF teacher",
            "dataset": "UDD6 validation",
            "validation_patches": len(loader.dataset),
            "statistical_unit": "35 original source images",
            "batch_size": args.batch_size,
            "time_steps": 1,
            "qif_code_range": [0, 8],
            "methods": ["qad", "ste"],
            "primary_metric": "source-balanced macro QIF code disagreement vs FP-QIF teacher",
            "feature_metrics": ["normalized_mse", "cosine_distance"],
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "max_batches": args.max_batches,
            "known_confound": "QAD observed 127 training epochs; STE-QAT observed 16 epochs",
        },
        "checkpoints": {
            method: {
                **metadata,
                "sha256_recomputed": sha256_file(Path(metadata["path"])),
            }
            for method, (_, metadata) in models_with_metadata.items()
        },
        "split": {
            "path": str(args.split_file.resolve()),
            "sha256": sha256_file(args.split_file),
        },
        "git": git_state(),
        "device": {
            "requested": args.device,
            "name": torch.cuda.get_device_name(device),
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    source_qif: dict[tuple[str, str, str], dict[str, int | float]] = defaultdict(
        lambda: defaultdict(int)
    )
    source_feature: dict[tuple[str, str, int], dict[str, int | float]] = defaultdict(
        lambda: defaultdict(float)
    )
    source_confusion: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    )
    source_flip: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sources: set[str] = set()
    active_names: list[str] | None = None
    completed_batches = 0
    completed_patches = 0
    started = time.perf_counter()

    qif_raw_path = output_dir / "per_patch_qif_counts.csv.gz"
    feature_raw_path = output_dir / "per_patch_feature_counts.csv.gz"
    with gzip.open(qif_raw_path, "wt", newline="", encoding="utf-8") as qif_handle, gzip.open(
        feature_raw_path, "wt", newline="", encoding="utf-8"
    ) as feature_handle:
        qif_writer = csv.DictWriter(qif_handle, fieldnames=RAW_QIF_FIELDS)
        feature_writer = csv.DictWriter(feature_handle, fieldnames=RAW_FEATURE_FIELDS)
        qif_writer.writeheader()
        feature_writer.writeheader()

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

                captures["fp"].clear()
                functional.reset_net(models["fp"])
                fp_output, fp_features = models["fp"].forward_qat(images)
                fp_output = temporal_average(fp_output)
                fp_features = [temporal_average(feature) for feature in fp_features]
                fp_prediction = fp_output.argmax(1)
                functional.reset_net(models["fp"])
                if active_names is None:
                    active_names = captures["fp"].active_names
                    if len(active_names) != 78:
                        raise RuntimeError(f"expected 78 active QIF nodes, got {len(active_names)}")
                elif active_names != captures["fp"].active_names:
                    raise RuntimeError("FP active QIF set changed")

                for sample, source in enumerate(batch_sources):
                    source_confusion[("fp", source)] += confusion_matrix(
                        labels[sample], fp_prediction[sample]
                    )

                for method in ("qad", "ste"):
                    captures[method].clear()
                    functional.reset_net(models[method])
                    output, features = models[method].forward_qat(images)
                    output = temporal_average(output)
                    features = [temporal_average(feature) for feature in features]
                    prediction = output.argmax(1)
                    functional.reset_net(models[method])
                    comparisons = compare_qif_captures(
                        captures["fp"], captures[method], batch_size=batch_size
                    )
                    if [str(row["layer"]) for row in comparisons] != active_names:
                        raise RuntimeError(f"active QIF order changed for {method}")

                    for sample, (patch_index, source, image_path) in enumerate(
                        zip(patch_indices, batch_sources, image_paths)
                    ):
                        valid = (labels[sample] >= 0) & (
                            labels[sample] < len(CLASS_NAMES)
                        )
                        flips = int(
                            ((prediction[sample] != fp_prediction[sample]) & valid)
                            .sum()
                            .item()
                        )
                        source_flip[(method, source)]["flip_count"] += flips
                        source_flip[(method, source)]["valid_count"] += int(
                            valid.sum().item()
                        )
                        source_confusion[(method, source)] += confusion_matrix(
                            labels[sample], prediction[sample]
                        )

                    for comparison in comparisons:
                        layer = str(comparison["layer"])
                        cpu = {
                            field: comparison[field].cpu().tolist()
                            for field in COUNT_FIELDS
                        }
                        raw_rows = []
                        for sample, (patch_index, source, image_path) in enumerate(
                            zip(patch_indices, batch_sources, image_paths)
                        ):
                            row = {
                                "method": method,
                                "patch_index": patch_index,
                                "source_id": source,
                                "image_path": image_path,
                                "layer": layer,
                                "invocations": int(comparison["invocations"]),
                                **{
                                    field: int(cpu[field][sample])
                                    for field in COUNT_FIELDS
                                },
                            }
                            raw_rows.append(row)
                            for field in COUNT_FIELDS:
                                source_qif[(method, source, layer)][field] += row[field]
                        qif_writer.writerows(raw_rows)

                    if len(features) != 6 or len(fp_features) != 6:
                        raise RuntimeError("expected six distillation features")
                    for feature_index, (reference, variant) in enumerate(
                        zip(fp_features, features), start=1
                    ):
                        counts = feature_counts(reference, variant, batch_size)
                        cpu = {
                            field: (
                                counts[field].cpu().tolist()
                                if torch.is_tensor(counts[field])
                                else counts[field]
                            )
                            for field in FEATURE_SUM_FIELDS
                        }
                        raw_rows = []
                        for sample, (patch_index, source, image_path) in enumerate(
                            zip(patch_indices, batch_sources, image_paths)
                        ):
                            row = {
                                "method": method,
                                "patch_index": patch_index,
                                "source_id": source,
                                "image_path": image_path,
                                "feature_index": feature_index,
                                **{
                                    field: (
                                        int(cpu[field])
                                        if field == "count"
                                        else float(cpu[field][sample])
                                    )
                                    for field in FEATURE_SUM_FIELDS
                                },
                            }
                            raw_rows.append(row)
                            for field in FEATURE_SUM_FIELDS:
                                source_feature[(method, source, feature_index)][field] += row[field]
                        feature_writer.writerows(raw_rows)

                    captures[method].clear()
                    del output, features, prediction, comparisons

                captures["fp"].clear()
                del images, labels, fp_output, fp_features, fp_prediction
                completed_batches += 1
                completed_patches += batch_size
                if completed_batches == 1 or completed_batches % 20 == 0:
                    print(
                        f"batches={completed_batches}/{len(loader)} "
                        f"patches={completed_patches} "
                        f"elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )

    qif_rows = rows_from_aggregates(
        source_qif, ("method", "source_id", "layer")
    )
    feature_rows = rows_from_aggregates(
        source_feature, ("method", "source_id", "feature_index")
    )
    confusion_rows = [
        {"method": method, "source_id": source, "matrix": matrix.tolist()}
        for (method, source), matrix in sorted(source_confusion.items())
    ]
    flip_rows = [
        {"method": method, "source_id": source, **counts}
        for (method, source), counts in sorted(source_flip.items())
    ]
    summary, detail = summarize_repair(
        qif_rows=qif_rows,
        feature_rows=feature_rows,
        confusion_rows=confusion_rows,
        flip_rows=flip_rows,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(output_dir / "source_qif_aggregates.json", qif_rows)
    write_json(output_dir / "source_feature_aggregates.json", feature_rows)
    write_json(output_dir / "source_confusion.json", confusion_rows)
    write_json(output_dir / "source_prediction_flips.json", flip_rows)
    write_json(output_dir / "analysis_summary.json", summary)
    write_flat_csv(output_dir / "source_statistics.csv", detail["sources"])
    write_flat_csv(output_dir / "layer_statistics.csv", detail["layers"])
    write_flat_csv(output_dir / "feature_statistics.csv", detail["features"])
    write_flat_csv(output_dir / "stage_statistics.csv", detail["stages"])

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
            "active_qif_layer_count": len(active_names or []),
            "source_count": len(sources),
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

