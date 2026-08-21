#!/usr/bin/env python3
"""Evaluate the FP32 and W8A8 QDQ ONNX models on the complete UDD6 validation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
DEFAULT_FP32_MODEL = REPO_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_fp32.onnx"
DEFAULT_W8A8_MODEL = REPO_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_int8_qdq.onnx"
DEFAULT_OUTPUT = REPO_ROOT / "onnx_precision_results/udd/w8a8_20260821"
EXPECTED_VALIDATION_PATCHES = 8478
CLASS_NAMES = ("background", "facade", "road", "vegetation", "vehicle", "roof")
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_FP32_MODEL)
    parser.add_argument("--w8a8-model", type=Path, default=DEFAULT_W8A8_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=positive_int, default=8)
    parser.add_argument("--threads-per-worker", type=positive_int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Evaluate only the first N samples; zero evaluates the complete split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_samples(split_file: Path, max_samples: int) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"expected image and mask at {split_file}:{line_number}")
            image_path, mask_path = fields
            if not Path(image_path).is_file():
                raise FileNotFoundError(image_path)
            if not Path(mask_path).is_file():
                raise FileNotFoundError(mask_path)
            samples.append((image_path, mask_path))
            if max_samples > 0 and len(samples) >= max_samples:
                break
    if not samples:
        raise RuntimeError(f"no validation samples found in {split_file}")
    return samples


def preprocess_image(path: str) -> np.ndarray:
    with Image.open(path) as image_file:
        image = image_file.convert("RGB")
        if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    chw = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(((chw - IMAGE_MEAN) / IMAGE_STD)[None], dtype=np.float32)


def read_mask(path: str) -> np.ndarray:
    with Image.open(path) as mask_file:
        if mask_file.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            mask_file = mask_file.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.NEAREST)
        mask = np.asarray(mask_file, dtype=np.int64)
    if mask.ndim != 2:
        raise ValueError(f"expected a single-channel mask, got shape {mask.shape}: {path}")
    return mask


def make_session(model_path: str, threads: int) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def update_confusion(matrix: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    classes = len(CLASS_NAMES)
    valid = (target >= 0) & (target < classes)
    indices = classes * target[valid] + prediction[valid]
    matrix += np.bincount(indices, minlength=classes**2).reshape(classes, classes)


def evaluate_partition(
    partition_index: int,
    samples: Sequence[tuple[str, str]],
    fp32_model: str,
    w8a8_model: str,
    threads: int,
) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    fp32_session = make_session(fp32_model, threads)
    w8a8_session = make_session(w8a8_model, threads)
    matrices = {
        "FP32 ONNX": np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64),
        "W8A8 QDQ ONNX": np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64),
    }
    agreement_count = 0
    prediction_pixel_count = 0
    started = time.perf_counter()

    for image_path, mask_path in samples:
        image = preprocess_image(image_path)
        target = read_mask(mask_path)
        fp32_logits = fp32_session.run(["logits"], {"image": image})[0]
        w8a8_logits = w8a8_session.run(["logits"], {"image": image})[0]
        expected_shape = (1, len(CLASS_NAMES), IMAGE_HEIGHT, IMAGE_WIDTH)
        if fp32_logits.shape != expected_shape or w8a8_logits.shape != expected_shape:
            raise RuntimeError(
                f"unexpected output shapes: FP32={fp32_logits.shape}, W8A8={w8a8_logits.shape}"
            )
        if not np.isfinite(fp32_logits).all() or not np.isfinite(w8a8_logits).all():
            raise RuntimeError(f"non-finite model output for {image_path}")
        fp32_prediction = fp32_logits.argmax(axis=1)[0]
        w8a8_prediction = w8a8_logits.argmax(axis=1)[0]
        update_confusion(matrices["FP32 ONNX"], target, fp32_prediction)
        update_confusion(matrices["W8A8 QDQ ONNX"], target, w8a8_prediction)
        agreement_count += int(np.count_nonzero(fp32_prediction == w8a8_prediction))
        prediction_pixel_count += int(fp32_prediction.size)

    return {
        "partition_index": partition_index,
        "sample_count": len(samples),
        "elapsed_seconds": time.perf_counter() - started,
        "confusion_matrices": {name: matrix.tolist() for name, matrix in matrices.items()},
        "prediction_agreement_count": agreement_count,
        "prediction_pixel_count": prediction_pixel_count,
    }


def partition_samples(
    samples: Sequence[tuple[str, str]], workers: int
) -> list[list[tuple[str, str]]]:
    partition_count = min(workers, len(samples))
    return [list(samples[index::partition_count]) for index in range(partition_count)]


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    per_class_iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan),
        where=union != 0,
    )
    return {
        "miou": float(np.nanmean(per_class_iou)),
        "per_class_iou": {
            name: float(value) for name, value in zip(CLASS_NAMES, per_class_iou)
        },
        "pixel_accuracy": float(intersection.sum() / matrix.sum()),
        "confusion_matrix": matrix.tolist(),
        "evaluated_pixels": int(matrix.sum()),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_csv(results: dict[str, dict[str, Any]], path: Path) -> None:
    fields = ["model", "miou", "pixel_accuracy", *[f"iou_{name}" for name in CLASS_NAMES]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, result in results.items():
            row = {
                "model": model,
                "miou": result["miou"],
                "pixel_accuracy": result["pixel_accuracy"],
            }
            row.update(
                {f"iou_{name}": result["per_class_iou"][name] for name in CLASS_NAMES}
            )
            writer.writerow(row)


def write_report(payload: dict[str, Any], path: Path) -> None:
    results = payload["results"]
    fp32 = results["FP32 ONNX"]
    w8a8 = results["W8A8 QDQ ONNX"]
    delta_pp = payload["comparison"]["w8a8_minus_fp32_miou_points"]
    lines = [
        "# UDD6 FP32 与 W8A8 ONNX mIoU 对比",
        "",
        "两个模型使用相同的完整 UDD6 validation split、相同图像预处理和同一个六分类全局混淆矩阵。",
        "W8A8 是从 FP32 checkpoint 静态校准得到的 QDQ 混合精度 ONNX；仅适合量化的 Conv、ConvTranspose、MatMul 和 Gemm 路径量化为 8-bit，其余算子可保留 FP32。",
        "",
        "| 模型 | mIoU | mIoU (%) | Pixel accuracy | 相对 FP32（pp） |",
        "|---|---:|---:|---:|---:|",
        f"| FP32 ONNX | {fp32['miou']:.6f} | {100 * fp32['miou']:.4f} | {fp32['pixel_accuracy']:.6f} | 0 |",
        f"| W8A8 QDQ ONNX | {w8a8['miou']:.6f} | {100 * w8a8['miou']:.4f} | {w8a8['pixel_accuracy']:.6f} | {delta_pp:+.4f} |",
        "",
        f"W8A8 相对 FP32 的全局 validation mIoU 变化为 **{delta_pp:+.4f} pp**；两者逐像素 argmax 一致率为 **{100 * payload['comparison']['prediction_agreement']:.4f}%**。",
        "",
        "## 逐类 IoU",
        "",
        "| 模型 | background | facade | road | vegetation | vehicle | roof |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| FP32 ONNX | " + " | ".join(f"{fp32['per_class_iou'][name]:.6f}" for name in CLASS_NAMES) + " |",
        "| W8A8 QDQ ONNX | " + " | ".join(f"{w8a8['per_class_iou'][name]:.6f}" for name in CLASS_NAMES) + " |",
        "",
        "## 口径限制",
        "",
        "- 这是 ONNX Runtime CPU 对源 FP32/W8A8 QDQ ONNX 的精度评估，不是 TensorRT engine 的 deployed mIoU。",
        "- W8A8 是训练后静态量化（PTQ），不是单独训练的 W8A8 QAT checkpoint。",
        "- validation 同时用于历史 checkpoint 选择，因此结果应称为 validation mIoU。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    for path in (args.split_file, args.fp32_model, args.w8a8_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_samples(args.split_file, args.max_samples)
    is_complete = args.max_samples == 0
    if is_complete and len(samples) != EXPECTED_VALIDATION_PATCHES:
        raise RuntimeError(
            f"expected {EXPECTED_VALIDATION_PATCHES} samples, observed {len(samples)}"
        )
    partitions = partition_samples(samples, args.workers)
    started = time.perf_counter()
    aggregate_matrices = {
        "FP32 ONNX": np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64),
        "W8A8 QDQ ONNX": np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64),
    }
    agreement_count = 0
    prediction_pixel_count = 0
    evaluated_samples = 0

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(partitions), mp_context=context) as executor:
        futures = [
            executor.submit(
                evaluate_partition,
                index,
                partition,
                str(args.fp32_model.resolve()),
                str(args.w8a8_model.resolve()),
                args.threads_per_worker,
            )
            for index, partition in enumerate(partitions)
        ]
        for future in as_completed(futures):
            result = future.result()
            evaluated_samples += result["sample_count"]
            agreement_count += result["prediction_agreement_count"]
            prediction_pixel_count += result["prediction_pixel_count"]
            for model_name, matrix in result["confusion_matrices"].items():
                aggregate_matrices[model_name] += np.asarray(matrix, dtype=np.int64)
            print(
                f"partition={result['partition_index']} samples={result['sample_count']} "
                f"elapsed={result['elapsed_seconds']:.1f}s total_samples={evaluated_samples}/{len(samples)}",
                flush=True,
            )

    if evaluated_samples != len(samples):
        raise RuntimeError(f"evaluated {evaluated_samples} samples, expected {len(samples)}")
    results = {
        model_name: metrics_from_confusion(matrix)
        for model_name, matrix in aggregate_matrices.items()
    }
    delta_pp = (results["W8A8 QDQ ONNX"]["miou"] - results["FP32 ONNX"]["miou"]) * 100.0
    payload = {
        "format": "udd-onnx-w8a8-miou-comparison-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if is_complete else "partial",
        "protocol": {
            "dataset": "UDD6 validation",
            "split_file": str(args.split_file.resolve()),
            "split_sha256": sha256_file(args.split_file),
            "expected_validation_patches": EXPECTED_VALIDATION_PATCHES,
            "evaluated_patches": len(samples),
            "classes": list(CLASS_NAMES),
            "metric": "six-class mIoU from one global confusion matrix",
            "input_shape": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
            "time_steps": 1,
            "preprocessing": "RGB float32 / 255, ImageNet mean/std normalization",
            "inference_backend": "ONNX Runtime CPUExecutionProvider",
            "workers": len(partitions),
            "threads_per_worker": args.threads_per_worker,
        },
        "models": {
            "FP32 ONNX": {
                "path": str(args.fp32_model.resolve()),
                "sha256": sha256_file(args.fp32_model),
                "precision": "FP32",
                "source": "FP32 QIF checkpoint exported through the T=1 ANN-equivalent graph",
            },
            "W8A8 QDQ ONNX": {
                "path": str(args.w8a8_model.resolve()),
                "sha256": sha256_file(args.w8a8_model),
                "precision": "static S8S8 QDQ mixed precision",
                "quantized_ops": ["Conv", "ConvTranspose", "MatMul", "Gemm"],
                "calibration_split": "UDD6 train",
                "calibration_samples": 300,
                "quantization_method": "MinMax, per-channel weights, per-tensor activations",
            },
        },
        "results": results,
        "comparison": {
            "w8a8_minus_fp32_miou_points": delta_pp,
            "prediction_agreement": agreement_count / prediction_pixel_count,
            "prediction_agreement_count": agreement_count,
            "prediction_pixel_count": prediction_pixel_count,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "git_commit": git_commit(),
        },
    }
    json_path = args.output_dir / "comparison.json"
    csv_path = args.output_dir / "comparison.csv"
    report_path = args.output_dir / "report.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(results, csv_path)
    write_report(payload, report_path)
    print(json.dumps({"comparison": payload["comparison"], "outputs": [str(json_path), str(csv_path), str(report_path)]}, indent=2))


if __name__ == "__main__":
    main()
