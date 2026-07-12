#!/usr/bin/env python3
"""Validate TensorRT-compatible ONNX rewrites against the original models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import tensorrt as trt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "tensorrt_energy_analysis/models"
DEFAULT_VAL_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_paths(split_file: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    with split_file.open() as handle:
        for line in handle:
            fields = line.strip().split()
            if fields:
                path = Path(fields[0])
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths.append(path)
            if len(paths) >= count:
                break
    if len(paths) != count:
        raise ValueError(f"requested {count} images, found {len(paths)} in {split_file}")
    return paths


def preprocess(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((400, 400), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / np.float32(255.0)
    chw = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(((chw - MEAN) / STD)[None], dtype=np.float32)


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def validate_pair(
    label: str,
    original: Path,
    rewritten: Path,
    paths: list[Path],
    threads: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    original_session = make_session(original, threads)
    rewritten_session = make_session(rewritten, threads)
    maximum_absolute_error = 0.0
    maximum_mean_absolute_error = 0.0
    samples: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        inputs = preprocess(path)
        expected = original_session.run(["logits"], {"image": inputs})[0]
        actual = rewritten_session.run(["logits"], {"image": inputs})[0]
        if expected.shape != (1, 6, 400, 400) or actual.shape != expected.shape:
            raise RuntimeError(f"{label} output shape mismatch: {expected.shape} vs {actual.shape}")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError(f"{label} produced non-finite output for {path}")
        difference = np.abs(expected - actual)
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())
        close = bool(np.allclose(expected, actual, rtol=rtol, atol=atol))
        if not close:
            raise RuntimeError(
                f"{label} rewrite failed allclose for {path}: max_abs={max_abs}, mean_abs={mean_abs}"
            )
        maximum_absolute_error = max(maximum_absolute_error, max_abs)
        maximum_mean_absolute_error = max(maximum_mean_absolute_error, mean_abs)
        samples.append({"index": index, "image": str(path), "max_abs": max_abs, "mean_abs": mean_abs})
        print(f"{label}: validated {index}/{len(paths)}", flush=True)
    return {
        "label": label,
        "original": str(original.resolve()),
        "original_sha256": sha256_file(original),
        "rewritten": str(rewritten.resolve()),
        "rewritten_sha256": sha256_file(rewritten),
        "samples": len(paths),
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_mean_absolute_error": maximum_mean_absolute_error,
        "sample_results": samples,
    }


def validate_tensorrt_parse(path: Path) -> dict[str, Any]:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse(path.read_bytes())
    errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
    if not parsed or errors:
        raise RuntimeError(f"TensorRT failed to parse {path}: {errors}")
    return {
        "model": str(path.resolve()),
        "sha256": sha256_file(path),
        "network_layers": network.num_layers,
        "inputs": network.num_inputs,
        "outputs": network.num_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fp32-original",
        type=Path,
        default=PROJECT_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_fp32_preprocessed.onnx",
    )
    parser.add_argument("--fp32-rewritten", type=Path, default=MODEL_DIR / "SpikingLETNet_shallow_max_ann_fp32_trt.onnx")
    parser.add_argument(
        "--int8-original",
        type=Path,
        default=PROJECT_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_int8_qdq.onnx",
    )
    parser.add_argument("--int8-rewritten", type=Path, default=MODEL_DIR / "SpikingLETNet_shallow_max_ann_int8_qdq_trt.onnx")
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "validation.json")
    args = parser.parse_args()
    paths = read_paths(args.val_split, args.samples)
    results = {
        "onnxruntime": ort.__version__,
        "tensorrt": trt.__version__,
        "pairs": [
            validate_pair("fp32", args.fp32_original, args.fp32_rewritten, paths, args.threads, args.rtol, args.atol),
            validate_pair("int8", args.int8_original, args.int8_rewritten, paths, args.threads, args.rtol, args.atol),
        ],
        "tensorrt_parse": [
            validate_tensorrt_parse(args.fp32_rewritten),
            validate_tensorrt_parse(args.int8_rewritten),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()

