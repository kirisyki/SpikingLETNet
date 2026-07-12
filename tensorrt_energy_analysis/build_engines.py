#!/usr/bin/env python3
"""Build reproducible strongly-typed TensorRT FP32 and explicit-QDQ INT8 engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pynvml
import tensorrt as trt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "tensorrt_energy_analysis/models"
ENGINE_DIR = PROJECT_ROOT / "tensorrt_energy_analysis/engines"
DEFAULT_MODELS = {
    "fp32": MODEL_DIR / "SpikingLETNet_shallow_max_ann_fp32_trt.onnx",
    "int8": MODEL_DIR / "SpikingLETNet_shallow_max_ann_int8_qdq_trt.onnx",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CapturingLogger(trt.ILogger):
    def __init__(self, minimum: trt.ILogger.Severity = trt.ILogger.Severity.INFO) -> None:
        trt.ILogger.__init__(self)
        self.minimum = minimum
        self.lines: list[str] = []

    def log(self, severity: trt.ILogger.Severity, message: str) -> None:
        if int(severity) <= int(self.minimum):
            line = f"[{severity.name}] {message}"
            self.lines.append(line)
            if int(severity) <= int(trt.ILogger.Severity.WARNING):
                print(line, flush=True)


def gpu_info() -> dict[str, Any]:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            "name": pynvml.nvmlDeviceGetName(handle),
            "uuid": pynvml.nvmlDeviceGetUUID(handle),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0,
        }
    finally:
        pynvml.nvmlShutdown()


def tensor_info(engine: trt.ICudaEngine) -> list[dict[str, Any]]:
    rows = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        rows.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "shape": list(engine.get_tensor_shape(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "location": str(engine.get_tensor_location(name)),
            }
        )
    return rows


def build_one(
    precision: str,
    onnx_path: Path,
    output_dir: Path,
    workspace_gib: float,
    optimization_level: int,
) -> dict[str, Any]:
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_path = output_dir / f"SpikingLETNet_shallow_max_{precision}.plan"
    cache_path = output_dir / f"SpikingLETNet_shallow_max_{precision}.timing_cache"
    layer_info_path = output_dir / f"SpikingLETNet_shallow_max_{precision}_layer_info.json"
    log_path = output_dir / f"SpikingLETNet_shallow_max_{precision}_build.log"

    logger = CapturingLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(f"failed to parse {onnx_path}: {errors}")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            f"expected one input and output, got {network.num_inputs}/{network.num_outputs}"
        )
    input_tensor = network.get_input(0)
    if list(input_tensor.shape) != [1, 3, 400, 400]:
        raise RuntimeError(f"unexpected input shape: {list(input_tensor.shape)}")

    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = optimization_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1024**3))
    )
    timing_cache = config.create_timing_cache(b"")
    if not config.set_timing_cache(timing_cache, ignore_mismatch=False):
        raise RuntimeError("TensorRT rejected the empty timing cache")

    print(f"building {precision} engine from {onnx_path}", flush=True)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        log_path.write_text("\n".join(logger.lines) + "\n")
        raise RuntimeError(f"TensorRT failed to build {precision}; see {log_path}")
    engine_path.write_bytes(bytes(serialized))
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    log_path.write_text("\n".join(logger.lines) + "\n")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {engine_path}")
    inspector = engine.create_engine_inspector()
    layer_info_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    layer_info_path.write_text(layer_info_text + ("\n" if not layer_info_text.endswith("\n") else ""))

    result = {
        "precision": precision,
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": sha256_file(onnx_path),
        "engine": str(engine_path.resolve()),
        "engine_sha256": sha256_file(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "timing_cache": str(cache_path.resolve()),
        "timing_cache_sha256": sha256_file(cache_path),
        "layer_info": str(layer_info_path.resolve()),
        "build_log": str(log_path.resolve()),
        "build_seconds": build_seconds,
        "strongly_typed": True,
        "tf32_enabled": config.get_flag(trt.BuilderFlag.TF32),
        "workspace_gib": workspace_gib,
        "builder_optimization_level": optimization_level,
        "io_tensors": tensor_info(engine),
        "engine_layers": engine.num_layers,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "int8"], default=["fp32", "int8"])
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_MODELS["fp32"])
    parser.add_argument("--int8-model", type=Path, default=DEFAULT_MODELS["int8"])
    parser.add_argument("--output-dir", type=Path, default=ENGINE_DIR)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--optimization-level", type=int, choices=range(0, 6), default=3)
    args = parser.parse_args()
    models = {"fp32": args.fp32_model, "int8": args.int8_model}
    started = datetime.now(timezone.utc)
    results = [
        build_one(
            precision,
            models[precision],
            args.output_dir,
            args.workspace_gib,
            args.optimization_level,
        )
        for precision in args.precisions
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started.isoformat(),
        "python": platform.python_version(),
        "tensorrt": trt.__version__,
        "gpu": gpu_info(),
        "results": results,
    }
    manifest_path = args.output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
