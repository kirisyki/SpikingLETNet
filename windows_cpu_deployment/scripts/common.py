#!/usr/bin/env python3
"""Shared helpers for the Windows CPU deployment bundle."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import psutil
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PACKAGE_ROOT / "models"
RESULTS_DIR = PACKAGE_ROOT / "results"
LOCAL_MODELS_DIR = MODELS_DIR / "local_optimized"
VALIDATION_LIST = PACKAGE_ROOT / "data/validation/validation.txt"

PORTABLE_MODELS = {
    "fp32": MODELS_DIR / "SpikingLETNet_shallow_max_fp32_portable.onnx",
    "w8a8": MODELS_DIR / "SpikingLETNet_shallow_max_w8a8_qdq_portable.onnx",
}
OPENVINO_PORTABLE_MODELS = {
    "fp32": MODELS_DIR / "SpikingLETNet_shallow_max_fp32_openvino_portable.onnx",
    "w8a8": MODELS_DIR / "SpikingLETNet_shallow_max_w8a8_qdq_openvino_portable.onnx",
}
CPU_OPTIMIZED_MODELS = {
    label: LOCAL_MODELS_DIR / f"SpikingLETNet_shallow_max_{label}_ort_cpu.onnx"
    for label in PORTABLE_MODELS
}

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def preprocess_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        image = image_file.convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    chw = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(((chw - IMAGE_MEAN) / IMAGE_STD)[None], dtype=np.float32)


def validation_entries(limit: int | None = None) -> list[tuple[Path, Path]]:
    if not VALIDATION_LIST.is_file():
        raise FileNotFoundError(VALIDATION_LIST)
    entries: list[tuple[Path, Path]] = []
    for line_number, line in enumerate(VALIDATION_LIST.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{VALIDATION_LIST}:{line_number}: expected image and mask")
        image = PACKAGE_ROOT / fields[0]
        mask = PACKAGE_ROOT / fields[1]
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing validation pair: {image}, {mask}")
        entries.append((image, mask))
        if limit is not None and len(entries) >= limit:
            break
    if not entries:
        raise RuntimeError("Validation list is empty")
    return entries


def load_inputs(limit: int | None = None) -> list[np.ndarray]:
    return [preprocess_image(image) for image, _ in validation_entries(limit)]


def _prepare_openvino_windows_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    try:
        import openvino
    except ImportError:
        return
    package_dir = Path(openvino.__file__).resolve().parent
    for candidate in (package_dir / "libs", package_dir.parent / "openvino" / "libs"):
        if candidate.is_dir():
            os.add_dll_directory(str(candidate))


def import_ort(openvino: bool = False) -> Any:
    if openvino:
        _prepare_openvino_windows_dlls()
    import onnxruntime as ort

    return ort


def create_session(
    model_path: Path,
    provider: str,
    threads: int,
    *,
    optimized_output: Path | None = None,
    cache_dir: Path | None = None,
    profile_prefix: Path | None = None,
) -> Any:
    if provider not in {"cpu", "openvino"}:
        raise ValueError(f"Unsupported provider: {provider}")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    ort = import_ort(openvino=provider == "openvino")
    available = ort.get_available_providers()
    required = "CPUExecutionProvider" if provider == "cpu" else "OpenVINOExecutionProvider"
    if required not in available:
        raise RuntimeError(f"{required} unavailable; installed providers: {available}")

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if profile_prefix is not None:
        profile_prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)

    if provider == "cpu":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized_output is not None:
            optimized_output.parent.mkdir(parents=True, exist_ok=True)
            options.optimized_model_filepath = str(optimized_output)
        providers: list[Any] = ["CPUExecutionProvider"]
    else:
        # OpenVINO recommends receiving the unmodified graph and running its own
        # hardware-specific optimization pipeline.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        config = {
            "CPU": {
                "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": "1",
                "INFERENCE_NUM_THREADS": str(threads),
                "INFERENCE_PRECISION_HINT": "f32",
            }
        }
        provider_options = {
            "device_type": "CPU",
            "load_config": json.dumps(config, separators=(",", ":")),
        }
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            provider_options["cache_dir"] = str(cache_dir)
        providers = [("OpenVINOExecutionProvider", provider_options), "CPUExecutionProvider"]

    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def model_path_for(provider: str, label: str, require_local_optimized: bool = True) -> Path:
    if label not in PORTABLE_MODELS:
        raise KeyError(label)
    if provider == "openvino":
        return OPENVINO_PORTABLE_MODELS[label]
    path = CPU_OPTIMIZED_MODELS[label]
    if path.is_file():
        return path
    if require_local_optimized:
        raise FileNotFoundError(
            f"Missing target-local ORT model {path}. Run scripts/prepare_local_models.py first."
        )
    return PORTABLE_MODELS[label]


@dataclass(frozen=True)
class CpuSet:
    cpu_id: int
    cpu_set_id: int
    group: int
    logical_index: int
    core_index: int
    efficiency_class: int
    parked: bool


def _windows_cpu_sets() -> list[CpuSet]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetSystemCpuSetInformation
    function.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p, ctypes.c_ulong]
    function.restype = ctypes.c_bool

    needed = ctypes.c_ulong(0)
    function(None, 0, ctypes.byref(needed), None, 0)
    if needed.value == 0:
        raise OSError(ctypes.get_last_error(), "GetSystemCpuSetInformation returned no data")
    buffer = ctypes.create_string_buffer(needed.value)
    if not function(buffer, needed.value, ctypes.byref(needed), None, 0):
        raise OSError(ctypes.get_last_error(), "GetSystemCpuSetInformation failed")

    result: list[CpuSet] = []
    offset = 0
    while offset < needed.value:
        size, info_type = struct.unpack_from("<II", buffer.raw, offset)
        if size < 20:
            raise RuntimeError(f"Invalid SYSTEM_CPU_SET_INFORMATION size {size}")
        if info_type == 0:  # CpuSetInformation
            cpu_set_id = struct.unpack_from("<I", buffer.raw, offset + 8)[0]
            group = struct.unpack_from("<H", buffer.raw, offset + 12)[0]
            logical, core, _llc, _numa, efficiency, flags = struct.unpack_from(
                "<BBBBBB", buffer.raw, offset + 14
            )
            result.append(
                CpuSet(
                    cpu_id=group * 64 + logical,
                    cpu_set_id=cpu_set_id,
                    group=group,
                    logical_index=logical,
                    core_index=core,
                    efficiency_class=efficiency,
                    parked=bool(flags & 0x01),
                )
            )
        offset += size
    if not result:
        raise RuntimeError("Windows returned no CPU-set records")
    return sorted(result, key=lambda item: (item.group, item.logical_index))


def discover_cpu_sets() -> tuple[list[CpuSet], str | None]:
    if os.name == "nt":
        try:
            return _windows_cpu_sets(), None
        except Exception as exc:  # keep the portable fallback auditable
            error = f"{type(exc).__name__}: {exc}"
    else:
        error = "Windows CPU Set API is unavailable on this OS"
    try:
        allowed = psutil.Process().cpu_affinity()
    except (AttributeError, NotImplementedError):
        logical = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        allowed = list(range(logical))
    sets = [CpuSet(i, i, 0, i, i, 0, False) for i in allowed]
    return sets, error


def affinity_profiles() -> tuple[dict[str, list[int]], dict[str, Any]]:
    cpu_sets, discovery_error = discover_cpu_sets()
    active = [item for item in cpu_sets if not item.parked]
    if not active:
        active = cpu_sets

    all_logical = [item.cpu_id for item in active]
    physical_by_core: dict[tuple[int, int], CpuSet] = {}
    for item in active:
        physical_by_core.setdefault((item.group, item.core_index), item)
    all_physical = [item.cpu_id for item in physical_by_core.values()]

    highest_efficiency = max(item.efficiency_class for item in active)
    p_sets = [item for item in active if item.efficiency_class == highest_efficiency]
    p_physical_by_core: dict[tuple[int, int], CpuSet] = {}
    for item in p_sets:
        p_physical_by_core.setdefault((item.group, item.core_index), item)

    profiles: dict[str, list[int]] = {
        "p_physical": [item.cpu_id for item in p_physical_by_core.values()],
        "p_all_logical": [item.cpu_id for item in p_sets],
        "all_physical": all_physical,
        "all_logical": all_logical,
    }
    profiles = {name: sorted(set(ids)) for name, ids in profiles.items() if ids}
    # Non-hybrid/fallback systems produce duplicate profiles; keep one of each.
    unique: dict[tuple[int, ...], str] = {}
    deduplicated: dict[str, list[int]] = {}
    for name, ids in profiles.items():
        key = tuple(ids)
        if key not in unique:
            unique[key] = name
            deduplicated[name] = ids

    metadata = {
        "discovery_error": discovery_error,
        "highest_efficiency_class": highest_efficiency,
        "cpu_sets": [asdict(item) for item in cpu_sets],
    }
    return deduplicated, metadata


@contextlib.contextmanager
def process_affinity(cpu_ids: Sequence[int]) -> Iterator[None]:
    process = psutil.Process()
    old_affinity: list[int] | None = None
    old_priority: Any = None
    if hasattr(process, "cpu_affinity"):
        old_affinity = process.cpu_affinity()
        process.cpu_affinity(list(cpu_ids))
    try:
        if os.name == "nt":
            old_priority = process.nice()
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        yield
    finally:
        if old_priority is not None:
            process.nice(old_priority)
        if old_affinity is not None:
            process.cpu_affinity(old_affinity)


def percentile(sorted_values: Sequence[float], percentage: float) -> float:
    if not sorted_values:
        raise ValueError("No values")
    position = (len(sorted_values) - 1) * percentage / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_latencies(seconds: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) * 1000.0 for value in seconds)
    if not ordered:
        raise ValueError("No latency samples")
    mean = statistics.fmean(ordered)
    stdev = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    return {
        "iterations": len(ordered),
        "mean_ms": mean,
        "median_ms": statistics.median(ordered),
        "p90_ms": percentile(ordered, 90),
        "p95_ms": percentile(ordered, 95),
        "p99_ms": percentile(ordered, 99),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": stdev,
        "cv": stdev / mean if mean else 0.0,
        "images_per_second": 1000.0 / mean if mean else 0.0,
    }


def measured_runs(
    session: Any,
    inputs: Sequence[np.ndarray],
    *,
    warmup_seconds: float,
    min_iterations: int,
    min_measure_seconds: float = 0.0,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("No inputs")
    warmup_started = time.perf_counter()
    index = 0
    checksum = 0.0
    while time.perf_counter() - warmup_started < warmup_seconds:
        output = session.run(["logits"], {"image": inputs[index % len(inputs)]})[0]
        checksum += float(output.reshape(-1)[0])
        index += 1

    latencies: list[float] = []
    measured_started = time.perf_counter()
    while len(latencies) < min_iterations or time.perf_counter() - measured_started < min_measure_seconds:
        started = time.perf_counter()
        output = session.run(["logits"], {"image": inputs[index % len(inputs)]})[0]
        latencies.append(time.perf_counter() - started)
        checksum += float(output.reshape(-1)[0])
        index += 1
    result = summarize_latencies(latencies)
    result.update({"warmup_seconds": warmup_seconds, "checksum": checksum})
    return result


def environment_summary(ort: Any | None = None) -> dict[str, Any]:
    if ort is None:
        ort = import_ort()
    return {
        "generated_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "numpy": np.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else "unknown",
        "psutil": psutil.__version__,
    }
