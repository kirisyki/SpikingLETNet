#!/usr/bin/env python3
"""Inspect OpenVINO CPU runtime precision for the portable ONNX models."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import LOCAL_MODELS_DIR, OPENVINO_PORTABLE_MODELS, RESULTS_DIR, load_inputs, utc_now, write_json


def rt_value(node: Any, key: str) -> str | None:
    try:
        value = node.get_rt_info()[key]
        # OpenVINO 2025.x exposes metadata through OVAny; str(OVAny) hides the payload.
        if hasattr(value, "get"):
            value = value.get()
        elif hasattr(value, "value"):
            value = value.value
        return str(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError("Install the pinned requirements before using OpenVINO inspection") from exc

    core = ov.Core()
    image = load_inputs(1)[0]
    models = {}
    for label, path in OPENVINO_PORTABLE_MODELS.items():
        cache_dir = LOCAL_MODELS_DIR / "openvino_direct_cache" / label
        cache_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": args.threads,
            "INFERENCE_PRECISION_HINT": "f32",
            "CACHE_DIR": str(cache_dir),
        }
        compiled = core.compile_model(str(path), "CPU", config)
        output = compiled([image])[compiled.output(0)]
        if tuple(output.shape) != (1, 6, 400, 400):
            raise RuntimeError(f"OpenVINO {label} output shape is {output.shape}")
        runtime_model = compiled.get_runtime_model()
        precision_counts: Counter[str] = Counter()
        implementation_counts: Counter[str] = Counter()
        layer_type_counts: Counter[str] = Counter()
        nodes = []
        for node in runtime_model.get_ordered_ops():
            precision = rt_value(node, "runtimePrecision") or rt_value(node, "outputPrecisions")
            implementation = rt_value(node, "execType") or rt_value(node, "primitiveType")
            layer_type = rt_value(node, "layerType") or node.get_type_name()
            precision_counts[precision or "unknown"] += 1
            implementation_counts[implementation or "unknown"] += 1
            layer_type_counts[layer_type or "unknown"] += 1
            nodes.append(
                {
                    "friendly_name": node.get_friendly_name(),
                    "type_name": node.get_type_name(),
                    "layer_type": layer_type,
                    "runtime_precision": precision,
                    "implementation": implementation,
                    "original_names": rt_value(node, "originalLayersNames"),
                }
            )
        integer_runtime_nodes = sum(
            count for name, count in precision_counts.items() if "i8" in name.lower() or "u8" in name.lower()
        )
        integer_implementations = sum(
            count for name, count in implementation_counts.items() if "i8" in name.lower() or "u8" in name.lower()
        )
        if label == "w8a8" and (integer_runtime_nodes == 0 or integer_implementations == 0):
            raise RuntimeError("OpenVINO W8A8 graph compiled without observable integer runtime kernels")
        models[label] = {
            "integer_runtime_nodes": integer_runtime_nodes,
            "integer_implementations": integer_implementations,
            "runtime_precision_counts": dict(precision_counts),
            "implementation_counts": dict(implementation_counts),
            "layer_type_counts": dict(layer_type_counts),
            "nodes": nodes,
        }
    report = {
        "generated_at_utc": utc_now(),
        "openvino": ov.__version__,
        "device": "CPU",
        "models": models,
        "note": "Use runtime_precision and implementation fields to verify that QDQ regions became integer kernels.",
    }
    output = RESULTS_DIR / "openvino_runtime_precision.json"
    write_json(output, report)
    print(f"PASS: OpenVINO precision report written to {output}")


if __name__ == "__main__":
    main()
