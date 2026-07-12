#!/usr/bin/env python3
"""Summarize actual TensorRT engine layer datatypes from Inspector JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = PROJECT_ROOT / "tensorrt_energy_analysis/engines"
MAIN_COMPUTE_TYPES = ("correlation", "deconv", "gemm")


def layer_datatypes(layer: dict[str, Any]) -> set[str]:
    return {
        tensor.get("Datatype", "unknown")
        for field in ("Inputs", "Outputs")
        for tensor in layer.get(field, [])
        if isinstance(tensor, dict)
    }


def inspect(path: Path, precision: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    layers = payload["Layers"]
    type_counts = Counter(layer.get("LayerType", "unknown") for layer in layers)
    datatype_counts = Counter()
    int8_layers = []
    for layer in layers:
        datatypes = layer_datatypes(layer)
        datatype_counts.update(datatypes)
        if "Int8" in datatypes:
            int8_layers.append(layer["Name"])
    main_compute: dict[str, Any] = {}
    for layer_type in MAIN_COMPUTE_TYPES:
        selected = [layer for layer in layers if layer.get("LayerType") == layer_type]
        int8_selected = [layer for layer in selected if "Int8" in layer_datatypes(layer)]
        main_compute[layer_type] = {
            "total": len(selected),
            "int8": len(int8_selected),
            "int8_ratio": len(int8_selected) / len(selected) if selected else None,
            "non_int8_layers": [layer["Name"] for layer in selected if layer not in int8_selected],
        }
    if precision == "fp32" and int8_layers:
        raise RuntimeError(f"FP32 engine unexpectedly has INT8 layers: {int8_layers[:10]}")
    if precision == "int8":
        incomplete = {
            layer_type: stats
            for layer_type, stats in main_compute.items()
            if stats["total"] and stats["int8"] != stats["total"]
        }
        if incomplete:
            raise RuntimeError(f"INT8 main-compute coverage is incomplete: {incomplete}")
    return {
        "precision": precision,
        "inspector_json": str(path.resolve()),
        "total_engine_layers": len(layers),
        "layer_type_counts": dict(type_counts),
        "layers_touching_datatype": dict(datatype_counts),
        "layers_touching_int8": len(int8_layers),
        "layers_touching_int8_ratio": len(int8_layers) / len(layers),
        "main_compute": main_compute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    parser.add_argument("--output", type=Path, default=ENGINE_DIR / "precision_coverage.json")
    args = parser.parse_args()
    result = {
        "fp32": inspect(args.engine_dir / "SpikingLETNet_shallow_max_fp32_layer_info.json", "fp32"),
        "int8": inspect(args.engine_dir / "SpikingLETNet_shallow_max_int8_layer_info.json", "int8"),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
