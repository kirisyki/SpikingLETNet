#!/usr/bin/env python3
"""Generate target-local ORT graphs and warm OpenVINO's compilation cache."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from common import (
    CPU_OPTIMIZED_MODELS,
    LOCAL_MODELS_DIR,
    OPENVINO_PORTABLE_MODELS,
    PORTABLE_MODELS,
    RESULTS_DIR,
    create_session,
    environment_summary,
    load_inputs,
    sha256_file,
    utc_now,
    write_json,
)


def graph_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    types = Counter(node.op_type for node in model.graph.node)
    domains = Counter(node.domain or "ai.onnx" for node in model.graph.node)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "node_count": len(model.graph.node),
        "node_types": dict(sorted(types.items())),
        "node_domains": dict(sorted(domains.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", choices=("cpu", "openvino"), default=["cpu"])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=LOCAL_MODELS_DIR)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    image = load_inputs(1)[0]
    results: dict[str, Any] = {}
    if "cpu" in args.providers:
        cpu_results = {}
        for label, source in PORTABLE_MODELS.items():
            target = (
                CPU_OPTIMIZED_MODELS[label]
                if args.output_dir.resolve() == LOCAL_MODELS_DIR.resolve()
                else args.output_dir / CPU_OPTIMIZED_MODELS[label].name
            )
            target.unlink(missing_ok=True)
            session = create_session(source, "cpu", args.threads, optimized_output=target)
            output = session.run(["logits"], {"image": image})[0]
            if output.shape != (1, 6, 400, 400) or not np.isfinite(output).all():
                raise RuntimeError(f"Target-local {label} ORT model failed smoke test")
            cpu_results[label] = graph_summary(target)
        results["cpu"] = cpu_results

    if "openvino" in args.providers:
        ov_results = {}
        for label, source in OPENVINO_PORTABLE_MODELS.items():
            cache_dir = args.output_dir / "openvino_cache" / label
            session = create_session(source, "openvino", args.threads, cache_dir=cache_dir)
            output = session.run(["logits"], {"image": image})[0]
            if output.shape != (1, 6, 400, 400) or not np.isfinite(output).all():
                raise RuntimeError(f"OpenVINO {label} cache warmup failed")
            ov_results[label] = {
                "portable_model": str(source),
                "portable_sha256": sha256_file(source),
                "cache_dir": str(cache_dir),
                "session_providers": session.get_providers(),
                "provider_options": session.get_provider_options(),
            }
        results["openvino"] = ov_results

    report = {
        "generated_at_utc": utc_now(),
        "target_local_only": True,
        "warning": "Do not copy these CPU-optimized artifacts to a different CPU/runtime.",
        "environment": environment_summary(),
        "results": results,
    }
    output = RESULTS_DIR / "local_model_preparation.json"
    write_json(output, report)
    print(f"PASS: local model preparation written to {output}")


if __name__ == "__main__":
    main()
