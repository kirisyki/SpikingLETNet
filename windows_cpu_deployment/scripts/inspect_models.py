#!/usr/bin/env python3
"""Audit portable ONNX files and run a small CPU smoke test."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from common import (
    PACKAGE_ROOT,
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


ELIGIBLE_OPS = {"Conv", "ConvTranspose", "MatMul", "Gemm"}


def dimensions(value_info: Any) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def inspect(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    initializers = {item.name: item for item in model.graph.initializer}

    coverage: Counter[str] = Counter()
    per_op: dict[str, Counter[str]] = defaultdict(Counter)
    details: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in ELIGIBLE_OPS:
            continue
        activation_dq = bool(node.input and producers.get(node.input[0], None) is not None and producers[node.input[0]].op_type == "DequantizeLinear")
        weight_dq = False
        weight_type = None
        if len(node.input) > 1:
            dq = producers.get(node.input[1])
            if dq is not None and dq.op_type == "DequantizeLinear" and dq.input:
                quantized = initializers.get(dq.input[0])
                if quantized is not None:
                    weight_type = onnx.TensorProto.DataType.Name(quantized.data_type)
                    weight_dq = weight_type in {"INT8", "UINT8"}
        output_q = any(
            consumer.op_type == "QuantizeLinear"
            for name in node.output
            for consumer in consumers.get(name, [])
        )
        status = "w8a8" if activation_dq and weight_dq else "partial_or_fp32"
        coverage[status] += 1
        per_op[node.op_type][status] += 1
        details.append(
            {
                "index": index,
                "name": node.name,
                "op_type": node.op_type,
                "activation_has_dq": activation_dq,
                "weight_has_int8_dq": weight_dq,
                "weight_storage_type": weight_type,
                "output_has_q": output_q,
            }
        )

    node_types = Counter(node.op_type for node in model.graph.node)
    node_domains = Counter(node.domain or "ai.onnx" for node in model.graph.node)
    initializer_types = Counter(
        onnx.TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer
    )
    return {
        "path": str(path.relative_to(PACKAGE_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "ir_version": model.ir_version,
        "opsets": [{"domain": item.domain or "ai.onnx", "version": item.version} for item in model.opset_import],
        "inputs": [{"name": item.name, "shape": dimensions(item)} for item in model.graph.input],
        "outputs": [{"name": item.name, "shape": dimensions(item)} for item in model.graph.output],
        "node_count": len(model.graph.node),
        "node_types": dict(sorted(node_types.items())),
        "node_domains": dict(sorted(node_domains.items())),
        "initializer_types": dict(sorted(initializer_types.items())),
        "qdq": {
            "quantize_linear": node_types.get("QuantizeLinear", 0),
            "dequantize_linear": node_types.get("DequantizeLinear", 0),
            "eligible_op_coverage": dict(coverage),
            "per_op_coverage": {name: dict(counts) for name, counts in sorted(per_op.items())},
            "eligible_nodes": details,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "model_audit.json")
    parser.add_argument("--smoke-samples", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    model_variants = {
        **{f"cpu_{label}": path for label, path in PORTABLE_MODELS.items()},
        **{f"openvino_{label}": path for label, path in OPENVINO_PORTABLE_MODELS.items()},
    }
    audits = {label: inspect(path) for label, path in model_variants.items()}
    expected_input = [{"name": "image", "shape": [1, 3, 400, 400]}]
    expected_output = [{"name": "logits", "shape": [1, 6, 400, 400]}]
    for label, audit in audits.items():
        if audit["inputs"] != expected_input or audit["outputs"] != expected_output:
            raise RuntimeError(f"{label} has an unexpected public interface")
    if audits["cpu_fp32"]["qdq"]["quantize_linear"] != 0:
        raise RuntimeError("FP32 model unexpectedly contains QuantizeLinear")
    if audits["cpu_w8a8"]["qdq"]["quantize_linear"] == 0:
        raise RuntimeError("W8A8 model has no QDQ nodes")
    if audits["cpu_w8a8"]["initializer_types"].get("INT8", 0) == 0:
        raise RuntimeError("W8A8 model has no INT8 weights")

    inputs = load_inputs(args.smoke_samples)
    outputs: dict[str, list[np.ndarray]] = {"fp32": [], "w8a8": []}
    sessions: dict[str, Any] = {}
    smoke: dict[str, Any] = {}
    for label, path in PORTABLE_MODELS.items():
        session = create_session(path, "cpu", args.threads)
        sessions[label] = session
        sample_reports = []
        for image in inputs:
            output = session.run(["logits"], {"image": image})[0]
            if output.shape != (1, 6, 400, 400) or not np.isfinite(output).all():
                raise RuntimeError(f"{label} produced an invalid output")
            outputs[label].append(output)
            sample_reports.append(
                {
                    "shape": list(output.shape),
                    "finite": True,
                    "minimum": float(output.min()),
                    "maximum": float(output.max()),
                }
            )
        smoke[label] = {
            "providers": session.get_providers(),
            "provider_options": session.get_provider_options(),
            "samples": sample_reports,
        }

    rewrite_equivalence = {}
    for label, path in OPENVINO_PORTABLE_MODELS.items():
        session = create_session(path, "cpu", args.threads)
        max_errors = []
        for image, reference in zip(inputs, outputs[label]):
            candidate = session.run(["logits"], {"image": image})[0]
            max_errors.append(float(np.max(np.abs(reference - candidate))))
            if not np.array_equal(reference, candidate):
                raise RuntimeError(f"{label} OpenVINO-compatible rewrite changed ORT CPU output")
        rewrite_equivalence[label] = {"exact_equal": True, "max_abs_errors": max_errors}

    comparisons = []
    for fp32, w8a8 in zip(outputs["fp32"], outputs["w8a8"]):
        comparisons.append(
            {
                "max_abs_difference": float(np.max(np.abs(fp32 - w8a8))),
                "argmax_agreement_observational_only": float(
                    np.mean(fp32.argmax(axis=1) == w8a8.argmax(axis=1))
                ),
            }
        )
    report = {
        "generated_at_utc": utc_now(),
        "passed": True,
        "accuracy_is_acceptance_criterion": False,
        "environment": environment_summary(),
        "models": audits,
        "cpu_smoke": smoke,
        "openvino_rewrite_equivalence_on_cpu_ep": rewrite_equivalence,
        "observational_comparison": comparisons,
    }
    write_json(args.output, report)
    print(f"PASS: model audit written to {args.output}")


if __name__ == "__main__":
    main()
