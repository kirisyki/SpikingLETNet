#!/usr/bin/env python3
"""Rewrite SpikingLETNet ONNX graphs into TensorRT-compatible equivalents.

Two graph constructs are rewritten without changing model semantics:

* The fixed-shape Col2Im produced by ``torch.nn.functional.fold`` becomes a
  nine-way static overlap-add composed of Reshape, Slice, Pad, and Add nodes.
* INT32 bias DequantizeLinear nodes become equivalent FP32 initializers.

The original ONNX files are treated as read-only inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP32_INPUT = PROJECT_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_fp32_preprocessed.onnx"
DEFAULT_INT8_INPUT = PROJECT_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_int8_qdq.onnx"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "tensorrt_energy_analysis/models"
DEFAULT_FP32_OUTPUT = DEFAULT_MODEL_DIR / "SpikingLETNet_shallow_max_ann_fp32_trt.onnx"
DEFAULT_INT8_OUTPUT = DEFAULT_MODEL_DIR / "SpikingLETNet_shallow_max_ann_int8_qdq_trt.onnx"


def unique_name(existing: set[str], prefix: str) -> str:
    candidate = prefix
    index = 0
    while candidate in existing:
        index += 1
        candidate = f"{prefix}_{index}"
    existing.add(candidate)
    return candidate


def add_int64_initializer(graph: onnx.GraphProto, name: str, values: list[int]) -> str:
    graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    return name


def add_float_initializer(graph: onnx.GraphProto, name: str, value: float) -> str:
    graph.initializer.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name))
    return name


def make_col2im_overlap_add(
    graph: onnx.GraphProto,
    node: onnx.NodeProto,
    existing_names: set[str],
) -> list[onnx.NodeProto]:
    attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
    if attrs.get("strides") != [1, 1] or attrs.get("pads") != [1, 1, 1, 1]:
        raise ValueError(f"unsupported Col2Im stride/pad in {node.name}: {attrs}")
    if attrs.get("dilations") != [1, 1]:
        raise ValueError(f"unsupported Col2Im dilation in {node.name}: {attrs}")

    initializers = {initializer.name: initializer for initializer in graph.initializer}
    output_size = numpy_helper.to_array(initializers[node.input[1]]).tolist()
    kernel_size = numpy_helper.to_array(initializers[node.input[2]]).tolist()
    if output_size != [25, 25] or kernel_size != [3, 3]:
        raise ValueError(
            f"expected fixed output/kernel [25,25]/[3,3], got {output_size}/{kernel_size}"
        )

    prefix = unique_name(existing_names, "trt_col2im")
    reshape_shape = add_int64_initializer(graph, f"{prefix}_reshape6_shape", [1, 64, 9, 625])
    patch_shape = add_int64_initializer(graph, f"{prefix}_patch_shape", [1, 64, 25, 25])
    slice_axes = add_int64_initializer(graph, f"{prefix}_slice_axes", [2])
    slice_steps = add_int64_initializer(graph, f"{prefix}_slice_steps", [1])
    zero = add_float_initializer(graph, f"{prefix}_pad_zero", 0.0)

    reshaped = f"{prefix}_reshape6"
    nodes = [
        helper.make_node(
            "Reshape",
            [node.input[0], reshape_shape],
            [reshaped],
            name=f"{prefix}_reshape6_node",
        )
    ]
    shifted_outputs: list[str] = []
    for kernel_y in range(3):
        for kernel_x in range(3):
            offset = kernel_y * 3 + kernel_x
            starts = add_int64_initializer(graph, f"{prefix}_slice_{offset}_starts", [offset])
            ends = add_int64_initializer(graph, f"{prefix}_slice_{offset}_ends", [offset + 1])
            sliced = f"{prefix}_slice_{offset}"
            patch = f"{prefix}_patch_{offset}"
            nodes.append(
                helper.make_node(
                    "Slice",
                    [reshaped, starts, ends, slice_axes, slice_steps],
                    [sliced],
                    name=f"{prefix}_slice_{offset}_node",
                )
            )
            nodes.append(
                helper.make_node(
                    "Reshape",
                    [sliced, patch_shape],
                    [patch],
                    name=f"{prefix}_patch_{offset}_reshape",
                )
            )

            source_y_start = max(0, 1 - kernel_y)
            source_y_end = min(25, 26 - kernel_y)
            source_x_start = max(0, 1 - kernel_x)
            source_x_end = min(25, 26 - kernel_x)
            crop_starts = add_int64_initializer(
                graph, f"{prefix}_crop_{offset}_starts", [source_y_start, source_x_start]
            )
            crop_ends = add_int64_initializer(
                graph, f"{prefix}_crop_{offset}_ends", [source_y_end, source_x_end]
            )
            crop_axes = add_int64_initializer(graph, f"{prefix}_crop_{offset}_axes", [2, 3])
            crop_steps = add_int64_initializer(graph, f"{prefix}_crop_{offset}_steps", [1, 1])
            cropped = f"{prefix}_crop_{offset}"
            nodes.append(
                helper.make_node(
                    "Slice",
                    [patch, crop_starts, crop_ends, crop_axes, crop_steps],
                    [cropped],
                    name=f"{prefix}_crop_{offset}_node",
                )
            )

            pad_top = max(0, kernel_y - 1)
            pad_bottom = max(0, 1 - kernel_y)
            pad_left = max(0, kernel_x - 1)
            pad_right = max(0, 1 - kernel_x)
            pads = add_int64_initializer(
                graph,
                f"{prefix}_pad_{offset}_pads",
                [0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right],
            )
            shifted = f"{prefix}_shifted_{offset}"
            nodes.append(
                helper.make_node(
                    "Pad",
                    [cropped, pads, zero],
                    [shifted],
                    name=f"{prefix}_pad_{offset}_node",
                    mode="constant",
                )
            )
            shifted_outputs.append(shifted)

    running = shifted_outputs[0]
    for index, shifted in enumerate(shifted_outputs[1:], start=1):
        is_last = index == len(shifted_outputs) - 1
        added = node.output[0] if is_last else f"{prefix}_sum_{index}"
        nodes.append(
            helper.make_node(
                "Add",
                [running, shifted],
                [added],
                name=f"{prefix}_add_{index}_node",
            )
        )
        running = added
    if running != node.output[0]:
        raise AssertionError("overlap-add did not preserve the Col2Im output name")
    return nodes


def fold_int32_bias_dequantizers(model: onnx.ModelProto) -> list[str]:
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    replacements: list[onnx.TensorProto] = []
    removed_nodes: list[onnx.NodeProto] = []
    folded_names: list[str] = []
    for node in graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 2:
            continue
        quantized = initializers.get(node.input[0])
        scale = initializers.get(node.input[1])
        zero_point = initializers.get(node.input[2]) if len(node.input) >= 3 else None
        if quantized is None or scale is None or quantized.data_type != TensorProto.INT32:
            continue
        q = numpy_helper.to_array(quantized).astype(np.float32)
        s = numpy_helper.to_array(scale).astype(np.float32)
        z = (
            numpy_helper.to_array(zero_point).astype(np.float32)
            if zero_point is not None
            else np.asarray(0.0, dtype=np.float32)
        )
        value = np.multiply(np.subtract(q, z, dtype=np.float32), s, dtype=np.float32)
        replacements.append(numpy_helper.from_array(value.astype(np.float32), name=node.output[0]))
        removed_nodes.append(node)
        folded_names.append(node.name)
    for node in removed_nodes:
        graph.node.remove(node)
    graph.initializer.extend(replacements)
    return folded_names


def rewrite_model(input_path: Path, output_path: Path, fold_biases: bool) -> dict[str, Any]:
    model = onnx.load(input_path)
    graph = model.graph
    existing_names = {
        name
        for node in graph.node
        for name in [node.name, *node.input, *node.output]
        if name
    }
    rewritten_col2im = 0
    replacement_nodes: list[onnx.NodeProto] = []
    for node in list(graph.node):
        if node.op_type != "Col2Im":
            replacement_nodes.append(node)
            continue
        replacement_nodes.extend(make_col2im_overlap_add(graph, node, existing_names))
        rewritten_col2im += 1
    del graph.node[:]
    graph.node.extend(replacement_nodes)
    folded_biases = fold_int32_bias_dequantizers(model) if fold_biases else []
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    checked = onnx.load(output_path)
    onnx.checker.check_model(checked)
    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rewritten_col2im": rewritten_col2im,
        "folded_int32_bias_dequantizers": folded_biases,
        "nodes": len(checked.graph.node),
        "initializers": len(checked.graph.initializer),
        "bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-input", type=Path, default=DEFAULT_FP32_INPUT)
    parser.add_argument("--int8-input", type=Path, default=DEFAULT_INT8_INPUT)
    parser.add_argument("--fp32-output", type=Path, default=DEFAULT_FP32_OUTPUT)
    parser.add_argument("--int8-output", type=Path, default=DEFAULT_INT8_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MODEL_DIR / "rewrite_manifest.json")
    args = parser.parse_args()
    results = [
        rewrite_model(args.fp32_input, args.fp32_output, fold_biases=False),
        rewrite_model(args.int8_input, args.int8_output, fold_biases=True),
    ]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
