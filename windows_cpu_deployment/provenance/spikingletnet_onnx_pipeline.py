#!/usr/bin/env python3
"""Export and static-quantize the ANN form of SpikingLETNet_shallow_max.

The public ONNX interface is a fixed NCHW tensor. Internally, the leading
length-one dimension required by SpikingJelly multi-step layers is added and
removed by :class:`ONNXWrapper`. QIF neurons stay in ANN mode: their bounded
integer activation is exported directly and is never expanded to a spike
sequence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = PROJECT_ROOT / "Network"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))


def install_optional_import_stubs() -> None:
    """Keep model-only tooling usable when plotting helpers are not installed."""

    if "torchsummary" not in sys.modules:
        module = types.ModuleType("torchsummary")
        module.summary = lambda *args, **kwargs: None
        sys.modules["torchsummary"] = module
    if "seaborn" not in sys.modules:
        module = types.ModuleType("seaborn")
        module.heatmap = lambda *args, **kwargs: None
        sys.modules["seaborn"] = module


install_optional_import_stubs()

spikingletnet_module = importlib.import_module("model.SpikingLETNet_shallow_max")
SpikingLETNet_shallow_max = spikingletnet_module.SpikingLETNet_shallow_max
from model.module.neuron import QIFNode  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth"
)
DEFAULT_TRAIN_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/train_patches.txt")
DEFAULT_VAL_SPLIT = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "onnx_models"
DEFAULT_FP32_MODEL = DEFAULT_OUTPUT_DIR / "SpikingLETNet_shallow_max_ann_fp32.onnx"
DEFAULT_PREPROCESSED_MODEL = (
    DEFAULT_OUTPUT_DIR / "SpikingLETNet_shallow_max_ann_fp32_preprocessed.onnx"
)
DEFAULT_INT8_MODEL = DEFAULT_OUTPUT_DIR / "SpikingLETNet_shallow_max_ann_int8_qdq.onnx"
DEFAULT_SMOKE_REPORT = DEFAULT_OUTPUT_DIR / "SpikingLETNet_shallow_max_ann_smoke.json"

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
FEATURE_HEIGHT = IMAGE_HEIGHT // 16
FEATURE_WIDTH = IMAGE_WIDTH // 16


def fixed_reverse_patches(
    images: torch.Tensor,
    _out_size: tuple[int, int],
    ksizes: tuple[int, int],
    strides: int,
    padding: int,
) -> torch.Tensor:
    """Fixed-shape equivalent needed by ONNX Col2Im symbolic export."""

    return torch.nn.functional.fold(
        images,
        output_size=(FEATURE_HEIGHT, FEATURE_WIDTH),
        kernel_size=ksizes,
        dilation=1,
        padding=padding,
        stride=strides,
    )


spikingletnet_module.reverse_patches = fixed_reverse_patches


def exportable_attention_forward(self: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """EffAttention equivalent using explicit slices for its uneven final chunk."""

    inputs = self.reduce(inputs)
    batch, tokens, channels = inputs.shape
    qkv = (
        self.qkv(inputs)
        .reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv[0], qkv[1], qkv[2]
    outputs = []
    for start, end in ((0, 156), (156, 312), (312, 468), (468, 624), (624, 625)):
        query_chunk = query[:, :, start:end, :]
        key_chunk = key[:, :, start:end, :]
        value_chunk = value[:, :, start:end, :]
        attention = (query_chunk @ key_chunk.transpose(-2, -1)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))
        outputs.append((attention @ value_chunk).transpose(1, 2))
    output = torch.cat(outputs, dim=1).reshape(batch, tokens, channels)
    return self.proj(output)


class ANNQuantNeuron(nn.Module):
    """Stateless ANN equivalent of a reset QIFNode with ``bin=False``."""

    def __init__(self, quant_max: int) -> None:
        super().__init__()
        if quant_max <= 0:
            raise ValueError(f"quant_max must be positive, got {quant_max}")
        self.quant_max = float(quant_max)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.round(torch.clamp(inputs, min=0.0, max=self.quant_max))


class ONNXWrapper(nn.Module):
    """Expose NCHW input/output while retaining length-one step mode inside."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image.unsqueeze(0))
        return logits.squeeze(0)


def replace_qif_with_ann(module: nn.Module) -> int:
    """Replace every ANN-mode QIF node and return the replacement count."""

    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, QIFNode):
            if child.bin:
                raise RuntimeError(f"{name} is in SNN/bin mode; ANN export requires bin=False")
            setattr(module, name, ANNQuantNeuron(int(child.T)))
            count += 1
        else:
            count += replace_qif_with_ann(child)
    return count


def unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {type(state_dict)!r}")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def build_ann_model(config: Path, checkpoint: Path, classes: int = 6) -> tuple[ONNXWrapper, int]:
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = SpikingLETNet_shallow_max(classes=classes, config=str(config))
    model.transformer1.atten.forward = types.MethodType(
        exportable_attention_forward, model.transformer1.atten
    )
    state_dict = unwrap_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_bn_missing = [key for key in missing if ".bn_prelu.bn." not in key]
    if non_bn_missing or unexpected:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing_non_bn={non_bn_missing[:20]} unexpected={unexpected[:20]}"
        )

    functional.set_step_mode(model, "m")
    replacement_count = replace_qif_with_ann(model)
    if replacement_count == 0:
        raise RuntimeError("No QIFNode instances were found; refusing to export an ambiguous model")
    model.eval()
    return ONNXWrapper(model).eval(), replacement_count


def read_image_paths(split_file: Path, limit: int) -> list[Path]:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if not split_file.is_file():
        raise FileNotFoundError(split_file)

    paths: list[Path] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            image_path = Path(fields[0])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing image referenced by {split_file}:{line_number}: {image_path}"
                )
            paths.append(image_path)
            if len(paths) >= limit:
                break
    if len(paths) < limit:
        raise ValueError(f"{split_file} contains {len(paths)} usable images, need {limit}")
    return paths


def preprocess_image(path: Path) -> np.ndarray:
    with Image.open(path) as image_file:
        image = image_file.convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    chw = np.transpose(array, (2, 0, 1))
    normalized = (chw - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(normalized[None], dtype=np.float32)


def import_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install ONNX dependencies with: pip install onnx onnxruntime") from exc
    return onnx


def import_ort() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install ONNX dependencies with: pip install onnx onnxruntime") from exc
    return ort


def check_onnx(path: Path) -> dict[str, int]:
    onnx = import_onnx()
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def export_fp32(args: argparse.Namespace) -> dict[str, Any]:
    args.fp32_model.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model, replacement_count = build_ann_model(args.config, args.checkpoint, args.classes)
    example = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)

    with torch.inference_mode():
        output = model(example)
        expected_shape = (1, args.classes, IMAGE_HEIGHT, IMAGE_WIDTH)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(f"Unexpected PyTorch output shape: {tuple(output.shape)} != {expected_shape}")
        torch.onnx.export(
            model,
            example,
            str(args.fp32_model),
            input_names=["image"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    node_counts = check_onnx(args.fp32_model)
    result = {
        "model": str(args.fp32_model.resolve()),
        "bytes": args.fp32_model.stat().st_size,
        "opset": args.opset,
        "qif_replacements": replacement_count,
        "input_shape": [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
        "output_shape": [1, args.classes, IMAGE_HEIGHT, IMAGE_WIDTH],
        "node_count": sum(node_counts.values()),
    }
    print(json.dumps(result, indent=2))
    return result


class UDDCalibrationDataReader:
    """ORT calibration reader that loads representative UDD images lazily."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = list(paths)
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None

    def _samples(self) -> Iterator[dict[str, np.ndarray]]:
        for path in self.paths:
            yield {"image": preprocess_image(path)}

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self._iterator = self._samples()
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = None


def quantize_int8(args: argparse.Namespace) -> dict[str, Any]:
    if not args.fp32_model.is_file():
        raise FileNotFoundError(f"Export FP32 ONNX first: {args.fp32_model}")
    args.int8_model.parent.mkdir(parents=True, exist_ok=True)

    from onnxruntime.quantization import (  # type: ignore[import-not-found]
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import (  # type: ignore[import-not-found]
        quant_pre_process,
    )

    calibration_paths = read_image_paths(args.train_split, args.calibration_samples)
    quant_pre_process(
        input_model=args.fp32_model,
        output_model_path=args.preprocessed_model,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
        auto_merge=True,
    )

    reader = UDDCalibrationDataReader(calibration_paths)
    started = time.perf_counter()
    quantize_static(
        model_input=args.preprocessed_model,
        model_output=args.int8_model,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm"],
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CPUExecutionProvider"],
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    elapsed = time.perf_counter() - started

    node_counts = check_onnx(args.int8_model)
    quantize_nodes = node_counts.get("QuantizeLinear", 0)
    dequantize_nodes = node_counts.get("DequantizeLinear", 0)
    if quantize_nodes == 0 or dequantize_nodes == 0:
        raise RuntimeError(
            f"Static quantization produced no QDQ pairs: Q={quantize_nodes}, DQ={dequantize_nodes}"
        )

    result = {
        "model": str(args.int8_model.resolve()),
        "bytes": args.int8_model.stat().st_size,
        "calibration_split": str(args.train_split.resolve()),
        "calibration_samples": len(calibration_paths),
        "calibration_seconds": elapsed,
        "quantize_linear_nodes": quantize_nodes,
        "dequantize_linear_nodes": dequantize_nodes,
        "node_count": sum(node_counts.values()),
    }
    print(json.dumps(result, indent=2))
    return result


def make_ort_session(model_path: Path, threads: int) -> Any:
    ort = import_ort()
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"CPUExecutionProvider unavailable: {ort.get_available_providers()}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def timed_ort_run(session: Any, image: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    output = session.run(["logits"], {"image": image})[0]
    return output, time.perf_counter() - started


def smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    if not args.fp32_model.is_file():
        raise FileNotFoundError(args.fp32_model)
    if not args.int8_model.is_file():
        raise FileNotFoundError(args.int8_model)

    fp32_counts = check_onnx(args.fp32_model)
    int8_counts = check_onnx(args.int8_model)
    if int8_counts.get("QuantizeLinear", 0) == 0 or int8_counts.get("DequantizeLinear", 0) == 0:
        raise RuntimeError("INT8 model does not contain QDQ nodes")

    # Native PyTorch CPU kernels align with ORT MLAS accumulation closely enough
    # to avoid amplifying oneDNN rounding noise at discrete Quant thresholds.
    torch.backends.mkldnn.enabled = False
    pytorch_model, replacement_count = build_ann_model(
        args.config, args.checkpoint, args.classes
    )
    fp32_session = make_ort_session(args.fp32_model, args.threads)
    int8_session = make_ort_session(args.int8_model, args.threads)
    paths = read_image_paths(args.val_split, args.smoke_samples)

    expected_shape = (1, args.classes, IMAGE_HEIGHT, IMAGE_WIDTH)
    samples: list[dict[str, Any]] = []
    allclose_passed = True
    min_fp32_argmax_agreement = 1.0

    with torch.inference_mode():
        for path in paths:
            image = preprocess_image(path)
            torch_output = pytorch_model(torch.from_numpy(image)).cpu().numpy()
            fp32_output, fp32_seconds = timed_ort_run(fp32_session, image)
            int8_output, int8_seconds = timed_ort_run(int8_session, image)

            for label, output in (("PyTorch", torch_output), ("ORT FP32", fp32_output), ("ORT INT8", int8_output)):
                if output.shape != expected_shape:
                    raise RuntimeError(f"{label} output shape {output.shape} != {expected_shape}")
                if not np.isfinite(output).all():
                    raise RuntimeError(f"{label} output contains NaN or Inf for {path}")

            is_close = bool(
                np.allclose(
                    torch_output,
                    fp32_output,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )
            allclose_passed = allclose_passed and is_close
            absolute_error = np.abs(torch_output - fp32_output)
            pytorch_prediction = torch_output.argmax(axis=1)
            fp32_prediction = fp32_output.argmax(axis=1)
            int8_prediction = int8_output.argmax(axis=1)
            fp32_argmax_agreement = float(np.mean(pytorch_prediction == fp32_prediction))
            int8_argmax_agreement = float(np.mean(fp32_prediction == int8_prediction))
            min_fp32_argmax_agreement = min(min_fp32_argmax_agreement, fp32_argmax_agreement)

            samples.append(
                {
                    "image": str(path),
                    "fp32_allclose": is_close,
                    "fp32_max_abs_error": float(absolute_error.max()),
                    "fp32_mean_abs_error": float(absolute_error.mean()),
                    "fp32_argmax_agreement": fp32_argmax_agreement,
                    "int8_vs_fp32_argmax_agreement": int8_argmax_agreement,
                    "fp32_ort_seconds": fp32_seconds,
                    "int8_ort_seconds": int8_seconds,
                }
            )

    argmax_passed = min_fp32_argmax_agreement >= args.min_argmax_agreement
    report = {
        "passed": allclose_passed and argmax_passed,
        "criteria": {
            "rtol": args.rtol,
            "atol": args.atol,
            "min_fp32_argmax_agreement": args.min_argmax_agreement,
        },
        "environment": {
            "torch": torch.__version__,
            "pytorch_mkldnn_enabled": torch.backends.mkldnn.enabled,
            "onnxruntime": import_ort().__version__,
            "providers": fp32_session.get_providers(),
            "threads": args.threads,
        },
        "models": {
            "fp32": str(args.fp32_model.resolve()),
            "int8": str(args.int8_model.resolve()),
            "qif_replacements": replacement_count,
            "fp32_node_count": sum(fp32_counts.values()),
            "int8_node_count": sum(int8_counts.values()),
            "quantize_linear_nodes": int8_counts.get("QuantizeLinear", 0),
            "dequantize_linear_nodes": int8_counts.get("DequantizeLinear", 0),
        },
        "samples": samples,
    }
    args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
    with args.smoke_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not allclose_passed:
        raise RuntimeError(
            f"FP32 ORT failed allclose(rtol={args.rtol}, atol={args.atol}); see {args.smoke_report}"
        )
    if not argmax_passed:
        raise RuntimeError(
            f"FP32 argmax agreement {min_fp32_argmax_agreement:.8f} is below "
            f"{args.min_argmax_agreement:.8f}; see {args.smoke_report}"
        )
    return report


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export, static-quantize, and smoke-test ANN SpikingLETNet_shallow_max"
    )
    parser.add_argument("command", choices=("export", "quantize", "smoke", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-split", type=Path, default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--fp32-model", type=Path, default=DEFAULT_FP32_MODEL)
    parser.add_argument("--preprocessed-model", type=Path, default=DEFAULT_PREPROCESSED_MODEL)
    parser.add_argument("--int8-model", type=Path, default=DEFAULT_INT8_MODEL)
    parser.add_argument("--smoke-report", type=Path, default=DEFAULT_SMOKE_REPORT)
    parser.add_argument("--classes", type=positive_int, default=6)
    parser.add_argument("--opset", type=positive_int, default=18)
    parser.add_argument("--calibration-samples", type=positive_int, default=300)
    parser.add_argument("--smoke-samples", type=positive_int, default=3)
    parser.add_argument("--threads", type=positive_int, default=52)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.9999)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_argmax_agreement <= 1.0:
        raise ValueError("--min-argmax-agreement must be in [0, 1]")
    if not math.isfinite(args.rtol) or not math.isfinite(args.atol) or args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be finite and non-negative")

    if args.command in {"export", "all"}:
        export_fp32(args)
    if args.command in {"quantize", "all"}:
        quantize_int8(args)
    if args.command in {"smoke", "all"}:
        smoke_test(args)


if __name__ == "__main__":
    main()
