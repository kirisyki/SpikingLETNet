#!/usr/bin/env python3
"""Measure residual W4A4 sensitivity at QAD and STE trained master weights."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
os.chdir(NETWORK_DIR)

import evaluate_quantization_comparison as common  # noqa: E402
from qad_distortion_repair.statistics import summarize_self_distortion  # noqa: E402
from quantization.int4_selfbuild import QLayer  # noqa: E402
from quantization_comparison.layer_adapter import temporal_average  # noqa: E402
from quantization_distortion_probe.qif_capture import (  # noqa: E402
    QIFCapture,
    compare_qif_captures,
)
from quantization_distortion_probe.statistics import CLASS_NAMES  # noqa: E402
from run_qad_distortion_repair_probe import (  # noqa: E402
    COUNT_FIELDS,
    EXPECTED_VALIDATION_PATCHES,
    FEATURE_SUM_FIELDS,
    DEFAULT_SPLIT,
    DEFAULT_STE_CHECKPOINT,
    confusion_matrix,
    feature_counts,
    rows_from_aggregates,
    sha256_file,
    source_id_from_path,
    write_json,
)
from spikingjelly.activation_based import functional  # noqa: E402


DEFAULT_OUTPUT = REPO_ROOT / "qad_distortion_repair_results/self_sensitivity_full_20260806"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ste-checkpoint", type=Path, default=DEFAULT_STE_CHECKPOINT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270807)
    return parser.parse_args()


def build_fp_shadow(model: torch.nn.Module) -> torch.nn.Module:
    shadow = copy.deepcopy(model)
    count = 0
    for module in shadow.modules():
        if isinstance(module, QLayer):
            module.quant = False
            module.activation_quant = False
            count += 1
    if count != 71:
        raise RuntimeError(f"expected 71 QLayers, observed {count}")
    return shadow


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    ).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size != 20:
        raise ValueError("use the historical validation batch size of 20")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("an explicit CUDA device index is required")
    torch.cuda.set_device(device.index)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    qad_w4, qad_meta = common.build_qad()
    ste_w4, ste_meta = common.build_baseline("ste", args.ste_checkpoint)
    models = {
        "qad_fp": build_fp_shadow(qad_w4),
        "qad_w4a4": qad_w4,
        "ste_fp": build_fp_shadow(ste_w4),
        "ste_w4a4": ste_w4,
    }
    for model in models.values():
        model.to(device).eval()
        functional.set_step_mode(model, step_mode="m")
        functional.reset_net(model)
    captures = {name: QIFCapture(model) for name, model in models.items()}

    _, loader = common.build_dataset_test(
        "udd", num_workers=args.workers, none_gt=False, batch_size=args.batch_size
    )
    if len(loader.dataset) != EXPECTED_VALIDATION_PATCHES:
        raise RuntimeError(f"unexpected validation size: {len(loader.dataset)}")
    dataset_paths = [sample[0] for sample in loader.dataset.samples]
    split_paths = [line.split()[0] for line in args.split_file.read_text().splitlines()]
    if dataset_paths != split_paths:
        raise RuntimeError("canonical loader order differs from split file")

    manifest = {
        "format": "qad-self-distortion-probe-v1",
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "design": "within-checkpoint FP-operator versus W4A4 sensitivity at fixed trained master weights",
            "dataset": "UDD6 validation",
            "validation_patches": len(loader.dataset),
            "statistical_unit": "35 original source images",
            "batch_size": args.batch_size,
            "time_steps": 1,
            "fp_shadow_definition": "same QLayer master weights and BN; QLayer weight/input fake quantization disabled",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "max_batches": args.max_batches,
            "known_confound": "QAD observed 127 training epochs; STE-QAT observed 16 epochs",
            "interpretation_boundary": "FP shadows are local sensitivity references, not separately trained task baselines",
        },
        "checkpoints": {
            "qad": {**qad_meta, "sha256_recomputed": sha256_file(Path(qad_meta["path"]))},
            "ste": {**ste_meta, "sha256_recomputed": sha256_file(Path(ste_meta["path"]))},
        },
        "device": {"requested": args.device, "name": torch.cuda.get_device_name(device)},
    }
    write_json(output_dir / "manifest.json", manifest)

    source_qif: dict[tuple[str, str, str], dict[str, int | float]] = defaultdict(
        lambda: defaultdict(int)
    )
    source_feature: dict[tuple[str, str, int], dict[str, int | float]] = defaultdict(
        lambda: defaultdict(float)
    )
    source_confusion: dict[tuple[str, str], np.ndarray] = defaultdict(
        lambda: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    )
    source_flip: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    sources: set[str] = set()
    active_names: list[str] | None = None
    completed_patches = 0
    completed_batches = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_index, (images_cpu, labels_cpu) in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            batch_size = images_cpu.shape[0]
            patch_indices = list(range(completed_patches, completed_patches + batch_size))
            image_paths = [dataset_paths[index] for index in patch_indices]
            batch_sources = [source_id_from_path(path) for path in image_paths]
            sources.update(batch_sources)
            images = images_cpu.to(device, non_blocking=True).unsqueeze(0)
            labels = labels_cpu.long().to(device, non_blocking=True)

            for method in ("qad", "ste"):
                fp_name = f"{method}_fp"
                w4_name = f"{method}_w4a4"
                captures[fp_name].clear()
                functional.reset_net(models[fp_name])
                fp_output, fp_features = models[fp_name].forward_qat(images)
                fp_output = temporal_average(fp_output)
                fp_features = [temporal_average(feature) for feature in fp_features]
                fp_prediction = fp_output.argmax(1)
                functional.reset_net(models[fp_name])

                captures[w4_name].clear()
                functional.reset_net(models[w4_name])
                w4_output, w4_features = models[w4_name].forward_qat(images)
                w4_output = temporal_average(w4_output)
                w4_features = [temporal_average(feature) for feature in w4_features]
                w4_prediction = w4_output.argmax(1)
                functional.reset_net(models[w4_name])

                comparisons = compare_qif_captures(
                    captures[fp_name], captures[w4_name], batch_size=batch_size
                )
                names = [str(row["layer"]) for row in comparisons]
                if active_names is None:
                    active_names = names
                    if len(active_names) != 78:
                        raise RuntimeError(f"expected 78 active QIF nodes, got {len(active_names)}")
                elif names != active_names:
                    raise RuntimeError(f"active QIF set changed for {method}")

                for sample, source in enumerate(batch_sources):
                    valid = (labels[sample] >= 0) & (labels[sample] < len(CLASS_NAMES))
                    source_confusion[(fp_name, source)] += confusion_matrix(
                        labels[sample], fp_prediction[sample]
                    )
                    source_confusion[(w4_name, source)] += confusion_matrix(
                        labels[sample], w4_prediction[sample]
                    )
                    source_flip[(method, source)]["flip_count"] += int(
                        ((w4_prediction[sample] != fp_prediction[sample]) & valid)
                        .sum()
                        .item()
                    )
                    source_flip[(method, source)]["valid_count"] += int(valid.sum().item())

                for comparison in comparisons:
                    layer = str(comparison["layer"])
                    for sample, source in enumerate(batch_sources):
                        for field in COUNT_FIELDS:
                            source_qif[(method, source, layer)][field] += int(
                                comparison[field][sample].item()
                            )

                for feature_index, (reference, variant) in enumerate(
                    zip(fp_features, w4_features), start=1
                ):
                    counts = feature_counts(reference, variant, batch_size)
                    for sample, source in enumerate(batch_sources):
                        for field in FEATURE_SUM_FIELDS:
                            value = counts[field]
                            source_feature[(method, source, feature_index)][field] += (
                                float(value[sample].item())
                                if torch.is_tensor(value)
                                else int(value)
                            )
                captures[fp_name].clear()
                captures[w4_name].clear()
                del fp_output, fp_features, fp_prediction
                del w4_output, w4_features, w4_prediction, comparisons

            del images, labels
            completed_batches += 1
            completed_patches += batch_size
            if completed_batches == 1 or completed_batches % 20 == 0:
                print(
                    f"batches={completed_batches}/{len(loader)} patches={completed_patches} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    qif_rows = rows_from_aggregates(source_qif, ("method", "source_id", "layer"))
    feature_rows = rows_from_aggregates(
        source_feature, ("method", "source_id", "feature_index")
    )
    confusion_rows = [
        {"method": method, "source_id": source, "matrix": matrix.tolist()}
        for (method, source), matrix in sorted(source_confusion.items())
    ]
    flip_rows = [
        {"method": method, "source_id": source, **counts}
        for (method, source), counts in sorted(source_flip.items())
    ]
    summary = summarize_self_distortion(
        qif_rows=qif_rows,
        feature_rows=feature_rows,
        confusion_rows=confusion_rows,
        flip_rows=flip_rows,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_json(output_dir / "source_qif_aggregates.json", qif_rows)
    write_json(output_dir / "source_feature_aggregates.json", feature_rows)
    write_json(output_dir / "source_confusion.json", confusion_rows)
    write_json(output_dir / "source_prediction_flips.json", flip_rows)
    write_json(output_dir / "analysis_summary.json", summary)
    for capture in captures.values():
        capture.close()
    manifest.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_batches": completed_batches,
            "completed_patches": completed_patches,
            "complete_validation": completed_patches == EXPECTED_VALIDATION_PATCHES,
            "elapsed_seconds": time.perf_counter() - started,
            "source_count": len(sources),
            "active_qif_layer_count": len(active_names or []),
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

