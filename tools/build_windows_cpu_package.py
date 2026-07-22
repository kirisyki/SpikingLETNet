#!/usr/bin/env python3
"""Populate the Windows deployment bundle with models and stratified UDD subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = PROJECT_ROOT / "windows_cpu_deployment"
DEFAULT_TRAIN = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/train_patches.txt")
DEFAULT_VAL = Path("/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scene_key(path: Path) -> str:
    stem = path.stem.removesuffix("_img").removesuffix("_mask")
    fields = stem.rsplit("_", 2)
    return fields[0] if len(fields) == 3 else stem


def read_split(path: Path) -> list[tuple[Path, Path]]:
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected image and mask")
        image, mask = map(Path, fields)
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"{path}:{line_number}: {image} or {mask}")
        entries.append((image, mask))
    return entries


def allocate(groups: dict[str, list[tuple[Path, Path]]], count: int) -> dict[str, int]:
    if count < len(groups):
        raise ValueError(f"count={count} cannot cover all {len(groups)} scenes")
    total = sum(len(items) for items in groups.values())
    exact = {name: count * len(items) / total for name, items in groups.items()}
    allocation = {name: max(1, math.floor(value)) for name, value in exact.items()}
    current = sum(allocation.values())
    order = sorted(groups, key=lambda name: (exact[name] - math.floor(exact[name]), name), reverse=True)
    index = 0
    while current < count:
        name = order[index % len(order)]
        if allocation[name] < len(groups[name]):
            allocation[name] += 1
            current += 1
        index += 1
    while current > count:
        name = order[index % len(order)]
        if allocation[name] > 1:
            allocation[name] -= 1
            current -= 1
        index += 1
    return allocation


def stratified(entries: list[tuple[Path, Path]], count: int) -> list[tuple[Path, Path]]:
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for entry in entries:
        groups[scene_key(entry[0])].append(entry)
    for items in groups.values():
        items.sort(key=lambda item: item[0].name)
    allocation = allocate(groups, count)
    selected = []
    for name in sorted(groups):
        items = groups[name]
        amount = allocation[name]
        indices = [min(len(items) - 1, math.floor((position + 0.5) * len(items) / amount)) for position in range(amount)]
        selected.extend(items[index] for index in indices)
    if len(selected) != count or len({item[0] for item in selected}) != count:
        raise RuntimeError("Stratified selection is not unique or has the wrong size")
    return selected


def copy_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "scene": scene_key(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--train-split", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--calibration-count", type=int, default=300)
    parser.add_argument("--validation-count", type=int, default=100)
    args = parser.parse_args()

    package = args.package.resolve()
    required_template = [package / "README_CN.md", package / "scripts/common.py"]
    if not all(path.is_file() for path in required_template):
        raise RuntimeError(f"Deployment template is incomplete: {package}")

    copies = {
        PROJECT_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_fp32_preprocessed.onnx": package / "models/SpikingLETNet_shallow_max_fp32_portable.onnx",
        PROJECT_ROOT / "onnx_models/SpikingLETNet_shallow_max_ann_int8_qdq.onnx": package / "models/SpikingLETNet_shallow_max_w8a8_qdq_portable.onnx",
        PROJECT_ROOT / "tensorrt_energy_analysis/models/SpikingLETNet_shallow_max_ann_fp32_trt.onnx": package / "models/SpikingLETNet_shallow_max_fp32_openvino_portable.onnx",
        PROJECT_ROOT / "tensorrt_energy_analysis/models/SpikingLETNet_shallow_max_ann_int8_qdq_trt.onnx": package / "models/SpikingLETNet_shallow_max_w8a8_qdq_openvino_portable.onnx",
        PROJECT_ROOT / "checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth": package / "provenance/model_best.pth",
        PROJECT_ROOT / "Network/configs/SpikingLETNet_shallow/1.3M.yaml": package / "provenance/1.3M.yaml",
        PROJECT_ROOT / "tools/spikingletnet_onnx_pipeline.py": package / "provenance/spikingletnet_onnx_pipeline.py",
        PROJECT_ROOT / "tensorrt_energy_analysis/rewrite_onnx.py": package / "provenance/rewrite_onnx.py",
        PROJECT_ROOT / "tensorrt_energy_analysis/models/rewrite_manifest.json": package / "provenance/rewrite_manifest.json",
        PROJECT_ROOT / "tensorrt_energy_analysis/models/validation.json": package / "provenance/rewrite_validation.json",
    }
    copied_artifacts = []
    for source, destination in copies.items():
        metadata = copy_file(source, destination)
        metadata.update({"destination": destination.relative_to(package).as_posix(), "source": source.relative_to(PROJECT_ROOT).as_posix()})
        copied_artifacts.append(metadata)

    train = read_split(args.train_split)
    val = read_split(args.val_split)
    calibration = stratified(train, args.calibration_count)
    validation = stratified(val, args.validation_count)

    calibration_manifest = []
    calibration_lines = []
    for image, _mask in calibration:
        destination = package / "data/calibration/images" / image.name
        metadata = copy_file(image, destination)
        metadata["relative_path"] = destination.relative_to(package).as_posix()
        calibration_manifest.append(metadata)
        calibration_lines.append(metadata["relative_path"])
    (package / "data/calibration/calibration.txt").write_text(
        "\n".join(calibration_lines) + "\n", encoding="utf-8", newline="\n"
    )

    validation_manifest = []
    validation_lines = []
    for image, mask in validation:
        image_destination = package / "data/validation/images" / image.name
        mask_destination = package / "data/validation/masks" / mask.name
        image_metadata = copy_file(image, image_destination)
        mask_metadata = copy_file(mask, mask_destination)
        image_relative = image_destination.relative_to(package).as_posix()
        mask_relative = mask_destination.relative_to(package).as_posix()
        validation_manifest.append(
            {
                "scene": scene_key(image),
                "image": {**image_metadata, "relative_path": image_relative},
                "mask": {**mask_metadata, "relative_path": mask_relative},
            }
        )
        validation_lines.append(f"{image_relative} {mask_relative}")
    (package / "data/validation/validation.txt").write_text(
        "\n".join(validation_lines) + "\n", encoding="utf-8", newline="\n"
    )

    subset_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "proportional scene-stratified sampling with evenly spaced samples inside each scene",
        "selection_is_deterministic": True,
        "calibration": {
            "source_entries": len(train),
            "source_scenes": len({scene_key(item[0]) for item in train}),
            "selected_entries": len(calibration),
            "selected_scenes": len({scene_key(item[0]) for item in calibration}),
            "files": calibration_manifest,
        },
        "validation": {
            "source_entries": len(val),
            "source_scenes": len({scene_key(item[0]) for item in val}),
            "selected_entries": len(validation),
            "selected_scenes": len({scene_key(item[0]) for item in validation}),
            "pairs": validation_manifest,
        },
    }
    (package / "data/subset_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifact_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "SpikingLETNet_shallow_max",
        "checkpoint": "provenance/model_best.pth",
        "input": {"name": "image", "dtype": "float32", "shape": [1, 3, 400, 400]},
        "output": {"name": "logits", "dtype": "float32", "shape": [1, 6, 400, 400]},
        "semantics": "single-step ANN/QIF equivalent; QIF bin=False; fixed batch and resolution",
        "accuracy_is_acceptance_criterion": False,
        "default_provider": "CPUExecutionProvider",
        "optional_provider": "OpenVINOExecutionProvider",
        "artifacts": copied_artifacts,
        "data": {
            "calibration_images": len(calibration),
            "validation_pairs": len(validation),
        },
    }
    (package / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"package": str(package), "artifacts": len(copied_artifacts), "calibration": len(calibration), "validation": len(validation)}, indent=2))


if __name__ == "__main__":
    main()
