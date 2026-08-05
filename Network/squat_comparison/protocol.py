"""Frozen protocol and filesystem guards for the QAD-vs-SQUAT experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
NETWORK_DIR = PACKAGE_DIR.parent
REPO_ROOT = NETWORK_DIR.parent
DEFAULT_CONFIG = NETWORK_DIR / "configs/SpikingLETNet_shallow/1.3M.yaml"
HISTORICAL_COMPARISON = (
    REPO_ROOT / "quantization_comparison_results/seed1234/comparison.json"
)
OUTPUT_ROOT = REPO_ROOT / "squat_experiment_outputs/seed1234"

PROTECTED_ROOTS = tuple(
    (REPO_ROOT / item).resolve()
    for item in (
        "checkpoint",
        "QAT_checkpoint",
        "ternary_QAT_checkpoint",
        "quantization_comparison_results",
        "Network/quantization",
        "Network/quantization_comparison",
    )
)

CLASS_NAMES = ("background", "facade", "road", "vegetation", "vehicle", "roof")


@dataclass(frozen=True)
class ExperimentProtocol:
    seed: int = 1234
    classes: int = 6
    time_steps: int = 8
    input_height: int = 400
    input_width: int = 400
    effective_train_batch: int = 64
    validation_batch: int = 20
    workers: int = 8
    fp32_max_epochs: int = 300
    squat_max_epochs: int = 127
    fp32_min_epochs: int = 100
    squat_min_epochs: int = 40
    fp32_learning_rate: float = 1e-3
    squat_learning_rate: float = 3e-4
    eta_min: float = 1e-6
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    historical_qad_miou: float = 0.6195473169172718
    historical_fp32_qif_miou: float = 0.6785024141945621
    comparable_band: float = 0.005
    expected_weight_layers: int = 71
    expected_validation_samples: int = 8478

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_writable_experiment_path(path: Path) -> Path:
    resolved = path.resolve()
    output = OUTPUT_ROOT.resolve()
    if not is_relative_to(resolved, output):
        raise PermissionError(f"refusing write outside isolated output root: {resolved}")
    for protected in PROTECTED_ROOTS:
        if is_relative_to(resolved, protected):
            raise PermissionError(f"refusing write below protected root: {protected}")
    return resolved


def prepare_new_directory(path: Path) -> Path:
    resolved = assert_writable_experiment_path(path)
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = assert_writable_experiment_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, resolved)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_historical_results(protocol: ExperimentProtocol) -> dict[str, Any]:
    payload = json.loads(HISTORICAL_COMPARISON.read_text(encoding="utf-8"))
    qad = payload["results"]["qad"]
    fp32 = payload["results"]["fp32"]
    if not math.isclose(qad["miou"], protocol.historical_qad_miou, abs_tol=1e-12):
        raise RuntimeError("frozen QAD mIoU changed")
    if not math.isclose(
        fp32["miou"], protocol.historical_fp32_qif_miou, abs_tol=1e-12
    ):
        raise RuntimeError("frozen FP32-QIF mIoU changed")
    return {
        "source": str(HISTORICAL_COMPARISON),
        "source_sha256": sha256_file(HISTORICAL_COMPARISON),
        "qad": {
            "miou": qad["miou"],
            "per_class_iou": qad["per_class_iou"],
        },
        "fp32_qif": {
            "miou": fp32["miou"],
            "per_class_iou": fp32["per_class_iou"],
        },
        "execution_policy": "read frozen metrics only; no QAD train/eval/convert",
    }


def scaled_epoch_budgets(
    full_hours: float, protocol: ExperimentProtocol
) -> tuple[int, int, float]:
    if full_hours <= 0:
        raise ValueError("full_hours must be positive")
    if full_hours < 48.0:
        return protocol.fp32_max_epochs, protocol.squat_max_epochs, 1.0
    scale = 44.0 / full_hours
    return (
        max(protocol.fp32_min_epochs, math.floor(protocol.fp32_max_epochs * scale)),
        max(protocol.squat_min_epochs, math.floor(protocol.squat_max_epochs * scale)),
        scale,
    )

