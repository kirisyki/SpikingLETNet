"""Frozen protocol and integrity checks for the W4A4 comparison."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NETWORK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NETWORK_DIR.parent
PROTOCOL_VERSION = "w4a4-comparison-v2-short16"
METHODS = ("ste", "lsq", "ewgs")

DEFAULT_CONFIG = NETWORK_DIR / "configs/SpikingLETNet_shallow/1.3M.yaml"
DEFAULT_FP_CHECKPOINT = (
    REPO_ROOT
    / "checkpoint/udd/"
    "SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/"
    "model_best.pth"
)
DEFAULT_QAD_CHECKPOINT = (
    REPO_ROOT
    / "QAT_checkpoint/udd/"
    "SpikingLETNet_shallow_maxbs64gpu1_trainval20260612-135249/"
    "model_q_best.pth"
)
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "quantization_comparison_checkpoint/udd/seed1234"
DEFAULT_RESULT_DIR = REPO_ROOT / "quantization_comparison_results/seed1234"

# Frozen immediately before implementation. These files remain read-only.
PROTECTED_HASHES = {
    "Network/QAT_snn_STE.py": (
        "6f568d7195ce7ee7682c2f9db94f1ec7dfbef4acdcacd9b6eca126374ddc6386"
    ),
    "Network/QAT_snn.py": (
        "385b9907dc2460da6b7f56d771c526cffe5b7cd45101426da85477898c434c20"
    ),
    "Network/quantization/int4_selfbuild.py": (
        "58f51d0da0f1088533f34bbe27a82bb46954d9ca2295924a1abd399c1a550e3e"
    ),
    "Network/model/SpikingLETNet_shallow_max.py": (
        "44ec0db9fb8044373bb0073a312af39dfb22a54faee5ad64ade6540936b1a907"
    ),
    "Network/evaluate_precision_comparison.py": (
        "d07c775f36321458246cd66e891c206adc2705dd7e95574cdf6bbb4f23eb6785"
    ),
}


@dataclass(frozen=True)
class ExperimentProtocol:
    """Reader-visible, immutable settings shared by all production runs."""

    method: str
    model: str = "SpikingLETNet_shallow_max"
    dataset: str = "udd"
    classes: int = 6
    input_height: int = 400
    input_width: int = 400
    train_type: str = "trainval"
    seed: int = 1234
    time_steps: int = 1
    workers: int = 6
    train_batch_size: int = 64
    validation_batch_size: int = 20
    run_epochs: int = 16
    schedule_epochs: int = 150
    batches_per_epoch: int = 401
    model_learning_rate: float = 1e-3
    ewgs_quantizer_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    poly_exponent: float = 0.9
    weight_bits: int = 4
    activation_bits: int = 4
    quantization_levels: int = 16
    expected_quantized_layers: int = 71
    quantization_granularity: str = "per_tensor"
    first_last_layer_exemption: bool = False
    random_mirror: bool = True
    random_scale: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.run_epochs <= 0 or self.schedule_epochs < self.run_epochs:
            raise ValueError("invalid run/schedule epoch budget")
        if self.time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if self.weight_bits != 4 or self.activation_bits != 4:
            raise ValueError("this protocol is frozen to W4A4")

    @property
    def schedule_iterations(self) -> int:
        return self.schedule_epochs * self.batches_per_epoch

    @property
    def run_iterations(self) -> int:
        return self.run_epochs * self.batches_per_epoch

    def lr_multiplier(self, global_iteration: int) -> float:
        if not 0 <= global_iteration < self.schedule_iterations:
            raise ValueError(
                f"global_iteration={global_iteration} is outside schedule "
                f"[0, {self.schedule_iterations})"
            )
        remaining = 1.0 - global_iteration / self.schedule_iterations
        return math.pow(remaining, self.poly_exponent)

    def model_lr(self, global_iteration: int) -> float:
        return self.model_learning_rate * self.lr_multiplier(global_iteration)

    def quantizer_lr(self, global_iteration: int) -> float:
        base = (
            self.ewgs_quantizer_learning_rate
            if self.method == "ewgs"
            else self.model_learning_rate
        )
        return base * self.lr_multiplier(global_iteration)

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_version": PROTOCOL_VERSION, **asdict(self)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protected_hashes() -> dict[str, str]:
    """Fail before execution if a protected historical file changed."""

    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for relative, expected in PROTECTED_HASHES.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            mismatches.append(f"missing protected file: {relative}")
            continue
        actual = sha256_file(path)
        observed[relative] = actual
        if actual != expected:
            mismatches.append(
                f"{relative}: expected {expected}, observed {actual}"
            )
    if mismatches:
        raise RuntimeError("protected-file integrity failure:\n" + "\n".join(mismatches))
    return observed


def source_manifest(protocol: ExperimentProtocol) -> dict[str, Any]:
    config = DEFAULT_CONFIG.resolve()
    fp_checkpoint = DEFAULT_FP_CHECKPOINT.resolve()
    qad_checkpoint = DEFAULT_QAD_CHECKPOINT.resolve()
    for path in (config, fp_checkpoint, qad_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "protocol": protocol.to_dict(),
        "sources": {
            "config": {
                "path": str(config),
                "sha256": sha256_file(config),
            },
            "fp_checkpoint": {
                "path": str(fp_checkpoint),
                "sha256": sha256_file(fp_checkpoint),
            },
            "qad_checkpoint": {
                "path": str(qad_checkpoint),
                "sha256": sha256_file(qad_checkpoint),
            },
        },
        "protected_files": validate_protected_hashes(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
