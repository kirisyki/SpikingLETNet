"""Shared deterministic accounting helpers for LETNet energy tools."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml


VALID_MODES = {"dense", "spike"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def strict_json_dump(value: Any, path: Path) -> None:
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def validate_finite_nonnegative(values: Mapping[str, Mapping[str, float]]) -> None:
    for precision, params in values.items():
        for name, value in params.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"energy parameter {precision}.{name} must be finite and non-negative: {value}")


@dataclass(frozen=True)
class LayerRule:
    rule_id: str
    pattern: re.Pattern[str]
    mode: str
    variants: Optional[frozenset[str]] = None
    op_types: Optional[frozenset[str]] = None

    def matches(self, layer: str, variant: str, op_type: str) -> bool:
        if self.variants is not None and variant not in self.variants:
            return False
        if self.op_types is not None and op_type not in self.op_types:
            return False
        return self.pattern.fullmatch(layer) is not None


class LayerModePolicy:
    def __init__(self, version: int, rules: Sequence[LayerRule], source: Path) -> None:
        self.version = int(version)
        self.rules = list(rules)
        self.source = source
        self.sha256 = sha256_file(source)

    @classmethod
    def load(cls, path: Path) -> "LayerModePolicy":
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or not isinstance(raw.get("rules"), list):
            raise ValueError(f"invalid layer policy: {path}")
        rules = []
        ids = set()
        for item in raw["rules"]:
            rule_id = str(item["id"])
            if rule_id in ids:
                raise ValueError(f"duplicate layer policy rule id: {rule_id}")
            ids.add(rule_id)
            mode = str(item["mode"])
            if mode not in VALID_MODES:
                raise ValueError(f"invalid layer policy mode {mode!r} in {rule_id}")
            rules.append(
                LayerRule(
                    rule_id=rule_id,
                    pattern=re.compile(str(item["pattern"])),
                    mode=mode,
                    variants=frozenset(map(str, item["variants"])) if item.get("variants") else None,
                    op_types=frozenset(map(str, item["op_types"])) if item.get("op_types") else None,
                )
            )
        return cls(version=int(raw["version"]), rules=rules, source=path)

    def classify(self, layer: str, variant: str, op_type: str) -> tuple[str, str]:
        matches = [rule for rule in self.rules if rule.matches(layer, variant, op_type)]
        if len(matches) != 1:
            matched_ids = [rule.rule_id for rule in matches]
            raise ValueError(
                f"layer policy expected exactly one match for {variant}:{layer} ({op_type}); "
                f"matched={matched_ids}"
            )
        return matches[0].mode, matches[0].rule_id

    def validate_inventory(self, inventory: Iterable[tuple[str, str, str]]) -> None:
        for variant, layer, op_type in inventory:
            self.classify(layer, variant, op_type)


def stable_sample_ids(paths: Sequence[str]) -> list[str]:
    return [hashlib.sha256(path.encode("utf-8")).hexdigest()[:16] for path in paths]


def cv_percent(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute CV of empty values")
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0 if all(value == 0 for value in values) else float("inf")
    if len(values) == 1:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / abs(mean) * 100.0
