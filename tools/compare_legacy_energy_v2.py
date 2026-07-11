#!/usr/bin/env python3
"""Reconcile legacy ANN/SNN arithmetic tables with version-2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from energy_accounting import sha256_file, strict_json_dump


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV while tolerating legacy pretty-print whitespace."""
    with path.open(newline="") as handle:
        rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value!r}")
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    def fmt(value: Any) -> str:
        return f"{value:.8g}" if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n")


def delta_row(common: Mapping[str, Any], metric: str, legacy: float, current: float) -> dict[str, Any]:
    return {
        **common,
        "metric": metric,
        "legacy": legacy,
        "v2": current,
        "v2_over_legacy": current / legacy if legacy else None,
        "percent_change": (current / legacy - 1.0) * 100.0 if legacy else None,
    }


def operation_reconciliation(
    legacy_t1: Sequence[Mapping[str, str]],
    legacy_t8: Sequence[Mapping[str, str]],
    v2_operations: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    v2 = {
        (row["run"], row["variant"], row["checkpoint_kind"]): row
        for row in v2_operations
    }
    rows: list[dict[str, Any]] = []
    for legacy in legacy_t1:
        if legacy["network"] == "FP32":
            kind = "fp"
        elif legacy["network"] == "INT4_SetA_conservative":
            kind = "qat"
        else:
            continue
        variant = legacy["variant"]
        current = v2[("qif_t1", variant, kind)]
        rows.append(delta_row(
            {"phase": "t1_qif_dense", "variant": variant, "checkpoint_kind": kind},
            "dense_macs_per_image",
            number(legacy["macs_per_image"], "legacy T1 MACs"),
            number(current["dense_macs_charged_per_image"], "v2 T1 MACs"),
        ))

    for legacy in legacy_t8:
        kind = {"FP": "fp", "INT4": "qat"}[legacy["network"]]
        variant = legacy["variant"]
        current = v2[("snn_t8", variant, kind)]
        common = {"phase": "t8_snn", "variant": variant, "checkpoint_kind": kind}
        rows.append(delta_row(
            common,
            "dense_macs_charged_per_image",
            number(legacy["dense_macs_charged_per_image"], "legacy dense MACs"),
            number(current["dense_macs_charged_per_image"], "v2 dense MACs"),
        ))
        rows.append(delta_row(
            common,
            "sops_per_image",
            number(legacy["sops_per_image"], "legacy SOPs"),
            number(current["sops_per_image"], "v2 SOPs"),
        ))
    return rows


def legacy_t1_energy_index(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str, str], float]:
    result: Dict[Tuple[str, str, str], float] = {}
    for row in rows:
        network = row["network"]
        variant = row["variant"]
        energy_mj = number(row["compute_energy_uJ_per_image"], "legacy T1 energy") / 1000.0
        if network == "FP32":
            result[(variant, "fp", "set_a")] = energy_mj
            result[(variant, "fp", "set_b")] = energy_mj
        elif network == "INT4_SetA_conservative":
            result[(variant, "qat", "set_a")] = energy_mj
        elif network == "INT4_SetB_custom":
            result[(variant, "qat", "set_b")] = energy_mj
    return result


def legacy_t8_energy_index(
    set_a_rows: Sequence[Mapping[str, str]], set_b_rows: Sequence[Mapping[str, str]]
) -> Dict[Tuple[str, str, str], float]:
    result: Dict[Tuple[str, str, str], float] = {}
    for label, rows in (("set_a", set_a_rows), ("set_b", set_b_rows)):
        for row in rows:
            kind = {"FP": "fp", "INT4": "qat"}[row["network"]]
            result[(row["variant"], kind, label)] = (
                number(row["total_compute_energy_uJ_per_image"], "legacy T8 energy") / 1000.0
            )
    return result


def energy_reconciliation(
    legacy_t1: Mapping[Tuple[str, str, str], float],
    legacy_t8: Mapping[Tuple[str, str, str], float],
    v2_energy: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for current in v2_energy:
        run = current["run"]
        key = (current["variant"], current["checkpoint_kind"], current["energy_set"])
        legacy = (legacy_t1 if run == "qif_t1" else legacy_t8)[key]
        rows.append(delta_row(
            {
                "phase": "t1_qif_dense" if run == "qif_t1" else "t8_snn",
                "variant": key[0],
                "checkpoint_kind": key[1],
                "energy_set": key[2],
            },
            "total_core_energy_mj_per_image",
            legacy,
            number(current["total_core_energy_mj_per_image"], "v2 energy"),
        ))
    return rows


def measurement_reconciliation(legacy_path: Path, current_path: Path) -> list[dict[str, Any]]:
    legacy = json.loads(legacy_path.read_text())
    current = json.loads(current_path.read_text())
    metric_map = {
        "gross_j_per_image": "gross_j_per_image",
        "net_j_per_image": "net_j_per_image",
        "images_per_second": "images_per_second",
    }
    common = {
        "legacy_driver": legacy["gpu_before"]["driver"],
        "v2_driver": current["gpu"]["driver"],
        "legacy_trials": 1,
        "v2_trials": int(current["trials"]),
        "legacy_energy_source": "trapezoidal_power_integration",
        "v2_energy_source": ",".join(sorted({row["energy_source"] for row in current["trial_results"]})),
        "comparison_scope": "protocol_and_environment_combined_difference",
    }
    rows: list[dict[str, Any]] = []
    for metric, legacy_key in metric_map.items():
        legacy_value = number(legacy[legacy_key], f"legacy measured {metric}")
        current_value = number(current["summary"][metric]["median"], f"v2 measured {metric}")
        rows.append(delta_row(
            common,
            metric,
            legacy_value,
            current_value,
        ))
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-ann", type=Path, default=PROJECT_ROOT / "energy/theoretical/legacy/qif_t1_dense/ann_all_mac_energy.csv")
    parser.add_argument("--legacy-snn-ops", type=Path, default=PROJECT_ROOT / "energy/theoretical/legacy/snn_t8_20_images/mac_sop_summary.csv")
    parser.add_argument("--legacy-snn-set-a", type=Path, default=PROJECT_ROOT / "energy/theoretical/legacy/snn_t8_20_images/inference_energy_set_a_fp32_int4_conservative.csv")
    parser.add_argument("--legacy-snn-set-b", type=Path, default=PROJECT_ROOT / "energy/theoretical/legacy/snn_t8_20_images/inference_energy_set_b_fp32_int4_custom.csv")
    parser.add_argument("--v2-operations", type=Path, default=PROJECT_ROOT / "energy/theoretical/v2/derived_20_images/mac_sop_summary.csv")
    parser.add_argument("--v2-energy", type=Path, default=PROJECT_ROOT / "energy/theoretical/v2/derived_20_images/inference_energy.csv")
    parser.add_argument("--legacy-measurement", type=Path, default=PROJECT_ROOT / "energy/measurements/legacy/pro6000_qif_t1_dense_fp32_5000_images/result.json")
    parser.add_argument("--v2-measurement", type=Path, default=PROJECT_ROOT / "energy/measurements/v2/pro6000_qif_t1_dense_fp32_5x5000_images/result.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "energy/theoretical/v2/reconciliation_20_images")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    legacy_ann_rows = read_csv(args.legacy_ann)
    operations = operation_reconciliation(
        legacy_ann_rows, read_csv(args.legacy_snn_ops), read_csv(args.v2_operations)
    )
    energies = energy_reconciliation(
        legacy_t1_energy_index(legacy_ann_rows),
        legacy_t8_energy_index(read_csv(args.legacy_snn_set_a), read_csv(args.legacy_snn_set_b)),
        read_csv(args.v2_energy),
    )
    measurements = measurement_reconciliation(
        args.legacy_measurement, args.v2_measurement
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "operation_reconciliation.csv", operations)
    write_csv(args.output_dir / "energy_reconciliation.csv", energies)
    write_csv(args.output_dir / "measurement_reconciliation.csv", measurements)
    write_markdown(
        args.output_dir / "operation_reconciliation.md", operations,
        ("phase", "variant", "checkpoint_kind", "metric", "legacy", "v2", "v2_over_legacy", "percent_change"),
    )
    write_markdown(
        args.output_dir / "energy_reconciliation.md", energies,
        ("phase", "variant", "checkpoint_kind", "energy_set", "legacy", "v2", "v2_over_legacy", "percent_change"),
    )
    write_markdown(
        args.output_dir / "measurement_reconciliation.md", measurements,
        ("metric", "legacy", "v2", "v2_over_legacy", "percent_change", "legacy_driver",
         "v2_driver", "legacy_energy_source", "v2_energy_source", "comparison_scope"),
    )
    inputs = [
        args.legacy_ann, args.legacy_snn_ops, args.legacy_snn_set_a,
        args.legacy_snn_set_b, args.v2_operations, args.v2_energy,
        args.legacy_measurement, args.v2_measurement,
    ]
    strict_json_dump(
        {
            "schema_version": 2,
            "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in inputs],
            "outputs": [
                "operation_reconciliation.csv", "operation_reconciliation.md",
                "energy_reconciliation.csv", "energy_reconciliation.md",
                "measurement_reconciliation.csv", "measurement_reconciliation.md",
            ],
        },
        args.output_dir / "reconciliation_manifest.json",
    )
    print(f"wrote legacy/v2 reconciliation to {args.output_dir}")


if __name__ == "__main__":
    main()
