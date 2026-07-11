#!/usr/bin/env python3
"""Build reproducible v2 energy tables from estimator per-layer outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml

from energy_accounting import sha256_file, strict_json_dump, validate_finite_nonnegative


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENERGY_SETS = {
    "set_a": PROJECT_ROOT / "tools/energy_params_set_a_fp32_int4_conservative.yaml",
    "set_b": PROJECT_ROOT / "tools/energy_params_set_b_fp32_int4_custom.yaml",
}


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value!r}")
    return result


def parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH or PATH")
    return label, Path(raw_path)


def load_energy_set(path: Path) -> Dict[str, Dict[str, float]]:
    raw = yaml.safe_load(path.read_text())
    energy = raw.get("energy_pj", raw) if isinstance(raw, dict) else None
    if not isinstance(energy, dict):
        raise ValueError(f"invalid energy parameter file: {path}")
    result: Dict[str, Dict[str, float]] = {}
    for precision in ("fp", "int4"):
        values = energy.get(precision)
        if not isinstance(values, dict) or "mac" not in values or "ac" not in values:
            raise ValueError(f"{path} must define {precision}.mac and {precision}.ac")
        result[precision] = {
            "mac": finite_float(values["mac"], f"{precision}.mac"),
            "ac": finite_float(values["ac"], f"{precision}.ac"),
        }
    validate_finite_nonnegative(result)
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty estimator layer CSV: {path}")
    return rows


def read_summaries(path: Path) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    payload = json.loads(path.read_text())
    summaries = payload.get("summaries") if isinstance(payload, dict) else None
    if not isinstance(summaries, list):
        raise ValueError(f"invalid estimator summary JSON: {path}")
    result: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for summary in summaries:
        key = (str(summary["variant"]), str(summary["checkpoint_kind"]))
        if key in result:
            raise ValueError(f"duplicate estimator summary key {key} in {path}")
        result[key] = summary
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1e6:
            return f"{value:,.0f}"
        return f"{value:.8g}"
    return str(value)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def group_layer_rows(rows: Iterable[Mapping[str, str]]) -> Dict[Tuple[str, str], list[Mapping[str, str]]]:
    grouped: Dict[Tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["checkpoint_kind"]))].append(row)
    return dict(grouped)


def summarize_group(
    run_label: str,
    group_key: Tuple[str, str],
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    energy_sets: Mapping[str, Mapping[str, Mapping[str, float]]],
    source_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    variant, checkpoint_kind = group_key
    timestep = int(summary["timestep"])
    dense_by_precision = defaultdict(float)
    sop_by_precision = defaultdict(float)
    spike_sum = 0.0
    spike_firing_rate_sum = 0.0
    spike_elems = 0
    spike_neuron_sites = 0.0
    spike_nonzero = 0
    spike_layers = 0

    for row in rows:
        precision = str(row["precision"])
        if precision not in {"fp", "int4"}:
            raise ValueError(f"unsupported precision {precision!r} in {source_csv}")
        dense = finite_float(row["dense_macs_charged_per_image"], "dense_macs_charged_per_image")
        sop = finite_float(row["sop_per_image"], "sop_per_image")
        if dense < 0 or sop < 0:
            raise ValueError(f"negative operation count in {source_csv}: {row.get('layer')}")
        dense_by_precision[precision] += dense
        sop_by_precision[precision] += sop
        if row["mode"] == "spike" and int(row["input_elems"]) > 0:
            spike_layers += 1
            spike_sum += finite_float(row["input_positive_sum"], "input_positive_sum")
            spike_elems += int(row["input_elems"])
            spike_firing_rate_sum += finite_float(
                row.get("input_firing_rate_sum", float(row["input_positive_sum"]) / timestep),
                "input_firing_rate_sum",
            )
            spike_neuron_sites += finite_float(
                row.get("input_neuron_sites", int(row["input_elems"]) / timestep),
                "input_neuron_sites",
            )
            spike_nonzero += int(row["input_nonzero"])

    common = {
        "run": run_label,
        "variant": variant,
        "checkpoint_kind": checkpoint_kind,
        "precision": str(summary["precision"]),
        "timestep": timestep,
        "processed_images": int(summary["processed_images"]),
        "classification_mode": str(summary["classification_mode"]),
        "sop_method": str(summary["sop_method"]),
    }
    operation = {
        **common,
        "dense_macs_charged_per_image": sum(dense_by_precision.values()),
        "sops_per_image": sum(sop_by_precision.values()),
        "fp_dense_macs_per_image": dense_by_precision["fp"],
        "int4_dense_macs_per_image": dense_by_precision["int4"],
        "fp_sops_per_image": sop_by_precision["fp"],
        "int4_sops_per_image": sop_by_precision["int4"],
        "source_layer_csv": str(source_csv),
    }
    sparsity = {
        **common,
        "spike_layers": spike_layers,
        "input_elements": spike_elems,
        "input_neuron_sites": spike_neuron_sites,
        "input_spike_count_sum": spike_sum,
        "input_firing_rate_sum": spike_firing_rate_sum,
        "input_nonzero": spike_nonzero,
        "mean_spike_count_per_input": spike_sum / spike_elems if spike_elems else None,
        "configured_firing_rate": spike_sum / spike_elems / timestep if spike_elems else None,
        "sop_density_per_executed_mac": spike_firing_rate_sum / spike_elems if spike_elems else None,
        "nonzero_ratio": spike_nonzero / spike_elems if spike_elems else None,
        "zero_ratio": 1.0 - spike_nonzero / spike_elems if spike_elems else None,
        "source_layer_csv": str(source_csv),
    }
    energies: list[dict[str, Any]] = []
    for label, energy in energy_sets.items():
        mac_pj = sum(dense_by_precision[p] * energy[p]["mac"] for p in ("fp", "int4"))
        sop_pj = sum(sop_by_precision[p] * energy[p]["ac"] for p in ("fp", "int4"))
        energies.append({
            **common,
            "energy_set": label,
            "mac_energy_mj_per_image": mac_pj / 1e9,
            "sop_energy_mj_per_image": sop_pj / 1e9,
            "total_core_energy_mj_per_image": (mac_pj + sop_pj) / 1e9,
            "source_layer_csv": str(source_csv),
        })
    return energies, operation, sparsity


def comparison_rows(
    inference_rows: Sequence[Mapping[str, Any]], baseline_run: str
) -> list[dict[str, Any]]:
    baseline: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for row in inference_rows:
        if row["run"] == baseline_run:
            baseline[(str(row["variant"]), str(row["checkpoint_kind"]), str(row["energy_set"]))] = row
    comparisons = []
    for row in inference_rows:
        if row["run"] == baseline_run:
            continue
        key = (str(row["variant"]), str(row["checkpoint_kind"]), str(row["energy_set"]))
        reference = baseline.get(key)
        if reference is None:
            continue
        base_energy = float(reference["total_core_energy_mj_per_image"])
        energy = float(row["total_core_energy_mj_per_image"])
        comparisons.append({
            "baseline_run": baseline_run,
            "candidate_run": row["run"],
            "variant": row["variant"],
            "checkpoint_kind": row["checkpoint_kind"],
            "energy_set": row["energy_set"],
            "baseline_timestep": reference["timestep"],
            "candidate_timestep": row["timestep"],
            "baseline_energy_mj_per_image": base_energy,
            "candidate_energy_mj_per_image": energy,
            "candidate_over_baseline": energy / base_energy if base_energy else None,
        })
    return comparisons


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_label_path, metavar="[LABEL=]DIR")
    parser.add_argument("--energy-set", action="append", type=parse_label_path, metavar="LABEL=YAML")
    parser.add_argument("--baseline-run", default=None, help="Run label used as comparison denominator; defaults to first run.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "energy/theoretical/v2/derived_20_images")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    energy_paths = dict(args.energy_set) if args.energy_set else DEFAULT_ENERGY_SETS
    energy_sets = {label: load_energy_set(path) for label, path in energy_paths.items()}
    inference: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    sparsity: list[dict[str, Any]] = []

    for run_label, run_dir in args.run:
        layer_csv = run_dir / "layer_energy.csv"
        summary_json = run_dir / "summary_energy.json"
        if not layer_csv.is_file() or not summary_json.is_file():
            raise FileNotFoundError(f"{run_dir} must contain layer_energy.csv and summary_energy.json")
        grouped = group_layer_rows(read_csv(layer_csv))
        summaries = read_summaries(summary_json)
        if set(grouped) != set(summaries):
            raise ValueError(
                f"layer/summary key mismatch in {run_dir}: "
                f"layers={sorted(grouped)} summaries={sorted(summaries)}"
            )
        for key in sorted(grouped):
            energy_rows, operation, spike = summarize_group(
                run_label, key, grouped[key], summaries[key], energy_sets, layer_csv
            )
            inference.extend(energy_rows)
            operations.append(operation)
            sparsity.append(spike)

    baseline_run = args.baseline_run or args.run[0][0]
    comparisons = comparison_rows(inference, baseline_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "inference_energy.csv", inference)
    write_csv(args.output_dir / "mac_sop_summary.csv", operations)
    write_csv(args.output_dir / "spike_sparsity_summary.csv", sparsity)
    write_csv(args.output_dir / "energy_comparison.csv", comparisons)
    write_markdown(
        args.output_dir / "inference_energy.md", inference,
        ("run", "variant", "checkpoint_kind", "timestep", "energy_set", "total_core_energy_mj_per_image"),
    )
    write_markdown(
        args.output_dir / "mac_sop_summary.md", operations,
        ("run", "variant", "checkpoint_kind", "timestep", "dense_macs_charged_per_image", "sops_per_image"),
    )
    write_markdown(
        args.output_dir / "spike_sparsity_summary.md", sparsity,
        ("run", "variant", "checkpoint_kind", "timestep", "mean_spike_count_per_input", "configured_firing_rate", "sop_density_per_executed_mac", "zero_ratio"),
    )
    manifest = {
        "schema_version": 2,
        "runs": [{"label": label, "path": str(path)} for label, path in args.run],
        "baseline_run": baseline_run,
        "energy_sets": [
            {"label": label, "path": str(path), "sha256": sha256_file(path), "energy_pj": energy_sets[label]}
            for label, path in energy_paths.items()
        ],
        "outputs": [
            "inference_energy.csv", "mac_sop_summary.csv", "spike_sparsity_summary.csv",
            "energy_comparison.csv", "inference_energy.md", "mac_sop_summary.md",
            "spike_sparsity_summary.md",
        ],
    }
    strict_json_dump(manifest, args.output_dir / "report_manifest.json")
    print(f"wrote v2 derived reports to {args.output_dir}")


if __name__ == "__main__":
    main()
