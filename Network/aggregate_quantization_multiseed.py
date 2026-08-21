#!/usr/bin/env python3
"""Aggregate the complete three-seed W4A4 validation results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


NETWORK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NETWORK_DIR.parent
SEEDS = (1234, 2345, 3456)
METHODS = ("qad", "ste", "lsq", "ewgs")
CLASS_NAMES = ("background", "facade", "road", "vegetation", "vehicle", "roof")
BUDGETS = {"qad": 127, "ste": 16, "lsq": 16, "ewgs": 16}
LABELS = {"qad": "QAD", "ste": "STE-QAT", "lsq": "LSQ-QAT", "ewgs": "EWGS-QAT"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed1234-result",
        type=Path,
        default=REPO_ROOT / "quantization_comparison_results/seed1234/comparison.json",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "quantization_multiseed_results/udd/w4a4_v1/per_seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "quantization_multiseed_results/udd/w4a4_v1/aggregate",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_seed_payloads(seed1234: Path, result_root: Path) -> dict[int, dict[str, Any]]:
    paths = {
        1234: seed1234,
        2345: result_root / "seed2345/comparison.json",
        3456: result_root / "seed3456/comparison.json",
    }
    payloads: dict[int, dict[str, Any]] = {}
    for seed, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = set(METHODS) - set(payload.get("results", {}))
        if missing:
            raise RuntimeError(f"seed {seed} missing methods: {sorted(missing)}")
        for method in METHODS:
            result = payload["results"][method]
            if int(result.get("evaluated_batches", -1)) != 424:
                raise RuntimeError(f"seed {seed} {method} is not a full validation")
        payloads[seed] = payload
    return payloads


def summarize(payloads: dict[int, dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        raw = {
            seed: float(payloads[seed]["results"][method]["miou"])
            for seed in SEEDS
        }
        per_class = {
            name: {
                "values": {
                    seed: float(payloads[seed]["results"][method]["per_class_iou"][name])
                    for seed in SEEDS
                }
            }
            for name in CLASS_NAMES
        }
        for item in per_class.values():
            values = list(item["values"].values())
            item["mean"] = statistics.mean(values)
            item["sample_sd"] = statistics.stdev(values)
        values = list(raw.values())
        methods[method] = {
            "epochs_per_seed": BUDGETS[method],
            "values": raw,
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "best_epochs": {
                seed: int(payloads[seed]["results"][method]["checkpoint"]["epoch"])
                for seed in SEEDS
            },
            "per_class": per_class,
        }

    paired: dict[str, Any] = {}
    for baseline in ("ste", "lsq", "ewgs"):
        values = {
            seed: methods["qad"]["values"][seed] - methods[baseline]["values"][seed]
            for seed in SEEDS
        }
        paired[baseline] = {
            "values": values,
            "mean": statistics.mean(values.values()),
            "sample_sd": statistics.stdev(values.values()),
            "qad_wins": sum(value > 0 for value in values.values()),
        }
    return {"seeds": list(SEEDS), "methods": methods, "paired_qad_minus_baseline": paired}


def write_raw_csv(payloads: dict[int, dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "epochs_per_seed",
        "seed",
        "miou",
        "best_epoch",
        *[f"iou_{name}" for name in CLASS_NAMES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHODS:
            for seed in SEEDS:
                result = payloads[seed]["results"][method]
                writer.writerow(
                    {
                        "method": method,
                        "epochs_per_seed": BUDGETS[method],
                        "seed": seed,
                        "miou": result["miou"],
                        "best_epoch": result["checkpoint"]["epoch"],
                        **{
                            f"iou_{name}": result["per_class_iou"][name]
                            for name in CLASS_NAMES
                        },
                    }
                )


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# UDD W4A4 量化训练方法三随机种子结果",
        "",
        "三个训练 seed：1234、2345、3456。所有指标均来自完整 UDD validation",
        "（8,478条目、424 batches、400×400、T=1）。",
        "",
        "> QAD每个 seed训练127 epochs；其余方法训练16 epochs。该表是不同训练路线的",
        "> 描述性对比，不能把全部差异归因于量化算法本身。",
        "",
        "| Method | Epochs/seed | seed 1234 | seed 2345 | seed 3456 | mIoU mean ± sample SD | Best epochs |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for method in METHODS:
        item = summary["methods"][method]
        values = item["values"]
        best = ", ".join(str(item["best_epochs"][seed]) for seed in SEEDS)
        lines.append(
            f"| {LABELS[method]} | {BUDGETS[method]} | "
            f"{values[1234]:.6f} | {values[2345]:.6f} | {values[3456]:.6f} | "
            f"{item['mean']:.6f} ± {item['sample_sd']:.6f} | {best} |"
        )

    lines.extend(["", "## QAD相对基线的同-seed差值", "", "| Baseline | Mean ± sample SD | QAD wins | seed 1234 | seed 2345 | seed 3456 |", "|---|---:|---:|---:|---:|---:|"])
    for baseline in ("ste", "lsq", "ewgs"):
        item = summary["paired_qad_minus_baseline"][baseline]
        values = item["values"]
        lines.append(
            f"| {LABELS[baseline]} | {item['mean']:.6f} ± {item['sample_sd']:.6f} | "
            f"{item['qad_wins']}/3 | {values[1234]:.6f} | {values[2345]:.6f} | {values[3456]:.6f} |"
        )

    lines.extend(["", "## 逐类三-seed汇总", ""])
    for method in METHODS:
        lines.extend([f"### {LABELS[method]}", "", "| Class | Mean | Sample SD |", "|---|---:|---:|"])
        for name in CLASS_NAMES:
            item = summary["methods"][method]["per_class"][name]
            lines.append(f"| {name} | {item['mean']:.6f} | {item['sample_sd']:.6f} |")
        lines.append("")

    lines.extend(
        [
            "## 解释限制",
            "",
            "- n=3，只报告原始值和描述性统计，不宣称严格统计显著性。",
            "- 所有量化训练都从同一个FP32 checkpoint开始，未覆盖FP32预训练方差。",
            "- 当前结果来自validation集合，不是独立test集合。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payloads = load_seed_payloads(args.seed1234_result.resolve(), args.result_root.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(payloads)
    (output_dir / "comparison_multiseed.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_raw_csv(payloads, output_dir / "comparison_multiseed.csv")
    write_markdown(summary, output_dir / "report.md")


if __name__ == "__main__":
    main()
