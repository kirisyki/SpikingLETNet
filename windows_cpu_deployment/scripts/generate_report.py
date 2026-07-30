#!/usr/bin/env python3
"""Create a compact Chinese summary from any completed benchmark stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import RESULTS_DIR, read_json, utc_now, write_json


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100.0:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", choices=("cpu", "openvino"), default=["cpu"])
    args = parser.parse_args()
    lines = [
        "# SpikingLETNet_shallow_max Windows CPU 测试结果",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        "主口径：batch=1、固定 400×400、仅统计 `session.run()`；能耗口径按结果文件中记录的采集后端解释。",
        "",
    ]
    summary = {"providers": {}}
    for provider in args.providers:
        provider_summary = {}
        latency_path = RESULTS_DIR / f"latency_{provider}.json"
        energy_candidates = [
            RESULTS_DIR / f"energy_{provider}.json",
            RESULTS_DIR / f"energy_hwinfo_{provider}.json",
        ]
        existing_energy = [path for path in energy_candidates if path.is_file()]
        energy_path = (
            max(existing_energy, key=lambda path: path.stat().st_mtime_ns)
            if existing_energy
            else None
        )
        lines.extend([f"## {provider}", ""])
        if latency_path.is_file():
            latency = read_json(latency_path)
            provider_summary["latency"] = latency["comparison"]
            lines.extend(
                [
                    "| 模型 | Median latency (ms) | P95 (ms) | images/s |",
                    "|---|---:|---:|---:|",
                ]
            )
            for label in ("fp32", "w8a8"):
                row = latency["models"][label]
                lines.append(
                    f"| {label.upper()} | {row['median_ms']:.4f} | {row['p95_ms']:.4f} | {row['images_per_second']:.3f} |"
                )
            lines.extend(
                [
                    "",
                    f"W8A8 延迟加速比：`{latency['comparison']['w8a8_speedup_over_fp32_median']:.4f}×`。",
                    "",
                ]
            )
            lines.extend(["辅助端到端中位延迟（PNG 解码到推理输出）：", ""])
            for label in ("fp32", "w8a8"):
                e2e = latency["models"][label].get("end_to_end_observational")
                if e2e:
                    lines.append(f"- {label.upper()}: `{e2e['median_ms']:.4f} ms`")
            lines.append("")
        else:
            lines.extend(["尚未生成延迟结果。", ""])
        if energy_path is not None:
            energy = read_json(energy_path)
            backend = energy.get("measurement_backend", "intel_pcm")
            provider_summary["energy"] = {
                "measurement_backend": backend,
                "comparison": energy.get("comparison"),
            }
            if backend == "hwinfo_csv_power_integration":
                lines.extend(
                    [
                        "能耗采集：`HWiNFO CPU Package Power CSV 梯形积分`（不是 PCM/RAPL 计数器差值）。",
                        "",
                        "| 模型 | HWiNFO Package J/image | Dynamic J/image | Package W | 有效窗 | 稳定 |",
                        "|---|---:|---:|---:|---:|:---:|",
                    ]
                )
                for label in ("fp32", "w8a8"):
                    row = energy["models"][label]
                    if row.get("valid_trials", 0):
                        package_energy = row["hwinfo_cpu_package_j_per_image_mean"]
                        dynamic_energy = row["hwinfo_dynamic_cpu_package_j_per_image_mean"]
                        package_power = row["hwinfo_cpu_package_power_w_mean"]
                        stable_text = "是" if row["stable"] else "否"
                        valid_trials = row["valid_trials"]
                        lines.append(
                            f"| {label.upper()} | {package_energy:.6f} | "
                            f"{dynamic_energy:.6f} | {package_power:.3f} | "
                            f"{valid_trials} | {stable_text} |"
                        )
                    else:
                        lines.append(f"| {label.upper()} | N/A | N/A | N/A | 0 | 否 |")
                reduction = (energy.get("comparison") or {}).get(
                    "w8a8_hwinfo_cpu_package_energy_reduction_fraction"
                )
            else:
                lines.extend(
                    [
                        "能耗采集：`Intel PCM CPU Package 能量计数器`。",
                        "",
                        "| 模型 | Package J/image | Dynamic J/image | Package W | 稳定 |",
                        "|---|---:|---:|---:|:---:|",
                    ]
                )
                for label in ("fp32", "w8a8"):
                    row = energy["models"][label]
                    if row.get("valid_trials", 0):
                        package_energy = row["package_j_per_image_mean"]
                        dynamic_energy = row["dynamic_j_per_image_mean"]
                        package_power = row["package_power_w_mean"]
                        stable_text = "是" if row["stable"] else "否"
                        lines.append(
                            f"| {label.upper()} | {package_energy:.6f} | "
                            f"{dynamic_energy:.6f} | {package_power:.3f} | {stable_text} |"
                        )
                    else:
                        lines.append(f"| {label.upper()} | N/A | N/A | N/A | 否（无有效试验） |")
                reduction = (energy.get("comparison") or {}).get(
                    "w8a8_package_energy_reduction_fraction"
                )
            lines.extend(
                ["", f"W8A8 Package 能耗变化：`{percent(reduction)}`（正值表示降低）。", ""]
            )
        else:
            lines.extend(["尚未生成能耗结果。", ""])
        summary["providers"][provider] = provider_summary

    lines.extend(
        [
            "## 解释限制",
            "",
            "- W8A8 是 QDQ 混合精度模型；不支持或不适合量化的算子保留 FP32。",
            "- CPU EP 与 OpenVINO EP 使用不同的优化器和内核，结果不可交叉拼接比较。",
            "- 本实验不设精度验收门槛；数值比较仅用于确认输出有效。",
            "- HWiNFO 功率日志积分与 PCM/RAPL 计数器差值是不同测量方法，指标必须保留 `HWiNFO` 标签，不能混合。",
            "- 若结果标记为不稳定，应先处理热降频或后台负载，再引用能耗结论。",
            "",
        ]
    )
    report_path = RESULTS_DIR / "result_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_json(RESULTS_DIR / "result_summary.json", summary)
    print(f"PASS: summary written to {report_path}")


if __name__ == "__main__":
    main()
