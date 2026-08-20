"""Build validated, paper-ready artifacts from a completed efficiency run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_efficiency_comparison.analysis import bootstrap_median_ci


TITLE = "Integer-LIF + QAD 与直接 SNN 量化训练效率对比"
GIB = 2**30
EFFICIENCY_SQL = (
    "SELECT dataset_id, row_json FROM staged_efficiency_report "
    "ORDER BY dataset_id, row_index"
)
ACCURACY_SQL = (
    "SELECT dataset_id, row_json FROM staged_accuracy_report "
    "ORDER BY dataset_id, row_index"
)


def _sqlite_roundtrip(
    table: str, datasets: dict[str, list[dict[str, Any]]], sql: str
) -> dict[str, list[dict[str, Any]]]:
    if table not in {"staged_efficiency_report", "staged_accuracy_report"}:
        raise ValueError(f"unsupported staging table: {table}")
    connection = sqlite3.connect(":memory:")
    connection.execute(
        f"CREATE TABLE {table} (dataset_id TEXT, row_index INTEGER, row_json TEXT)"
    )
    for dataset_id, rows in datasets.items():
        connection.executemany(
            f"INSERT INTO {table} VALUES (?, ?, ?)",
            [
                (dataset_id, index, json.dumps(row, ensure_ascii=False, sort_keys=True))
                for index, row in enumerate(rows)
            ],
        )
    rebuilt: dict[str, list[dict[str, Any]]] = {key: [] for key in datasets}
    for dataset_id, row_json in connection.execute(sql):
        rebuilt[str(dataset_id)].append(json.loads(row_json))
    connection.close()
    if rebuilt != datasets:
        raise ValueError("SQLite provenance round trip changed report rows")
    return rebuilt


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group(summary: dict[str, Any], mode: str, method: str, time_steps: int) -> dict[str, Any]:
    matches = [
        row
        for row in summary["groups"]
        if row["mode"] == mode
        and row["method"] == method
        and int(row["time_steps"]) == time_steps
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one group for {(mode, method, time_steps)}, got {len(matches)}")
    return matches[0]


def _run_median(path: Path) -> float:
    run = _read_json(path)
    if run.get("status") != "ok":
        raise ValueError(f"non-successful measured run: {path}")
    return statistics.median(float(row["wall_seconds"]) for row in run["trials"])


def _paired_speedup(
    run_dir: Path, qad_pattern: str, squat_pattern: str, rounds: int
) -> dict[str, Any]:
    ratios = []
    for round_index in range(rounds):
        qad = _run_median(run_dir / qad_pattern.format(round=round_index))
        squat = _run_median(run_dir / squat_pattern.format(round=round_index))
        ratios.append(squat / qad)
    low, high = bootstrap_median_ci(ratios, seed=1234)
    return {
        "round_ratios": ratios,
        "median": statistics.median(ratios),
        "block_bootstrap_median_ci95_low": low,
        "block_bootstrap_median_ci95_high": high,
        "independent_pairs": len(ratios),
    }


def _phase(group: dict[str, Any], *names: str) -> float:
    return sum(float(group["phase_seconds"][name]["median"]) for name in names)


def validate_run(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    run_paths = sorted((output_dir / "runs").glob("*.json"))
    runs = [_read_json(path) for path in run_paths]
    statuses: dict[str, int] = {}
    for run in runs:
        status = str(run.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    if statuses != {"ok": 37, "oom": 5}:
        raise ValueError(f"unexpected run statuses: {statuses}")

    measured = [run for run in runs if run["mode"] != "probe"]
    if any(run.get("status") != "ok" for run in measured):
        raise ValueError("a measured non-probe run failed")
    if any(int(run["gradient_audit"]["parameters_with_grad"]) <= 0 for run in measured):
        raise ValueError("a measured run has no parameter gradients")
    qad_runs = [run for run in measured if run["method"] == "qad"]
    if any(int(run["gradient_audit"]["teacher_parameters_with_grad"]) != 0 for run in qad_runs):
        raise ValueError("the frozen QAD teacher received gradients")

    capacities = summary["capacities"]
    if capacities != {"qad": 32, "squat": 4}:
        raise ValueError(f"unexpected capacity result: {capacities}")
    return {
        "run_files": len(runs),
        "status_counts": statuses,
        "gradient_audit": "passed",
        "qad_teacher_gradient_audit": "passed",
        "capacity_probe": capacities,
    }


def build_validated_summary(
    output_dir: Path, summary: dict[str, Any], accuracy_source: Path
) -> dict[str, Any]:
    matched_qad = _group(summary, "matched", "qad", 1)
    matched_squat = _group(summary, "matched", "squat", 8)
    capacity_qad = _group(summary, "capacity", "qad", 1)
    capacity_squat = _group(summary, "capacity", "squat", 8)
    qif_ste = _group(summary, "qad_ablation", "qif_ste", 1)
    timestep = [_group(summary, "timestep_ablation", "squat", t) for t in (1, 2, 4, 8)]
    matched_comparison = next(row for row in summary["comparisons"] if row["mode"] == "matched")
    capacity_comparison = next(row for row in summary["comparisons"] if row["mode"] == "capacity")

    accuracy = _read_json(accuracy_source)
    qad_miou = float(accuracy["historical"]["qad"]["miou"])
    squat_miou = float(accuracy["new_results"]["w4m4s1_squat"]["miou"])
    if abs((qad_miou - squat_miou) + float(accuracy["comparison"]["delta_route"])) > 1e-12:
        raise ValueError("accuracy route delta does not reconcile")

    matched_paired = _paired_speedup(
        output_dir / "runs",
        "matched_qad_t1_p4_e4_r{round}.json",
        "matched_squat_t8_p4_e4_r{round}.json",
        5,
    )
    capacity_paired = _paired_speedup(
        output_dir / "runs",
        "capacity_qad_t1_p32_e64_r{round}.json",
        "capacity_squat_t8_p4_e64_r{round}.json",
        5,
    )

    epoch_by_method = {row["method"]: row for row in summary["epoch_projection"]}
    validation = validate_run(output_dir, summary)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
        "matched_batch": {
            "physical_batch": 4,
            "effective_batch": 4,
            "qad_peak_allocated_gib": matched_qad["memory"]["peak_allocated_bytes"] / GIB,
            "squat_peak_allocated_gib": matched_squat["memory"]["peak_allocated_bytes"] / GIB,
            "qad_memory_reduction_fraction": matched_comparison["qad_memory_reduction_fraction"],
            "squat_to_qad_memory_ratio": matched_comparison["squat_to_qad_memory_ratio"],
            "qad_update_seconds": matched_qad["wall_seconds"]["median"],
            "squat_update_seconds": matched_squat["wall_seconds"]["median"],
            "qad_images_per_second": matched_qad["images_per_second"]["median"],
            "squat_images_per_second": matched_squat["images_per_second"]["median"],
            "throughput_speedup_ratio_of_pooled_medians": matched_comparison["qad_throughput_speedup"],
            "paired_round_speedup": matched_paired,
        },
        "capacity_batch": {
            "effective_batch": 64,
            "qad_physical_batch": 32,
            "squat_physical_batch": 4,
            "physical_batch_capacity_ratio": 8.0,
            "memory_comparable": False,
            "qad_peak_allocated_gib": capacity_qad["memory"]["peak_allocated_bytes"] / GIB,
            "squat_peak_allocated_gib": capacity_squat["memory"]["peak_allocated_bytes"] / GIB,
            "qad_update_seconds": capacity_qad["wall_seconds"]["median"],
            "squat_update_seconds": capacity_squat["wall_seconds"]["median"],
            "qad_images_per_second": capacity_qad["images_per_second"]["median"],
            "squat_images_per_second": capacity_squat["images_per_second"]["median"],
            "throughput_speedup_ratio_of_pooled_medians": capacity_comparison["qad_throughput_speedup"],
            "paired_round_speedup": capacity_paired,
            "epoch_projection_minutes": {
                "qad": epoch_by_method["qad"]["projected_training_minutes"],
                "squat": epoch_by_method["squat"]["projected_training_minutes"],
            },
        },
        "mechanism": {
            "squat_t8_to_t1_peak_memory_ratio": (
                timestep[-1]["memory"]["peak_allocated_bytes"]
                / timestep[0]["memory"]["peak_allocated_bytes"]
            ),
            "squat_t8_to_t1_update_time_ratio": (
                timestep[-1]["wall_seconds"]["median"]
                / timestep[0]["wall_seconds"]["median"]
            ),
            "qad_teacher_kd_memory_overhead_gib": (
                matched_qad["memory"]["peak_allocated_bytes"]
                - qif_ste["memory"]["peak_allocated_bytes"]
            ) / GIB,
            "qad_to_qif_ste_update_time_ratio": (
                matched_qad["wall_seconds"]["median"] / qif_ste["wall_seconds"]["median"]
            ),
        },
        "accuracy_context": {
            "qad_miou": qad_miou,
            "squat_miou": squat_miou,
            "qad_minus_squat_miou": qad_miou - squat_miou,
            "single_seed": True,
            "causal_comparison": False,
            "source_sha256": _sha256(accuracy_source),
        },
    }
    return result


def _main_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "qad": "Integer-LIF + QAD",
        "squat": "Direct-LIF + QAT+SQUAT",
    }
    rows = []
    for mode in ("matched", "capacity"):
        for method, t in (("qad", 1), ("squat", 8)):
            group = _group(summary, mode, method, t)
            rows.append(
                {
                    "benchmark": "同物理 batch" if mode == "matched" else "同有效 batch",
                    "route": labels[method],
                    "chart_label": f"{'Matched' if mode == 'matched' else 'Capacity'} · {'QAD' if method == 'qad' else 'SQUAT'}",
                    "physical_batch": group["physical_batch"],
                    "effective_batch": group["effective_batch"],
                    "time_steps": group["time_steps"],
                    "temporal_gradient": "无时间展开 BPTT" if method == "qad" else "完整 BPTT",
                    "peak_allocated_gib": group["memory"]["peak_allocated_bytes"] / GIB,
                    "peak_reserved_gib": group["memory"]["peak_reserved_bytes"] / GIB,
                    "update_ms": group["wall_seconds"]["median"] * 1000,
                    "update_ci_low_ms": group["wall_seconds"]["block_bootstrap_median_ci95_low"] * 1000,
                    "update_ci_high_ms": group["wall_seconds"]["block_bootstrap_median_ci95_high"] * 1000,
                    "images_per_second": group["images_per_second"]["median"],
                    "rounds": group["rounds"],
                }
            )
    return rows


def _ablation_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for time_steps in (1, 2, 4, 8):
        group = _group(summary, "timestep_ablation", "squat", time_steps)
        rows.append(
            {
                "time_step": f"T={time_steps}",
                "time_steps": time_steps,
                "physical_batch": 4,
                "effective_batch": 4,
                "peak_allocated_gib": group["memory"]["peak_allocated_bytes"] / GIB,
                "update_ms": group["wall_seconds"]["median"] * 1000,
                "update_ci_low_ms": group["wall_seconds"]["block_bootstrap_median_ci95_low"] * 1000,
                "update_ci_high_ms": group["wall_seconds"]["block_bootstrap_median_ci95_high"] * 1000,
                "images_per_second": group["images_per_second"]["median"],
                "forward_ms": _phase(group, "snn_forward") * 1000,
                "backward_ms": _phase(group, "bptt_backward") * 1000,
            }
        )
    return rows


def _phase_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode in ("matched", "capacity"):
        qad = _group(summary, mode, "qad", 1)
        squat = _group(summary, mode, "squat", 8)
        rows.extend(
            [
                {
                    "benchmark": "同物理 batch" if mode == "matched" else "同有效 batch",
                    "route": "Integer-LIF + QAD",
                    "forward_ms": _phase(qad, "teacher_forward", "student_forward") * 1000,
                    "backward_ms": _phase(qad, "backward") * 1000,
                    "optimizer_ms": _phase(qad, "optimizer_step") * 1000,
                    "note": "teacher + student forward；无时间展开 BPTT",
                },
                {
                    "benchmark": "同物理 batch" if mode == "matched" else "同有效 batch",
                    "route": "Direct-LIF + QAT+SQUAT",
                    "forward_ms": _phase(squat, "snn_forward") * 1000,
                    "backward_ms": _phase(squat, "bptt_backward") * 1000,
                    "optimizer_ms": _phase(squat, "optimizer_step") * 1000,
                    "note": "T=8 recurrent forward + full BPTT",
                },
            ]
        )
    return rows


def _source_specs(generated_at: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "efficiency_summary",
            "label": "Formal efficiency summary",
            "path": "summary.json",
            "query": {
                "engine": "SQLite 3 (in-memory staging)",
                "language": "sql",
                "sql": EFFICIENCY_SQL,
                "tables_used": ["staged_efficiency_report (materialized from summary.json)"],
                "description": "Aggregate of isolated-process CUDA benchmark runs on one RTX 5090.",
                "executed_at": generated_at,
                "filters": [
                    "UDD real samples resized to 400×400",
                    "AMP disabled; data loading, validation, and checkpoint I/O excluded",
                    "Matched memory comparison uses physical batch 4 for both routes",
                ],
                "metric_definitions": [
                    "Peak allocated memory is the maximum torch.cuda.max_memory_allocated value across independent runs.",
                    "Update time is the pooled median of CUDA-synchronized optimizer-update wall times; 95% CIs resample run-level medians.",
                    "Throughput is effective batch divided by CUDA-synchronized update wall time.",
                ],
            },
        },
        {
            "id": "raw_trials",
            "label": "Per-update benchmark observations",
            "path": "raw_trials.csv",
            "query": {
                "engine": "Python 3.12",
                "language": "python",
                "description": "All successful measured optimizer updates, including CUDA event phase timings.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "protocol",
            "label": "Benchmark protocol and environment",
            "path": "protocol.json",
            "query": {
                "description": "Warmup, repetition, batch, and capacity-probe settings for the formal run.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "accuracy_context",
            "label": "Frozen route-level accuracy evidence",
            "path": "accuracy_evidence.json",
            "query": {
                "engine": "SQLite 3 (in-memory staging)",
                "language": "sql",
                "sql": ACCURACY_SQL,
                "tables_used": ["staged_accuracy_report (materialized from accuracy_evidence.json)"],
                "description": "Exact mIoU values copied from the frozen seed-1234 comparison artifact with original path and SHA-256 recorded.",
                "executed_at": generated_at,
                "filters": ["Frozen metrics only; no new accuracy training or evaluation in this benchmark"],
                "metric_definitions": ["mIoU is the unweighted mean of six UDD class IoUs."],
            },
        },
    ]


def build_artifact(
    summary: dict[str, Any], validated: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    generated_at = validated["generated_at"]
    main_rows = _main_rows(summary)
    ablation_rows = _ablation_rows(summary)
    phase_rows = _phase_rows(summary)
    matched = validated["matched_batch"]
    capacity = validated["capacity_batch"]
    mechanism = validated["mechanism"]
    accuracy = validated["accuracy_context"]
    sources = _source_specs(generated_at)
    efficiency_headlines = [
        {
            "memory_reduction": matched["qad_memory_reduction_fraction"],
            "memory_ratio": matched["squat_to_qad_memory_ratio"],
            "matched_speedup": matched["throughput_speedup_ratio_of_pooled_medians"],
            "matched_time_reduction": 1 - matched["qad_update_seconds"] / matched["squat_update_seconds"],
            "capacity_speedup": capacity["throughput_speedup_ratio_of_pooled_medians"],
            "capacity_ratio": capacity["physical_batch_capacity_ratio"],
        }
    ]
    accuracy_headlines = [
        {
            "qad_miou": accuracy["qad_miou"],
            "squat_miou": accuracy["squat_miou"],
            "miou_gap": accuracy["qad_minus_squat_miou"],
        }
    ]
    cards = [
        {
            "id": "matched_memory",
            "description": "physical batch=4、effective batch=4 下，QAD 相对 SQUAT 的 PyTorch 峰值 allocated memory 降幅。",
            "dataset": "efficiency_headlines",
            "sourceId": "efficiency_summary",
            "metrics": [
                {"label": "同 batch 峰值显存降低", "field": "memory_reduction", "format": "percent"},
                {"label": "SQUAT/QAD", "field": "memory_ratio", "format": "number"},
            ],
        },
        {
            "id": "matched_speed",
            "description": "physical/effective batch 均为 4 时的 CUDA 同步训练吞吐倍率。",
            "dataset": "efficiency_headlines",
            "sourceId": "efficiency_summary",
            "metrics": [
                {"label": "同 batch 吞吐加速（×）", "field": "matched_speedup", "format": "number"},
                {"label": "update 时间降低", "field": "matched_time_reduction", "format": "percent"},
            ],
        },
        {
            "id": "capacity_speed",
            "description": "effective batch=64 时，分别采用各路线可运行的最大物理 batch。",
            "dataset": "efficiency_headlines",
            "sourceId": "efficiency_summary",
            "metrics": [{"label": "同有效 batch 吞吐加速（×）", "field": "capacity_speedup", "format": "number"}],
        },
        {
            "id": "capacity_batch",
            "description": "RTX 5090 上容量探测所得最大可运行 physical batch：QAD=32、SQUAT=4。",
            "dataset": "efficiency_headlines",
            "sourceId": "efficiency_summary",
            "metrics": [{"label": "物理 batch 容量优势（×）", "field": "capacity_ratio", "format": "number"}],
        },
        {
            "id": "accuracy_gap",
            "description": "冻结单种子历史结果；训练预算与路线不同，仅作有效性上下文，不是受控因果比较。",
            "dataset": "accuracy_headlines",
            "sourceId": "accuracy_context",
            "metrics": [
                {"label": "QAD−SQUAT mIoU", "field": "miou_gap", "format": "number", "signed": True},
                {"label": "QAD mIoU", "field": "qad_miou", "format": "number"},
                {"label": "SQUAT mIoU", "field": "squat_miou", "format": "number"},
            ],
        },
    ]
    charts = [
        {
            "id": "throughput_chart",
            "title": "两种训练口径下的吞吐量",
            "subtitle": "Matched 使用 physical/effective batch=4；Capacity 使用 effective batch=64 和各路线最大可运行 physical batch。",
            "intent": "comparison",
            "question": "QAD 在同 batch 和同有效 batch 下分别能提供多大吞吐？",
            "rationale": "单指标柱状图直接比较每秒处理图像数；模式与路线合并为显式类别，避免不受控的分组推断。",
            "comparisonContext": {"denominator": "CUDA-synchronized optimizer update wall time", "grain": "route × benchmark mode", "unit": "images/s"},
            "type": "bar",
            "dataset": "main_efficiency",
            "sourceId": "efficiency_summary",
            "encodings": {
                "x": {"field": "chart_label", "type": "nominal", "label": "基准口径与路线"},
                "y": {"field": "images_per_second", "type": "quantitative", "label": "吞吐量", "unit": "images/s"},
                "tooltip": [
                    {"field": "physical_batch", "type": "quantitative", "label": "Physical batch"},
                    {"field": "effective_batch", "type": "quantitative", "label": "Effective batch"},
                    {"field": "update_ms", "type": "quantitative", "label": "Update time", "unit": "ms"},
                ],
            },
            "valueFormat": "number",
            "unit": "images/s",
            "layout": "full",
            "maxRows": 4,
            "settings": {"showValues": True, "sort": "none", "categoryLabelPolicy": "wrap"},
        },
        {
            "id": "timestep_memory_chart",
            "title": "SQUAT 时间步消融：峰值 allocated memory",
            "subtitle": "同一 Direct-LIF + QAT+SQUAT 路线，physical/effective batch=4；T 从 1 增至 8。",
            "intent": "comparison",
            "question": "时间展开如何改变 recurrent SNN 量化训练的峰值显存？",
            "rationale": "四个离散时间步适合从零基线的单系列柱状图。",
            "comparisonContext": {"baseline": "T=1", "grain": "time step", "unit": "GiB"},
            "type": "bar",
            "dataset": "timestep_ablation",
            "sourceId": "efficiency_summary",
            "encodings": {
                "x": {"field": "time_step", "type": "ordinal", "label": "时间步"},
                "y": {"field": "peak_allocated_gib", "type": "quantitative", "label": "峰值 allocated memory", "unit": "GiB"},
                "tooltip": [
                    {"field": "update_ms", "type": "quantitative", "label": "Update time", "unit": "ms"},
                    {"field": "images_per_second", "type": "quantitative", "label": "Throughput", "unit": "images/s"},
                ],
            },
            "valueFormat": "number",
            "unit": "GiB",
            "layout": "half",
            "maxRows": 4,
            "settings": {"showValues": True, "sort": "none"},
        },
        {
            "id": "timestep_time_chart",
            "title": "SQUAT 时间步消融：训练更新时间",
            "subtitle": "与显存消融相同的模型、输入与 batch；误差范围见精确结果表。",
            "intent": "comparison",
            "question": "时间展开如何改变 recurrent SNN 量化训练的更新时间？",
            "rationale": "单系列柱状图呈现相同配置下随 T 增长的 CUDA 同步 wall time。",
            "comparisonContext": {"baseline": "T=1", "grain": "time step", "unit": "ms/update"},
            "type": "bar",
            "dataset": "timestep_ablation",
            "sourceId": "efficiency_summary",
            "encodings": {
                "x": {"field": "time_step", "type": "ordinal", "label": "时间步"},
                "y": {"field": "update_ms", "type": "quantitative", "label": "Update time", "unit": "ms"},
                "tooltip": [
                    {"field": "forward_ms", "type": "quantitative", "label": "Forward", "unit": "ms"},
                    {"field": "backward_ms", "type": "quantitative", "label": "Backward/BPTT", "unit": "ms"},
                ],
            },
            "valueFormat": "number",
            "unit": "ms/update",
            "layout": "half",
            "maxRows": 4,
            "settings": {"showValues": True, "sort": "none"},
        },
    ]
    tables = [
        {
            "id": "main_table",
            "title": "主比较精确结果",
            "subtitle": "显存倍率只在同 physical batch 行之间解释；Capacity 两行的峰值显存不可直接作路线倍率。",
            "dataset": "main_efficiency",
            "sourceId": "efficiency_summary",
            "defaultSort": {"field": "benchmark", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "benchmark", "label": "口径", "type": "text"},
                {"field": "route", "label": "路线", "type": "text"},
                {"field": "physical_batch", "label": "Physical B", "format": "number"},
                {"field": "effective_batch", "label": "Effective B", "format": "number"},
                {"field": "time_steps", "label": "T", "format": "number"},
                {"field": "peak_allocated_gib", "label": "Peak GiB", "format": "number"},
                {"field": "update_ms", "label": "ms/update", "format": "number"},
                {"field": "images_per_second", "label": "images/s", "format": "number"},
            ],
        },
        {
            "id": "phase_table",
            "title": "前向、反向与优化器阶段耗时",
            "subtitle": "阶段为 CUDA event 中位数；QAD forward 同时包含冻结教师与量化学生，SQUAT backward 为 full BPTT。",
            "dataset": "phase_breakdown",
            "sourceId": "efficiency_summary",
            "defaultSort": {"field": "benchmark", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "benchmark", "label": "口径", "type": "text"},
                {"field": "route", "label": "路线", "type": "text"},
                {"field": "forward_ms", "label": "Forward ms", "format": "number"},
                {"field": "backward_ms", "label": "Backward ms", "format": "number"},
                {"field": "optimizer_ms", "label": "Optimizer ms", "format": "number"},
                {"field": "note", "label": "定义", "type": "text"},
            ],
        },
        {
            "id": "ablation_table",
            "title": "SQUAT 时间步消融精确结果",
            "subtitle": "每个 T 为 3 个独立进程轮次、每轮 10 个 measured updates；95% CI 对轮次中位数做 block bootstrap。",
            "dataset": "timestep_ablation",
            "sourceId": "efficiency_summary",
            "defaultSort": {"field": "time_step", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "time_step", "label": "T", "type": "text"},
                {"field": "peak_allocated_gib", "label": "Peak GiB", "format": "number"},
                {"field": "update_ms", "label": "Median ms", "format": "number"},
                {"field": "update_ci_low_ms", "label": "CI low ms", "format": "number"},
                {"field": "update_ci_high_ms", "label": "CI high ms", "format": "number"},
                {"field": "images_per_second", "label": "images/s", "format": "number"},
            ],
        },
        {
            "id": "accuracy_table",
            "title": "冻结精度证据（支持性上下文）",
            "subtitle": "单种子且训练预算、神经元、时间步、量化对象与蒸馏设置不同；只说明现有完整路线结果。",
            "dataset": "accuracy_rows",
            "sourceId": "accuracy_context",
            "defaultSort": {"field": "miou", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "route", "label": "路线", "type": "text"},
                {"field": "miou", "label": "mIoU", "format": "number"},
                {"field": "evidence_role", "label": "证据角色", "type": "text"},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}", "layout": "full"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "efficiency_summary",
            "body": (
                "## Executive Summary\n\n"
                f"在 RTX 5090、400×400 输入、AMP 关闭的受控 microbenchmark 中，Integer-LIF + QAD 在同 physical/effective batch=4 时只占 {matched['qad_peak_allocated_gib']:.3f} GiB，而 Direct-LIF + QAT+SQUAT（T=8、full BPTT）为 {matched['squat_peak_allocated_gib']:.3f} GiB；峰值 allocated memory 降低 {matched['qad_memory_reduction_fraction']:.2%}，吞吐提高 {matched['throughput_speedup_ratio_of_pooled_medians']:.3f}×。在 effective batch=64、各自最大可运行 physical batch 下，QAD 达 {capacity['qad_images_per_second']:.2f} images/s，SQUAT 为 {capacity['squat_images_per_second']:.2f} images/s，即 {capacity['throughput_speedup_ratio_of_pooled_medians']:.3f}×。容量探测显示 QAD 可运行 physical batch=32，SQUAT 为 4。"
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["matched_memory", "matched_speed", "capacity_speed", "capacity_batch", "accuracy_gap"], "layout": "full"},
        {
            "id": "key_findings",
            "type": "markdown",
            "layout": "full",
            "sourceId": "efficiency_summary",
            "body": (
                "## Key Findings\n\n"
                f"1. **公平显存口径支持主要结论。** 同 physical batch=4 时，SQUAT/QAD 的峰值 allocated memory 比为 {matched['squat_to_qad_memory_ratio']:.3f}×。\n"
                f"2. **同有效 batch 的训练速度差距更大。** effective batch=64 时，单次 optimizer update 从 {capacity['squat_update_seconds']:.3f}s 降至 {capacity['qad_update_seconds']:.3f}s；按 25,700 个样本、402 个 updates 的纯训练投影约为 {capacity['epoch_projection_minutes']['squat']:.2f} 分钟/epoch 对 {capacity['epoch_projection_minutes']['qad']:.2f} 分钟/epoch。\n"
                f"3. **独立轮次支持速度差异。** 同 batch 的成对轮次加速中位数为 {matched['paired_round_speedup']['median']:.3f}×（block-bootstrap 95% CI {matched['paired_round_speedup']['block_bootstrap_median_ci95_low']:.3f}–{matched['paired_round_speedup']['block_bootstrap_median_ci95_high']:.3f}×）；同有效 batch 为 {capacity['paired_round_speedup']['median']:.3f}×（{capacity['paired_round_speedup']['block_bootstrap_median_ci95_low']:.3f}–{capacity['paired_round_speedup']['block_bootstrap_median_ci95_high']:.3f}×）。"
            ),
        },
        {"id": "throughput", "type": "chart", "chartId": "throughput_chart", "layout": "full"},
        {"id": "main_results", "type": "table", "tableId": "main_table", "layout": "full"},
        {
            "id": "mechanism",
            "type": "markdown",
            "layout": "full",
            "sourceId": "efficiency_summary",
            "body": (
                "## Mechanism Evidence\n\n"
                f"在相同 Direct-LIF + QAT+SQUAT 模型、输入与 batch 下，只把 T 从 1 增到 8，峰值显存增至 {mechanism['squat_t8_to_t1_peak_memory_ratio']:.3f}×，update 时间增至 {mechanism['squat_t8_to_t1_update_time_ratio']:.3f}×。这直接支持：recurrent state/activation 的时间展开与 full BPTT 是基线训练开销的主要来源之一。完整 QAD 相对无教师/KD 的 QIF-STE 多用 {mechanism['qad_teacher_kd_memory_overhead_gib']:.3f} GiB；两者 pooled median update time 比为 {mechanism['qad_to_qif_ste_update_time_ratio']:.3f}×。"
            ),
        },
        {"id": "timestep_memory", "type": "chart", "chartId": "timestep_memory_chart", "layout": "half"},
        {"id": "timestep_time", "type": "chart", "chartId": "timestep_time_chart", "layout": "half"},
        {"id": "ablation_results", "type": "table", "tableId": "ablation_table", "layout": "full"},
        {"id": "phase_results", "type": "table", "tableId": "phase_table", "layout": "full"},
        {
            "id": "accuracy_context_heading",
            "type": "markdown",
            "layout": "full",
            "sourceId": "accuracy_context",
            "body": (
                "## Accuracy Context\n\n"
                f"冻结 seed-1234 记录中，QAD mIoU={accuracy['qad_miou']:.6f}，SQUAT mIoU={accuracy['squat_miou']:.6f}，路线差为 +{accuracy['qad_minus_squat_miou']:.6f}。该结果与效率方向一致，但由于训练预算及完整路线设置不同，只能作为“不以现有精度换取效率”的支持性证据。"
            ),
        },
        {"id": "accuracy_results", "type": "table", "tableId": "accuracy_table", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "layout": "full",
            "sourceId": "protocol",
            "body": (
                "## Scope, Data, and Metric Definitions\n\n"
                "- **QAD 路线：** Integer-LIF/QIF compact neuronal code，T=1，W4A4，冻结 FP32 教师，feature KD weight=0.1；没有跨时间展开的 BPTT。\n"
                "- **SNN 对照：** Direct-LIF，T=8，W4M4S1（权重/膜电位/脉冲量化），recurrent state 按时间展开并执行完整、非截断 BPTT。\n"
                "- **Peak allocated memory：** 每个隔离进程内 `torch.cuda.max_memory_allocated`，跨轮次取最大值；不含 CUDA context 与非 PyTorch 分配。\n"
                "- **Update time / throughput：** CUDA 同步 wall clock；吞吐为 effective batch / update time。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "sourceId": "protocol",
            "body": (
                "## Methodology\n\n"
                "固定读取 UDD 训练集前 64 个真实样本并 resize 到 400×400；每个配置在独立 subprocess 中加载 checkpoint、预热并测量。主比较每条路线 5 个独立轮次，每轮 10 个 warmup updates + 20 个 measured updates，并交替执行顺序。时间步与 QIF-STE 消融为 3 轮、每轮 5 warmups + 10 measurements。随机种子为 1234，AMP 关闭，cuDNN deterministic 开启。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Limitations, Uncertainty, and Robustness\n\n"
                "- 这是固定真实 batch 的训练内核 microbenchmark；排除了数据加载、验证和 checkpoint I/O，epoch 数字是投影，不是完整 epoch 实测。\n"
                "- 主比较同时改变神经元、T、BPTT、量化对象和蒸馏，因此结论属于完整训练路线，不能把全部收益单独归因于 QAD。\n"
                "- Capacity 模式的 physical batch 不同，故不得用其中的 24.006 GiB 与 14.082 GiB 推导显存优劣；可解释的是可运行 batch 容量与同 effective batch 吞吐。\n"
                "- QAD matched 第 4 轮出现较慢 backward，已保留并进入 block-bootstrap CI；没有事后删除。\n"
                "- 精度来自冻结的单种子、不同预算结果；需要等预算多种子实验才能作严格精度归因。\n"
                "- PyTorch fake-quant 训练速度不代表低比特硬件推理延迟或能耗。"
            ),
        },
        {
            "id": "paper_wording",
            "type": "markdown",
            "layout": "full",
            "sourceId": "efficiency_summary",
            "body": (
                "## Paper-ready Claim\n\n"
                f"“On a single RTX 5090 with identical physical and effective batch sizes of 4, Integer-LIF + QAD reduced peak PyTorch-allocated training memory from {matched['squat_peak_allocated_gib']:.2f} to {matched['qad_peak_allocated_gib']:.2f} GiB ({matched['qad_memory_reduction_fraction']:.1%}) and improved training throughput by {matched['throughput_speedup_ratio_of_pooled_medians']:.2f}× over the T=8 Direct-LIF QAT+SQUAT baseline with full BPTT. At an effective batch size of 64, QAD supported an 8× larger physical batch and achieved {capacity['throughput_speedup_ratio_of_pooled_medians']:.2f}× higher throughput.”"
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Next Steps and Further Questions\n\n"
                "1. 运行 3–5 个独立训练种子，并统一 optimizer、epoch/update budget 与数据顺序，报告 mIoU 的均值和置信区间。\n"
                "2. 增加真正的整 epoch 端到端计时，以量化 dataloader、验证和 checkpoint I/O 后的实际节省。\n"
                "3. 增加等 T 或等神经元的桥接消融，以分离 compact code、时间步、BPTT、膜电位量化和蒸馏的边际贡献。"
            ),
        },
    ]
    efficiency_datasets = {
        "efficiency_headlines": efficiency_headlines,
        "main_efficiency": main_rows,
        "timestep_ablation": ablation_rows,
        "phase_breakdown": phase_rows,
    }
    accuracy_datasets = {
        "accuracy_headlines": accuracy_headlines,
        "accuracy_rows": [
            {"route": "Integer-LIF + QAD", "miou": accuracy["qad_miou"], "evidence_role": "冻结单种子支持性证据"},
            {"route": "Direct-LIF + QAT+SQUAT", "miou": accuracy["squat_miou"], "evidence_role": "冻结单种子支持性证据"},
        ],
    }
    datasets = _sqlite_roundtrip(
        "staged_efficiency_report", efficiency_datasets, EFFICIENCY_SQL
    )
    datasets.update(
        _sqlite_roundtrip(
            "staged_accuracy_report", accuracy_datasets, ACCURACY_SQL
        )
    )

    report_manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "Same-GPU training-memory and speed comparison with timestep ablations and explicit causal boundaries.",
        "generatedAt": generated_at,
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    return {
        "surface": "report",
        "manifest": report_manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def build_markdown(validated: dict[str, Any], manifest: dict[str, Any]) -> str:
    matched = validated["matched_batch"]
    capacity = validated["capacity_batch"]
    mechanism = validated["mechanism"]
    accuracy = validated["accuracy_context"]
    gpu = manifest["gpu"]["name"]
    return f"""# {TITLE}

## 结论

在 `{gpu}`、400×400 输入、AMP 关闭的训练内核 benchmark 中，同 physical/effective batch=4 时，Integer-LIF + QAD 的峰值 allocated memory 为 **{matched['qad_peak_allocated_gib']:.3f} GiB**，Direct-LIF + QAT+SQUAT（T=8、full BPTT）为 **{matched['squat_peak_allocated_gib']:.3f} GiB**：QAD 降低 **{matched['qad_memory_reduction_fraction']:.2%}**，吞吐提高 **{matched['throughput_speedup_ratio_of_pooled_medians']:.3f}×**。

effective batch=64 时，QAD 采用 physical batch=32、SQUAT 采用 physical batch=4，吞吐分别为 **{capacity['qad_images_per_second']:.2f}** 与 **{capacity['squat_images_per_second']:.2f} images/s**（**{capacity['throughput_speedup_ratio_of_pooled_medians']:.3f}×**）；这组峰值显存因 physical batch 不同而不作倍率比较。

## 主结果

| 口径 | 路线 | physical/effective B | T | Peak allocated GiB | Median s/update | images/s |
|---|---|---:|---:|---:|---:|---:|
| 同 physical batch | Integer-LIF + QAD | 4/4 | 1 | {matched['qad_peak_allocated_gib']:.3f} | {matched['qad_update_seconds']:.6f} | {matched['qad_images_per_second']:.3f} |
| 同 physical batch | Direct-LIF + QAT+SQUAT | 4/4 | 8 | {matched['squat_peak_allocated_gib']:.3f} | {matched['squat_update_seconds']:.6f} | {matched['squat_images_per_second']:.3f} |
| 同 effective batch | Integer-LIF + QAD | 32/64 | 1 | {capacity['qad_peak_allocated_gib']:.3f}* | {capacity['qad_update_seconds']:.6f} | {capacity['qad_images_per_second']:.3f} |
| 同 effective batch | Direct-LIF + QAT+SQUAT | 4/64 | 8 | {capacity['squat_peak_allocated_gib']:.3f}* | {capacity['squat_update_seconds']:.6f} | {capacity['squat_images_per_second']:.3f} |

*physical batch 不同，不用于显存倍率结论。

## 机制证据

在同一 SQUAT 路线内，T=8 相比 T=1 的峰值显存为 **{mechanism['squat_t8_to_t1_peak_memory_ratio']:.3f}×**，更新时间为 **{mechanism['squat_t8_to_t1_update_time_ratio']:.3f}×**，直接支持时间展开和 full BPTT 是额外开销的重要来源。完整 QAD 相对无教师/KD 的 QIF-STE 多用 **{mechanism['qad_teacher_kd_memory_overhead_gib']:.3f} GiB**，pooled median 更新时间比为 **{mechanism['qad_to_qif_ste_update_time_ratio']:.3f}×**。

## 精度上下文

冻结 seed-1234 记录：QAD mIoU **{accuracy['qad_miou']:.6f}**，SQUAT mIoU **{accuracy['squat_miou']:.6f}**，差 **+{accuracy['qad_minus_squat_miou']:.6f}**。这是不同训练预算与完整路线设置下的支持性证据，不能视为受控因果比较。

## 论文推荐表述

> On a single RTX 5090 with identical physical and effective batch sizes of 4, Integer-LIF + QAD reduced peak PyTorch-allocated training memory from {matched['squat_peak_allocated_gib']:.2f} to {matched['qad_peak_allocated_gib']:.2f} GiB ({matched['qad_memory_reduction_fraction']:.1%}) and improved training throughput by {matched['throughput_speedup_ratio_of_pooled_medians']:.2f}× over the T=8 Direct-LIF QAT+SQUAT baseline with full BPTT. At an effective batch size of 64, QAD supported an 8× larger physical batch and achieved {capacity['throughput_speedup_ratio_of_pooled_medians']:.2f}× higher throughput.

## 方法与边界

- 主比较：5 个独立进程轮次，每轮 10 warmups + 20 measured updates；轮次中位数 block bootstrap 95% CI。
- 时间步消融：3 轮，每轮 5 warmups + 10 measurements。
- 固定 64 个真实 UDD 样本反复测量；数据加载、验证与 checkpoint I/O 排除，epoch 时间仅为投影。
- 主比较改变了神经元、T、BPTT、量化对象和蒸馏，只能解释完整路线差异。
- `torch.cuda.max_memory_allocated` 不含 CUDA context 与非 PyTorch 分配；fake-quant 训练速度不代表低比特硬件推理速度或能耗。
"""


def build_latex(validated: dict[str, Any]) -> str:
    matched = validated["matched_batch"]
    capacity = validated["capacity_batch"]
    return rf"""% Auto-generated from validated_summary.json.
\begin{{table}}[t]
\centering
\caption{{Training efficiency on one RTX 5090. Peak-memory ratios are reported only for matched physical batch size.}}
\label{{tab:training_efficiency}}
\begin{{tabular}}{{llrrrr}}
\toprule
Setting & Method & $B_{{phys}}/B_{{eff}}$ & $T$ & Peak GiB & img/s \\
\midrule
Matched & Integer-LIF + QAD & 4/4 & 1 & {matched['qad_peak_allocated_gib']:.2f} & {matched['qad_images_per_second']:.2f} \\
Matched & Direct-LIF + SQUAT & 4/4 & 8 & {matched['squat_peak_allocated_gib']:.2f} & {matched['squat_images_per_second']:.2f} \\
Capacity & Integer-LIF + QAD & 32/64 & 1 & {capacity['qad_peak_allocated_gib']:.2f}$^\dagger$ & {capacity['qad_images_per_second']:.2f} \\
Capacity & Direct-LIF + SQUAT & 4/64 & 8 & {capacity['squat_peak_allocated_gib']:.2f}$^\dagger$ & {capacity['squat_images_per_second']:.2f} \\
\bottomrule
\end{{tabular}}
\vspace{{2pt}}
\footnotesize Matched-batch QAD: {matched['qad_memory_reduction_fraction']:.1%} lower peak allocated memory and {matched['throughput_speedup_ratio_of_pooled_medians']:.2f}$\times$ throughput. Capacity setting: {capacity['throughput_speedup_ratio_of_pooled_medians']:.2f}$\times$ throughput and 8$\times$ physical-batch capacity. $^\dagger$Not directly comparable because physical batch sizes differ.
\end{{table}}
"""


def build_accuracy_evidence(source: Path) -> dict[str, Any]:
    payload = _read_json(source)
    return {
        "copied_at": datetime.now(timezone.utc).isoformat(),
        "original_source": str(source),
        "original_source_sha256": _sha256(source),
        "execution_policy": "read frozen metrics only; no new accuracy train/eval/convert",
        "qad_miou": payload["historical"]["qad"]["miou"],
        "squat_miou": payload["new_results"]["w4m4s1_squat"]["miou"],
        "qad_minus_squat_miou": -payload["comparison"]["delta_route"],
        "historical_qad_quantization_loss": payload["comparison"]["historical_qad_quantization_loss"],
        "squat_quantization_loss": payload["comparison"]["squat_quantization_loss"],
        "limitations": [
            "single seed",
            "different training budgets",
            "route-level comparison; not a controlled causal ablation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--accuracy-source", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    accuracy_source = args.accuracy_source.resolve()
    summary = _read_json(output_dir / "summary.json")
    manifest = _read_json(output_dir / "manifest.json")
    validated = build_validated_summary(output_dir, summary, accuracy_source)
    _write_json(output_dir / "validated_summary.json", validated)
    _write_json(output_dir / "accuracy_evidence.json", build_accuracy_evidence(accuracy_source))
    artifact = build_artifact(summary, validated, manifest)
    _write_json(output_dir / "artifact.json", artifact)
    (output_dir / "report.md").write_text(build_markdown(validated, manifest), encoding="utf-8")
    (output_dir / "paper_table.tex").write_text(build_latex(validated), encoding="utf-8")
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "validation": validated["validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
