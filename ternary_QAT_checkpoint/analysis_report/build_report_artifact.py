#!/usr/bin/env python3
"""Build the bounded Data Analytics artifact for the ternary QAT experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
TERNARY_ROOT = REPO_ROOT / "ternary_QAT_checkpoint"
COMPARISON_PATH = TERNARY_ROOT / "comparison_seed1234" / "comparison.json"
PIPELINE_PATH = TERNARY_ROOT / "pipeline_manifest.json"

RUNS = {
    "W1.58/A4": TERNARY_ROOT
    / "udd"
    / "SpikingLETNet_shallow_max_w1p58_a4_bs64_seed1234",
    "W1.58/A1.58": TERNARY_ROOT
    / "udd"
    / "SpikingLETNet_shallow_max_w1p58_a1p58_bs64_seed1234",
}

CLASS_LABELS = {
    "background": "背景",
    "facade": "立面",
    "road": "道路",
    "vegetation": "植被",
    "vehicle": "车辆",
    "roof": "屋顶",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_fields = {
        "epoch",
        "train_loss",
        "task_loss",
        "kd_loss",
        "lr",
        "train_batches",
        "train_seconds",
        "val_miou",
        "val_batches",
        "val_seconds",
        "weight_zero_fraction",
        "elapsed_seconds",
    }
    parsed = []
    for row in rows:
        parsed_row = {
            key: (float(value) if key in numeric_fields else value)
            for key, value in row.items()
        }
        parsed_row["epoch"] = int(parsed_row["epoch"])
        parsed_row["train_batches"] = int(parsed_row["train_batches"])
        parsed_row["val_batches"] = int(parsed_row["val_batches"])
        parsed.append(parsed_row)
    return parsed


def assert_complete(comparison: dict, pipeline: dict, metrics_by_run: dict) -> None:
    assert pipeline["status"] == "complete"
    assert [stage["status"] for stage in pipeline["stages"]] == [
        "complete",
        "complete",
        "complete",
    ]
    assert set(comparison["results"]) == {
        "FP32",
        "W4/A4",
        "W1.58/A4",
        "W1.58/A1.58",
    }
    for variant, rows in metrics_by_run.items():
        epochs = [row["epoch"] for row in rows]
        assert epochs == list(range(1, 151)), f"{variant}: incomplete epoch sequence"
        assert all(
            math.isfinite(row[field])
            for row in rows
            for field in ("train_loss", "val_miou", "weight_zero_fraction")
        )


def source_definitions(completed_at: str) -> list[dict]:
    return [
        {
            "id": "unified_comparison",
            "label": "统一精度评估结果",
            "query": {
                "engine": "repository JSON",
                "executed_at": completed_at,
                "description": (
                    "读取统一评估输出；四个模型均在 UDD val_patches.txt 上，"
                    "以 batch size 20、T=1、完整 424 batches 计算六类 mean IoU。"
                ),
                "sql": (
                    "SELECT * FROM read_json_auto("
                    "'ternary_QAT_checkpoint/comparison_seed1234/comparison.json'"
                    ");"
                ),
                "tables_used": [
                    "ternary_QAT_checkpoint/comparison_seed1234/comparison.json"
                ],
                "filters": [
                    "dataset = UDD val_patches.txt",
                    "evaluated_batches = 424",
                    "classes = background, facade, road, vegetation, vehicle, roof",
                ],
                "metric_definitions": [
                    "mIoU = 六个类别 IoU（含 background）的算术平均值。",
                    "W4 优势（pp） = 100 × (W4/A4 mIoU − W1.58/A4 mIoU)。",
                    "保留率 = 100 × 候选模型 mIoU ÷ 参考模型 mIoU。",
                    "理论权重位宽仅比较码字位数，不计比例因子、封装开销或硬件实现。",
                ],
            },
        },
        {
            "id": "ternary_training",
            "label": "三元 QAT 训练曲线与最佳指标",
            "query": {
                "engine": "repository CSV/JSON",
                "executed_at": completed_at,
                "description": (
                    "读取两个 seed=1234 三元 QAT 运行的 150 轮 metrics.csv 与 "
                    "best_metrics.json，并复核轮次连续性和数值有限性。"
                ),
                "sql": (
                    "SELECT 'W1.58/A4' AS variant, * FROM read_csv_auto("
                    "'ternary_QAT_checkpoint/udd/"
                    "SpikingLETNet_shallow_max_w1p58_a4_bs64_seed1234/metrics.csv'"
                    ") UNION ALL SELECT 'W1.58/A1.58' AS variant, * "
                    "FROM read_csv_auto('ternary_QAT_checkpoint/udd/"
                    "SpikingLETNet_shallow_max_w1p58_a1p58_bs64_seed1234/"
                    "metrics.csv');"
                ),
                "tables_used": [
                    (
                        "ternary_QAT_checkpoint/udd/"
                        "SpikingLETNet_shallow_max_w1p58_a4_bs64_seed1234/metrics.csv"
                    ),
                    (
                        "ternary_QAT_checkpoint/udd/"
                        "SpikingLETNet_shallow_max_w1p58_a1p58_bs64_seed1234/metrics.csv"
                    ),
                ],
                "filters": [
                    "seed = 1234",
                    "epochs = 1..150",
                    "batch_size = 64",
                ],
                "metric_definitions": [
                    "末 10 轮均值/标准差基于各运行 metrics.csv 中 epoch 141–150 的 val_miou。",
                    "权重零值率 = 三元量化权重中码字 0 的数量 ÷ 量化权重总数。",
                ],
            },
        },
        {
            "id": "pipeline_manifest",
            "label": "实验流水线完成清单",
            "query": {
                "engine": "repository JSON",
                "executed_at": completed_at,
                "description": (
                    "实验预注册判据、阶段状态、主种子、可选复现实验种子以及完成时间。"
                ),
                "tables_used": ["ternary_QAT_checkpoint/pipeline_manifest.json"],
                "metric_definitions": [
                    "预注册判据：W4/A4 − W1.58/A4 ≥ 2.0 个绝对 mIoU 百分点，"
                    "则支持 4-bit 的精度价值；否则追加两个种子。"
                ],
            },
        },
        {
            "id": "w4_training_log",
            "label": "历史 W4/A4 QAT 训练日志",
            "query": {
                "engine": "repository log",
                "description": (
                    "历史 W4/A4 日志声明 max_epochs=150，但包含 epoch 0–126 "
                    "共 127 个训练轮次；统一比较使用已保存的最佳量化权重。"
                ),
                "tables_used": [
                    (
                        "QAT_checkpoint/udd/"
                        "SpikingLETNet_shallow_maxbs64gpu1_trainval20260612-135249/"
                        "log.txt"
                    )
                ],
                "filters": ["seed = 1234", "quant_bits = 4", "activation_quant = true"],
            },
        },
    ]


def main() -> None:
    comparison = load_json(COMPARISON_PATH)
    pipeline = load_json(PIPELINE_PATH)
    metrics_by_run = {
        variant: load_metrics(run_dir / "metrics.csv")
        for variant, run_dir in RUNS.items()
    }
    assert_complete(comparison, pipeline, metrics_by_run)

    results = comparison["results"]
    decision = comparison["decision"]
    fp_miou = results["FP32"]["miou"]
    w4_miou = results["W4/A4"]["miou"]
    ternary_a4_miou = results["W1.58/A4"]["miou"]
    ternary_a1p58_miou = results["W1.58/A1.58"]["miou"]
    ternary_bits = results["W1.58/A4"]["weight_bits"]
    completed_at = pipeline["completed_at"]

    model_rows = []
    for rank, model in enumerate(("FP32", "W4/A4", "W1.58/A4", "W1.58/A1.58"), 1):
        result = results[model]
        checkpoint = result["checkpoint"]
        model_rows.append(
            {
                "rank": rank,
                "model": model,
                "miou": result["miou"],
                "miou_percent": 100.0 * result["miou"],
                "gap_vs_fp_pp": 100.0 * (fp_miou - result["miou"]),
                "gap_vs_w4_pp": 100.0 * (w4_miou - result["miou"]),
                "retention_vs_fp_pct": 100.0 * result["miou"] / fp_miou,
                "weight_bits": result["weight_bits"],
                "activation_bits": result["activation_bits"],
                "evaluated_batches": result["evaluated_batches"],
                "best_epoch": checkpoint["epoch"],
                "weight_zero_fraction": result.get("weight_zero_fraction"),
            }
        )

    class_rows = []
    classes = comparison["protocol"]["classes"]
    for model, result in results.items():
        for class_name, class_iou in zip(classes, result["per_class_iou"], strict=True):
            class_rows.append(
                {
                    "model": model,
                    "class": CLASS_LABELS[class_name],
                    "class_key": class_name,
                    "iou": class_iou,
                    "iou_percent": 100.0 * class_iou,
                    "overall_miou": result["miou"],
                    "weight_bits": result["weight_bits"],
                    "activation_bits": result["activation_bits"],
                }
            )

    gap_rows = []
    for index, class_name in enumerate(classes):
        w4_iou = results["W4/A4"]["per_class_iou"][index]
        ternary_a4_iou = results["W1.58/A4"]["per_class_iou"][index]
        ternary_a1_iou = results["W1.58/A1.58"]["per_class_iou"][index]
        gap_rows.append(
            {
                "class": CLASS_LABELS[class_name],
                "class_key": class_name,
                "w4_iou": w4_iou,
                "ternary_a4_iou": ternary_a4_iou,
                "ternary_a1p58_iou": ternary_a1_iou,
                "w4_advantage_pp": 100.0 * (w4_iou - ternary_a4_iou),
                "activation_penalty_pp": 100.0 * (ternary_a4_iou - ternary_a1_iou),
            }
        )
    gap_rows.sort(key=lambda row: row["w4_advantage_pp"], reverse=True)

    training_rows = []
    training_summary = []
    for variant, rows in metrics_by_run.items():
        best_row = max(rows, key=lambda row: row["val_miou"])
        for row in rows:
            training_rows.append(
                {
                    "variant": variant,
                    "epoch": row["epoch"],
                    "val_miou": row["val_miou"],
                    "train_loss": row["train_loss"],
                    "task_loss": row["task_loss"],
                    "kd_loss": row["kd_loss"],
                    "lr": row["lr"],
                    "weight_zero_fraction": row["weight_zero_fraction"],
                    "train_batches": row["train_batches"],
                    "val_batches": row["val_batches"],
                }
            )
        last_ten = [row["val_miou"] for row in rows[-10:]]
        training_summary.append(
            {
                "variant": variant,
                "epochs": len(rows),
                "best_epoch": best_row["epoch"],
                "best_val_miou": best_row["val_miou"],
                "final_val_miou": rows[-1]["val_miou"],
                "last10_mean": statistics.fmean(last_ten),
                "last10_std": statistics.pstdev(last_ten),
                "elapsed_hours": rows[-1]["elapsed_seconds"] / 3600.0,
                "zero_fraction_start": rows[0]["weight_zero_fraction"],
                "zero_fraction_best": best_row["weight_zero_fraction"],
                "zero_fraction_final": rows[-1]["weight_zero_fraction"],
            }
        )

    theoretical_bit_reduction_pct = 100.0 * (1.0 - ternary_bits / 4.0)
    headline_rows = [
        {
            "w4_advantage_pp": decision[
                "primary_gap_miou_points_w4_minus_w1p58_a4"
            ],
            "threshold_pp": decision["threshold_miou_points"],
            "margin_over_threshold_pp": (
                decision["primary_gap_miou_points_w4_minus_w1p58_a4"]
                - decision["threshold_miou_points"]
            ),
            "ternary_a4_retention_vs_w4_pct": 100.0 * ternary_a4_miou / w4_miou,
            "ternary_a4_retention_vs_fp_pct": 100.0 * ternary_a4_miou / fp_miou,
            "theoretical_bit_reduction_vs_w4_pct": theoretical_bit_reduction_pct,
            "activation_penalty_pp": decision[
                "secondary_activation_penalty_miou_points"
            ],
            "ternary_a1p58_miou": ternary_a1p58_miou,
            "ternary_a4_zero_fraction": results["W1.58/A4"][
                "weight_zero_fraction"
            ],
        }
    ]

    sources = source_definitions(completed_at)
    title = "SpikingLETNet 三元量化实验技术报告"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "UDD 验证集上的 FP32、W4/A4、W1.58/A4 与 W1.58/A1.58 "
                "统一评估；结论按预注册 2.0 mIoU 百分点门槛解释。"
            ),
            "generatedAt": completed_at,
            "sources": sources,
            "cards": [
                {
                    "id": "card_w4_gap",
                    "dataset": "headline_metrics",
                    "description": "统一评估 mIoU 的绝对百分点差。",
                    "sourceId": "unified_comparison",
                    "metrics": [
                        {
                            "label": "W4 精度优势",
                            "field": "w4_advantage_pp",
                            "format": "number",
                        },
                        {
                            "label": "预设门槛",
                            "field": "threshold_pp",
                            "format": "number",
                        },
                        {
                            "label": "超过门槛",
                            "field": "margin_over_threshold_pp",
                            "format": "number",
                            "signed": True,
                        },
                    ],
                },
                {
                    "id": "card_ternary_retention",
                    "dataset": "headline_metrics",
                    "description": "W1.58/A4 相对 W4/A4 的 mIoU 保留率。",
                    "sourceId": "unified_comparison",
                    "metrics": [
                        {
                            "label": "相对 W4 保留率",
                            "field": "ternary_a4_retention_vs_w4_pct",
                            "format": "number",
                        },
                        {
                            "label": "理论权重位数减少",
                            "field": "theoretical_bit_reduction_vs_w4_pct",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "card_activation_penalty",
                    "dataset": "headline_metrics",
                    "description": "同为三元权重时，A4 改为 A1.58 的 mIoU 损失。",
                    "sourceId": "unified_comparison",
                    "metrics": [
                        {
                            "label": "激活量化损失",
                            "field": "activation_penalty_pp",
                            "format": "number",
                        },
                        {
                            "label": "W1.58/A1.58 mIoU",
                            "field": "ternary_a1p58_miou",
                            "format": "percent",
                        },
                    ],
                },
            ],
            "charts": [
                {
                    "id": "chart_model_miou",
                    "title": "四种精度方案的统一验证集 mIoU",
                    "subtitle": (
                        "W4/A4 比 W1.58/A4 高 2.85 个百分点；"
                        "W1.58/A1.58 出现显著精度塌缩。"
                    ),
                    "showDescription": True,
                    "dataset": "model_comparison",
                    "sourceId": "unified_comparison",
                    "type": "bar",
                    "encodings": {
                        "x": {
                            "field": "model",
                            "type": "nominal",
                            "label": "精度方案",
                        },
                        "y": {
                            "field": "miou",
                            "type": "quantitative",
                            "label": "mIoU",
                            "format": "percent",
                        },
                        "tooltip": [
                            {
                                "field": "weight_bits",
                                "label": "权重位宽",
                                "format": "number",
                            },
                            {
                                "field": "activation_bits",
                                "label": "激活位宽",
                                "format": "number",
                            },
                            {
                                "field": "retention_vs_fp_pct",
                                "label": "相对 FP32 保留率(%)",
                                "format": "number",
                            },
                            {
                                "field": "evaluated_batches",
                                "label": "评估 batches",
                                "format": "number",
                            },
                        ],
                    },
                },
                {
                    "id": "chart_training_curves",
                    "title": "三元 QAT 验证 mIoU 训练曲线",
                    "subtitle": (
                        "两个运行均完成 150 轮；A4 最佳点在第 150 轮，"
                        "A1.58 最佳点在第 97 轮。"
                    ),
                    "showDescription": True,
                    "dataset": "training_curves",
                    "sourceId": "ternary_training",
                    "type": "line",
                    "encodings": {
                        "x": {
                            "field": "epoch",
                            "type": "quantitative",
                            "label": "训练轮次",
                        },
                        "y": {
                            "field": "val_miou",
                            "type": "quantitative",
                            "label": "验证 mIoU",
                            "format": "percent",
                        },
                        "color": {
                            "field": "variant",
                            "type": "nominal",
                            "label": "方案",
                        },
                        "tooltip": [
                            {
                                "field": "train_loss",
                                "label": "训练损失",
                                "format": "number",
                            },
                            {
                                "field": "weight_zero_fraction",
                                "label": "权重零值率",
                                "format": "percent",
                            },
                            {
                                "field": "lr",
                                "label": "学习率",
                                "format": "number",
                            },
                        ],
                    },
                },
                {
                    "id": "chart_class_gap",
                    "title": "W4/A4 相对 W1.58/A4 的逐类别 IoU 优势",
                    "subtitle": "差距主要来自道路、背景和车辆；各类别差值均为正。",
                    "showDescription": True,
                    "dataset": "class_gaps",
                    "sourceId": "unified_comparison",
                    "type": "bar",
                    "options": {"orientation": "horizontal", "grouping": "single"},
                    "encodings": {
                        "x": {
                            "field": "class",
                            "type": "nominal",
                            "label": "类别",
                        },
                        "y": {
                            "field": "w4_advantage_pp",
                            "type": "quantitative",
                            "label": "W4 优势（mIoU 百分点）",
                            "format": "number",
                        },
                        "tooltip": [
                            {
                                "field": "w4_iou",
                                "label": "W4/A4 IoU",
                                "format": "percent",
                            },
                            {
                                "field": "ternary_a4_iou",
                                "label": "W1.58/A4 IoU",
                                "format": "percent",
                            },
                            {
                                "field": "activation_penalty_pp",
                                "label": "A1.58 激活损失(pp)",
                                "format": "number",
                            },
                        ],
                    },
                },
            ],
            "tables": [
                {
                    "id": "table_model_comparison",
                    "title": "统一评估明细",
                    "subtitle": "所有模型使用相同 UDD 验证协议；按 mIoU 降序。",
                    "showDescription": True,
                    "dataset": "model_comparison",
                    "sourceId": "unified_comparison",
                    "columns": [
                        {"field": "model", "label": "方案", "type": "text"},
                        {
                            "field": "miou",
                            "label": "mIoU",
                            "format": "percent",
                        },
                        {
                            "field": "gap_vs_fp_pp",
                            "label": "距 FP32 (pp)",
                            "format": "number",
                        },
                        {
                            "field": "retention_vs_fp_pct",
                            "label": "FP32 保留率 (%)",
                            "format": "number",
                        },
                        {
                            "field": "weight_bits",
                            "label": "权重位宽",
                            "format": "number",
                        },
                        {
                            "field": "activation_bits",
                            "label": "激活位宽",
                            "format": "number",
                        },
                        {
                            "field": "best_epoch",
                            "label": "最佳轮次",
                            "format": "number",
                        },
                    ],
                    "defaultSort": {"field": "miou", "direction": "desc"},
                },
                {
                    "id": "table_training_summary",
                    "title": "三元 QAT 训练质量摘要",
                    "subtitle": "末 10 轮统计反映最终阶段的验证波动。",
                    "showDescription": True,
                    "dataset": "training_summary",
                    "sourceId": "ternary_training",
                    "columns": [
                        {"field": "variant", "label": "方案", "type": "text"},
                        {
                            "field": "best_epoch",
                            "label": "最佳轮次",
                            "format": "number",
                        },
                        {
                            "field": "best_val_miou",
                            "label": "最佳 mIoU",
                            "format": "percent",
                        },
                        {
                            "field": "final_val_miou",
                            "label": "末轮 mIoU",
                            "format": "percent",
                        },
                        {
                            "field": "last10_mean",
                            "label": "末10轮均值",
                            "format": "percent",
                        },
                        {
                            "field": "last10_std",
                            "label": "末10轮标准差",
                            "format": "percent",
                        },
                        {
                            "field": "elapsed_hours",
                            "label": "耗时(h)",
                            "format": "number",
                        },
                        {
                            "field": "zero_fraction_final",
                            "label": "末轮零值率",
                            "format": "percent",
                        },
                    ],
                    "defaultSort": {
                        "field": "best_val_miou",
                        "direction": "desc",
                    },
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "unified_comparison",
                    "body": (
                        "## 技术摘要\n\n"
                        "**实验已完整结束。** 在相同 UDD 验证协议下，W4/A4 的 "
                        "mIoU 为 **61.95%**，W1.58/A4 为 **59.11%**，前者领先 "
                        "**2.85 个绝对百分点**，超过预注册的 2.0 pp 判据。因此，"
                        "在当前模型、量化器与训练配方下，结果支持 4-bit 方案具有"
                        "实际精度价值。\n\n"
                        "这不是对三元权重的全面否定：W1.58/A4 仍保留 W4/A4 的 "
                        "**95.41%** mIoU，同时理论权重码位数减少 **60.38%**。"
                        "若系统可接受约 2.85 pp 的精度损失，它仍是值得做硬件验证的"
                        "压缩候选。相反，W1.58/A1.58 仅有 **22.33%** mIoU，"
                        "当前三元激活方案不可用。"
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "card_w4_gap",
                        "card_ternary_retention",
                        "card_activation_penalty",
                    ],
                },
                {
                    "id": "findings_heading",
                    "type": "markdown",
                    "body": (
                        "## 关键发现与视觉证据\n\n"
                        "下图和明细表都来自同一次统一评估，避免使用不同验证批次"
                        "或不同 mIoU 口径造成横向偏差。"
                    ),
                },
                {
                    "id": "model_chart_block",
                    "type": "chart",
                    "chartId": "chart_model_miou",
                },
                {
                    "id": "model_table_block",
                    "type": "table",
                    "tableId": "table_model_comparison",
                },
                {
                    "id": "class_analysis",
                    "type": "markdown",
                    "sourceId": "unified_comparison",
                    "body": (
                        "## 逐类别差异\n\n"
                        "W4/A4 在六个类别上都高于 W1.58/A4。差距最大的是道路 "
                        "**5.64 pp**，其次为背景 **3.56 pp** 和车辆 **3.34 pp**；"
                        "立面、植被、屋顶的差距均约为 1.46–1.58 pp。说明总体差距"
                        "不是由单一类别完全驱动，但道路类对结论贡献最大。"
                    ),
                },
                {
                    "id": "class_chart_block",
                    "type": "chart",
                    "chartId": "chart_class_gap",
                },
                {
                    "id": "training_quality",
                    "type": "markdown",
                    "sourceId": "ternary_training",
                    "body": (
                        "## 训练质量与稳定性\n\n"
                        "两个新增运行均具有连续的 1–150 轮记录，且每轮包含 401 个"
                        "训练 batch 和 424 个验证 batch。W1.58/A4 的最佳点出现在"
                        "第 150 轮；末 10 轮验证 mIoU 均值为 **53.68%**、总体标准差"
                        "为 **2.61 pp**，说明后期仍有较明显波动。W1.58/A1.58 的"
                        "最佳点在第 97 轮，末轮退至 17.71%，未表现出可用的稳定收敛。"
                    ),
                },
                {
                    "id": "training_chart_block",
                    "type": "chart",
                    "chartId": "chart_training_curves",
                },
                {
                    "id": "training_table_block",
                    "type": "table",
                    "tableId": "table_training_summary",
                },
                {
                    "id": "scope_and_metrics",
                    "type": "markdown",
                    "sourceId": "pipeline_manifest",
                    "body": (
                        "## 范围、数据与指标定义\n\n"
                        "模型为 `SpikingLETNet_shallow_max`，时间步 `T=1`。两个新增"
                        "三元模型从同一个 FP32 最佳检查点启动，主种子为 1234，训练"
                        "150 轮、batch size 64。最终比较在 UDD `val_patches.txt` "
                        "完整 8,478 个 patch（424 batches，batch size 20）上执行。\n\n"
                        "主指标 mIoU 是背景、立面、道路、植被、车辆和屋顶六类 IoU "
                        "的算术平均。所有“pp”均指 mIoU 的绝对百分点差，而不是相对"
                        "百分比。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "pipeline_manifest",
                    "body": (
                        "## 方法与实验设计\n\n"
                        "新增量化框架独立于既有 W4 框架。三元权重严格使用 "
                        "`{-α, 0, +α}`，按输出通道绝对均值缩放并通过 STE 训练；"
                        "覆盖 Conv2d、ConvTranspose2d 和 Linear。W1.58/A4 使用"
                        "独立框架内的 4-bit 激活，W1.58/A1.58 使用每张量动态缩放的"
                        "有符号三元激活。偏置、BN 与神经元状态保留 FP32。\n\n"
                        "优化器为 Adam，初始学习率 1e-3，poly 调度，KD 权重 0.1。"
                        "预注册主判据为 W4/A4 − W1.58/A4 ≥ 2.0 pp；只有未达到该"
                        "门槛时才追加种子 4321 和 2026。本次差距达到 2.85 pp，故"
                        "流水线按设计未触发额外种子。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "w4_training_log",
                    "body": (
                        "## 局限、不确定性与稳健性\n\n"
                        "第一，核心结论只有单种子证据；且 cuDNN 未强制确定性，因此"
                        "该结果应表述为“支持”而不是“证明”。第二，历史 W4/A4 日志"
                        "虽然配置 `max_epochs=150`，实际只记录 epoch 0–126 共 127 "
                        "轮；统一评估使用其已保存的最佳权重（第 119 个训练轮次）。"
                        "因此比较的是各自保存的最佳模型，而非严格相同的完整训练暴露。"
                        "\n\n"
                        "第三，W4 权重采用既有框架的每张量 max 缩放，三元权重采用"
                        "每输出通道 absmean 缩放；这是两个完整量化配方的系统级比较，"
                        "不能把全部差距只归因于位宽。第四，本实验没有测量部署后的"
                        "延迟、显存、能耗或真实模型包大小；60.38% 仅是理论权重码位数"
                        "减少，不包含缩放因子和硬件封装开销。"
                    ),
                },
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "body": (
                        "## 建议的下一步\n\n"
                        "1. 若目标是当前精度优先，继续采用 W4/A4；它跨过预注册门槛，"
                        "且六类 IoU 全面领先。\n"
                        "2. 若目标是极限压缩，保留 W1.58/A4 作为候选，先在目标硬件上"
                        "测量实际延迟、能耗、峰值内存和模型包大小，再决定 2.85 pp "
                        "是否值得交换。\n"
                        "3. 为形成可发表或可推广的统计结论，仍建议补跑种子 4321、"
                        "2026，报告均值、标准差和置信区间；这属于增强证据，不是本次"
                        "预注册判据要求的补跑。\n"
                        "4. 暂停当前 W1.58/A1.58 激活方案。若继续探索，应先记录各层"
                        "激活分布，并测试非负三元/阈值学习、首末层保留 A4，或逐层"
                        "混合精度，再进行完整训练。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "- 目标硬件是否对稀疏三元权重（W1.58/A4 最佳权重零值率 "
                        "86.20%）有原生加速支持？\n"
                        "- 业务或部署约束能否接受 2.85 pp 的 mIoU 损失，换取理论"
                        "权重码位数减少 60.38%？\n"
                        "- W4 若补足至 150 轮，最佳模型是否会变化？\n"
                        "- 道路和车辆类别是否是下游任务的关键类别，从而需要比全局"
                        "mIoU 更高的决策权重？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": completed_at,
            "datasets": {
                "headline_metrics": headline_rows,
                "model_comparison": model_rows,
                "class_comparison": class_rows,
                "class_gaps": gap_rows,
                "training_curves": training_rows,
                "training_summary": training_summary,
            },
        },
        "sources": sources,
        "package_info": {
            "audience": "technical",
            "language": "zh-CN",
            "artifact_kind": "experiment-analysis",
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "artifact.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(output_path)


if __name__ == "__main__":
    main()
