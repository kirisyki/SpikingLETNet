#!/usr/bin/env python3
"""Build the canonical technical report artifact from reviewed probe outputs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
PRIMARY_DIR = REPO_ROOT / "qad_distortion_repair_results/qad_vs_ste16_full_20260806"
SELF_DIR = REPO_ROOT / "qad_distortion_repair_results/self_sensitivity_full_20260806"
ORIGINAL_DIR = REPO_ROOT / "quantization_distortion_results/full_20260805/statistical_analysis"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:.2f} pp"


def source_object(generated_at: str) -> dict:
    return {
        "id": "repair-analysis",
        "label": "QAD distortion-repair derived analysis",
        "path": "qad_distortion_repair_results/paper_readiness_report_20260806/report_snapshot.json",
        "query": {
            "description": "Reviewed synthesis of the common-teacher audit, same-master-weight sensitivity audit, and original frozen-checkpoint distortion probe.",
            "engine": "DuckDB",
            "language": "sql",
            "sql": "SELECT * FROM read_json_auto('qad_distortion_repair_results/paper_readiness_report_20260806/report_snapshot.json', format = 'unstructured');",
            "query": "Run build_report.py after the two probe entrypoints to regenerate the bounded report snapshot and artifact.",
            "tables_used": [
                "qad_distortion_repair_results/qad_vs_ste16_full_20260806/analysis_summary.json",
                "qad_distortion_repair_results/qad_vs_ste16_full_20260806/source_statistics.csv",
                "qad_distortion_repair_results/self_sensitivity_full_20260806/analysis_summary.json",
                "quantization_distortion_results/full_20260805/statistical_analysis/analysis_summary.json",
            ],
            "filters": [
                "UDD6 validation only",
                "8,478 patches grouped into 35 original source images",
                "78 active QIF nodes and six forward_qat distillation features",
                "T=1 runtime input step; QIF code range 0 through 8",
                "Historical QAD checkpoint versus existing 16-epoch STE-QAT checkpoint",
            ],
            "metric_definitions": [
                "QIF disagreement is the fraction of Integer-LIF output codes unequal to the stated reference, macro-averaged equally over active layers within each source and then equally over sources.",
                "Feature normalized MSE is sum((student-teacher)^2) divided by sum(teacher^2), computed per feature and source before equal-weight averaging.",
                "Feature repair is STE normalized MSE minus QAD normalized MSE; positive values favor QAD.",
                "All 95% intervals resample the 35 source-image clusters with replacement 10,000 times.",
                "The common-teacher comparison is representation alignment, while the FP-shadow comparison is local quantization sensitivity; neither removes the 127-versus-16 epoch training-budget confound.",
            ],
            "executed_at": generated_at,
        },
    }


def build() -> dict:
    primary = read_json(PRIMARY_DIR / "analysis_summary.json")
    self_result = read_json(SELF_DIR / "analysis_summary.json")
    original = read_json(ORIGINAL_DIR / "analysis_summary.json")
    source_rows = read_csv(PRIMARY_DIR / "source_statistics.csv")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    qif = primary["representation"]["disagreement"]
    qif_micro = primary["representation"]["micro_disagreement"]
    feature = primary["distillation_features"]["normalized_mse"]
    feature_ex6 = primary["distillation_features_excluding_feature6"]["normalized_mse"]
    cosine = primary["distillation_features"]["cosine_distance"]
    task = primary["task"]
    self_qif = self_result["representation"]["disagreement"]
    self_feature = self_result["features"]["normalized_mse"]
    flip_reduction = 1.0 - (
        task["qad"]["prediction_flip_rate_vs_fp"]
        / task["ste"]["prediction_flip_rate_vs_fp"]
    )
    task_gap = task["qad"]["miou"] - task["ste"]["miou"]
    fp_qad_gap = task["fp"]["miou"] - task["qad"]["miou"]
    fp_ste_gap = task["fp"]["miou"] - task["ste"]["miou"]
    gap_closed = 1.0 - fp_qad_gap / fp_ste_gap

    headline = [
        {
            "qif_delta": qif["qad_minus_ste"],
            "feature_nmse_reduction": feature["relative_reduction_fraction"],
            "feature_nmse_reduction_ex6": feature_ex6["relative_reduction_fraction"],
            "prediction_flip_reduction": flip_reduction,
            "feature6_share": primary["feature6_share_of_raw_mse_reduction"],
            "miou_gap": task_gap,
            "self_qif_delta": self_qif["qad_minus_ste"],
        }
    ]
    qif_reference = [
        {
            "reference": "Common FP-QIF teacher",
            "reference_order": 0,
            "method": "QAD",
            "disagreement": qif["qad_mean"],
        },
        {
            "reference": "Common FP-QIF teacher",
            "reference_order": 0,
            "method": "STE-QAT",
            "disagreement": qif["ste_mean"],
        },
        {
            "reference": "Own FP-operator shadow",
            "reference_order": 1,
            "method": "QAD",
            "disagreement": self_qif["qad_mean"],
        },
        {
            "reference": "Own FP-operator shadow",
            "reference_order": 1,
            "method": "STE-QAT",
            "disagreement": self_qif["ste_mean"],
        },
    ]
    feature_chart = []
    feature_table = []
    for row in primary["feature_detail"]:
        index = row["feature_index"]
        nmse = row["normalized_mse"]
        cos = row["cosine_distance"]
        for method, key in (("QAD", "qad_mean"), ("STE-QAT", "ste_mean")):
            feature_chart.append(
                {
                    "feature": f"F{index}",
                    "feature_order": index,
                    "method": method,
                    "normalized_mse": nmse[key],
                    "paired_relative_reduction": nmse["relative_reduction_fraction"],
                }
            )
        feature_table.append(
            {
                "feature": f"F{index}",
                "network_position": {
                    1: "Encoder block 1",
                    2: "Encoder block 2",
                    3: "Encoder block 3",
                    4: "Decoder block 4",
                    5: "Decoder block 5",
                    6: "Decoder block 6 / pre-logit",
                }[index],
                "qad_nmse": nmse["qad_mean"],
                "ste_nmse": nmse["ste_mean"],
                "relative_reduction": nmse["relative_reduction_fraction"],
                "reduction_ci_low": nmse["relative_reduction_cluster_bootstrap_95ci"][0],
                "reduction_ci_high": nmse["relative_reduction_cluster_bootstrap_95ci"][1],
                "qad_minus_ste_cosine_distance": cos["qad_minus_ste"],
            }
        )
    stage_order = {
        "stem": 0,
        "encoder_1": 1,
        "encoder_2": 2,
        "encoder_3": 3,
        "decoder_4": 4,
        "decoder_5": 5,
        "decoder_6": 6,
    }
    stage_labels = {
        "stem": "Stem",
        "encoder_1": "Encoder 1",
        "encoder_2": "Encoder 2",
        "encoder_3": "Encoder 3",
        "decoder_4": "Decoder 4",
        "decoder_5": "Decoder 5",
        "decoder_6": "Decoder 6",
    }
    stage_delta = [
        {
            "stage": stage_labels[row["stage"]],
            "stage_order": stage_order[row["stage"]],
            "layer_count": row["layer_count"],
            "qad_disagreement": row["disagreement"]["qad_mean"],
            "ste_disagreement": row["disagreement"]["ste_mean"],
            "qad_minus_ste": row["disagreement"]["qad_minus_ste"],
            "ci_low": row["disagreement"]["qad_minus_ste_cluster_bootstrap_95ci"][0],
            "ci_high": row["disagreement"]["qad_minus_ste_cluster_bootstrap_95ci"][1],
        }
        for row in primary["stage_detail"]
    ]
    stage_delta.sort(key=lambda row: row["stage_order"])
    source_scatter = [
        {
            "source_id": row["source_id"],
            "feature_repair": float(row["feature_repair"]),
            "qif_repair": float(row["qif_repair"]),
            "qad_minus_ste_miou": float(row["qad_minus_ste_miou"]),
            "qad_miou": float(row["qad_miou"]),
            "ste_miou": float(row["ste_miou"]),
        }
        for row in source_rows
    ]
    class_rows = []
    for class_name in ("background", "facade", "road", "vegetation", "vehicle", "roof"):
        class_rows.append(
            {
                "class": class_name,
                "fp_iou": task["fp"]["per_class_iou"][class_name],
                "qad_iou": task["qad"]["per_class_iou"][class_name],
                "ste_iou": task["ste"]["per_class_iou"][class_name],
                "qad_minus_ste": task["qad"]["per_class_iou"][class_name]
                - task["ste"]["per_class_iou"][class_name],
            }
        )
    claims = [
        {
            "claim": "QAD reduces global Integer-LIF code distortion",
            "assessment": "Contradicted",
            "evidence": f"Teacher-referenced D is {pct(qif['qad_mean'])} for QAD vs {pct(qif['ste_mean'])} for STE; QAD is higher on 35/35 sources.",
        },
        {
            "claim": "QAD improves the six features directly targeted by KD",
            "assessment": "Supported descriptively",
            "evidence": f"Mean normalized MSE is lower by {pct(feature['relative_reduction_fraction'])}; excluding F6 it remains lower by {pct(feature_ex6['relative_reduction_fraction'])}.",
        },
        {
            "claim": "Feature repair explains the source-level mIoU gain",
            "assessment": "Not supported",
            "evidence": "Spearman rho = -0.184 with a 95% bootstrap interval spanning zero.",
        },
        {
            "claim": "QAD improves mIoU relative to STE-QAT",
            "assessment": "Observed, not causal",
            "evidence": f"QAD is higher by {pp(task_gap)}, but QAD saw 127 epochs and STE saw 16.",
        },
        {
            "claim": "QAD specifically solves the previously diagnosed W4A4 distortion",
            "assessment": "Insufficient evidence",
            "evidence": "The direction differs by metric: feature alignment improves while code fidelity and local quantization sensitivity worsen.",
        },
    ]

    snapshot = {
        "headline": headline,
        "qif_reference": qif_reference,
        "feature_chart": feature_chart,
        "feature_table": feature_table,
        "stage_delta": stage_delta,
        "source_scatter": source_scatter,
        "class_rows": class_rows,
        "claims": claims,
    }
    write_payload = {
        "generated_at": generated_at,
        "headline": headline[0],
        "common_teacher_summary": primary,
        "self_sensitivity_summary": self_result,
        "original_frozen_probe_headline": original["findings"],
        "report_datasets": snapshot,
    }
    (OUTPUT_DIR / "report_snapshot.json").write_text(
        json.dumps(write_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source = source_object(generated_at)
    cards = [
        {
            "id": "card-qif",
            "description": "Positive means QAD has more QIF code mismatches than STE against the common FP-QIF teacher.",
            "dataset": "headline",
            "sourceId": "repair-analysis",
            "metrics": [
                {
                    "label": "QAD − STE QIF disagreement",
                    "field": "qif_delta",
                    "format": "percent",
                    "signed": True,
                }
            ],
        },
        {
            "id": "card-feature",
            "description": "Equal-weight mean normalized MSE across the six forward_qat KD targets.",
            "dataset": "headline",
            "sourceId": "repair-analysis",
            "metrics": [
                {
                    "label": "KD-feature NMSE reduction",
                    "field": "feature_nmse_reduction",
                    "format": "percent",
                },
                {
                    "label": "Excluding F6",
                    "field": "feature_nmse_reduction_ex6",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "card-flip",
            "description": "Relative reduction in segmentation-prediction flips against the common FP-QIF teacher.",
            "dataset": "headline",
            "sourceId": "repair-analysis",
            "metrics": [
                {
                    "label": "Prediction-flip reduction",
                    "field": "prediction_flip_reduction",
                    "format": "percent",
                }
            ],
        },
        {
            "id": "card-concentration",
            "description": "Share of the raw six-term KD-MSE reduction contributed by the final pre-logit feature.",
            "dataset": "headline",
            "sourceId": "repair-analysis",
            "metrics": [
                {
                    "label": "Raw KD improvement from F6",
                    "field": "feature6_share",
                    "format": "percent",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "chart-qif-reference",
            "title": "QIF code disagreement under two reference definitions",
            "subtitle": "Source-balanced macro disagreement across 78 active QIF nodes and 35 UDD source images.",
            "intent": "comparison",
            "question": "Does QAD reduce discrete Integer-LIF code mismatch?",
            "rationale": "Grouped bars separate common-teacher alignment from same-master-weight quantization sensitivity.",
            "comparisonContext": {"grain": "reference × method", "unit": "rate"},
            "type": "bar",
            "dataset": "qif_reference",
            "sourceId": "repair-analysis",
            "encodings": {
                "x": {"field": "reference", "type": "ordinal", "label": "Reference"},
                "y": {"field": "disagreement", "type": "quantitative", "format": "percent", "label": "QIF code disagreement"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
            },
            "legend": {"position": "bottom", "title": "Method"},
            "settings": {"groupMode": "grouped", "sort": "none"},
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "chart-feature",
            "title": "Teacher-referenced normalized MSE at six KD features",
            "subtitle": "Lower is better; F6 is the final decoder/pre-logit feature.",
            "intent": "comparison",
            "question": "Where does QAD improve the representations directly constrained by distillation?",
            "rationale": "Feature-level grouped bars reveal broad effects and concentration at F6.",
            "comparisonContext": {"baseline": "FP-QIF teacher", "grain": "feature × method", "unit": "normalized MSE"},
            "type": "bar",
            "dataset": "feature_chart",
            "sourceId": "repair-analysis",
            "encodings": {
                "x": {"field": "feature", "type": "ordinal", "label": "Distillation feature"},
                "y": {"field": "normalized_mse", "type": "quantitative", "format": "number", "label": "Normalized MSE"},
                "color": {"field": "method", "type": "nominal", "label": "Method"},
                "tooltip": [
                    {"field": "paired_relative_reduction", "type": "quantitative", "format": "percent", "label": "QAD relative reduction"}
                ],
            },
            "legend": {"position": "bottom", "title": "Method"},
            "settings": {"groupMode": "grouped", "sort": "none"},
            "layout": "full",
        },
        {
            "id": "chart-stage-delta",
            "title": "QAD minus STE QIF disagreement by network stage",
            "subtitle": "Positive values mean QAD is farther from the common FP-QIF teacher; all intervals are source-cluster bootstrap 95% CIs.",
            "intent": "comparison",
            "question": "Is QAD's higher code mismatch confined to one stage?",
            "rationale": "A signed stage bar shows direction and magnitude without hiding the all-positive pattern.",
            "comparisonContext": {"baseline": "STE-QAT", "grain": "network stage", "unit": "disagreement-rate difference"},
            "type": "bar",
            "dataset": "stage_delta",
            "sourceId": "repair-analysis",
            "encodings": {
                "x": {"field": "stage", "type": "ordinal", "label": "Network stage"},
                "y": {"field": "qad_minus_ste", "type": "quantitative", "format": "percent", "label": "QAD − STE disagreement"},
                "tooltip": [
                    {"field": "ci_low", "type": "quantitative", "format": "percent", "label": "95% CI low"},
                    {"field": "ci_high", "type": "quantitative", "format": "percent", "label": "95% CI high"},
                    {"field": "layer_count", "type": "quantitative", "label": "Layers"},
                ],
            },
            "settings": {"sort": "none"},
            "valueFormat": "percent",
            "layout": "full",
        },
        {
            "id": "chart-source-closure",
            "title": "Source-level feature repair and mIoU gain",
            "subtitle": "35 source-image clusters; Spearman rho = -0.184 and the 95% bootstrap interval spans zero.",
            "intent": "relationship",
            "question": "Do sources with stronger feature repair receive larger task gains?",
            "rationale": "One point per independent source cluster tests the proposed representation-task closure at the correct grain.",
            "comparisonContext": {"grain": "source image", "unit": "normalized-MSE repair versus mIoU"},
            "type": "scatter",
            "dataset": "source_scatter",
            "sourceId": "repair-analysis",
            "encodings": {
                "x": {"field": "feature_repair", "type": "quantitative", "format": "number", "label": "Feature repair (STE NMSE − QAD NMSE)"},
                "y": {"field": "qad_minus_ste_miou", "type": "quantitative", "format": "percent", "label": "QAD − STE source mIoU"},
                "label": {"field": "source_id", "type": "text", "label": "Source"},
                "tooltip": [
                    {"field": "qif_repair", "type": "quantitative", "format": "percent", "label": "QIF repair"},
                    {"field": "qad_miou", "type": "quantitative", "format": "percent", "label": "QAD mIoU"},
                    {"field": "ste_miou", "type": "quantitative", "format": "percent", "label": "STE mIoU"},
                ],
            },
            "layout": "full",
        },
    ]
    tables = [
        {
            "id": "table-features",
            "title": "Distillation-feature alignment audit",
            "subtitle": "Common FP-QIF teacher reference; positive relative reduction favors QAD.",
            "dataset": "feature_table",
            "defaultSort": {"field": "feature", "direction": "asc"},
            "density": "dense",
            "sourceId": "repair-analysis",
            "layout": "full",
            "columns": [
                {"field": "feature", "label": "Feature", "type": "text"},
                {"field": "network_position", "label": "Position", "type": "text"},
                {"field": "qad_nmse", "label": "QAD NMSE", "format": "number"},
                {"field": "ste_nmse", "label": "STE NMSE", "format": "number"},
                {"field": "relative_reduction", "label": "Relative reduction", "format": "percent", "movement": True},
                {"field": "qad_minus_ste_cosine_distance", "label": "Cosine-distance delta", "format": "number", "movement": True},
            ],
        },
        {
            "id": "table-claims",
            "title": "Paper-claim evidence audit",
            "subtitle": "Assessment incorporates the 127-versus-16 epoch budget mismatch.",
            "dataset": "claims",
            "defaultSort": {"field": "claim", "direction": "asc"},
            "density": "spacious",
            "sourceId": "repair-analysis",
            "layout": "full",
            "columns": [
                {"field": "claim", "label": "Candidate claim", "type": "text"},
                {"field": "assessment", "label": "Assessment", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
            ],
        },
        {
            "id": "table-classes",
            "title": "Per-class IoU under the common validation protocol",
            "subtitle": "The QAD-versus-STE differences are descriptive because training budgets differ.",
            "dataset": "class_rows",
            "defaultSort": {"field": "qad_minus_ste", "direction": "desc"},
            "density": "dense",
            "sourceId": "repair-analysis",
            "layout": "full",
            "columns": [
                {"field": "class", "label": "Class", "type": "text"},
                {"field": "fp_iou", "label": "FP-QIF", "format": "percent"},
                {"field": "qad_iou", "label": "QAD", "format": "percent"},
                {"field": "ste_iou", "label": "STE-QAT", "format": "percent"},
                {"field": "qad_minus_ste", "label": "QAD − STE", "format": "percent", "movement": True},
            ],
        },
    ]
    blocks = [
        {"id": "title", "type": "markdown", "body": "# QAD 对 W4A4 Integer-LIF 失真的修复程度与论文证据边界"},
        {
            "id": "technical-summary",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## 技术结论：QAD 改善了蒸馏特征与任务输出，但没有修复全网离散码失真\n\n"
                f"**当前证据只部分支持 QAD 的作用解释，不足以支撑“QAD 解决了 Integer-LIF quantization distortion”的强表述。** "
                f"相对现有 STE-QAT，QAD 将六个 KD 特征的 teacher-referenced normalized MSE 降低 **{pct(feature['relative_reduction_fraction'])}**，"
                f"prediction-flip rate 相对降低 **{pct(flip_reduction)}**，验证 mIoU 高 **{pp(task_gap)}**。但其 78 个 QIF 节点的宏平均码不一致率反而高 **{pp(qif['qad_minus_ste'])}**，"
                "且 35/35 个 source 都是同一方向。\n\n"
                "结果支持的窄机制是：**feature KD 将学生拉向 teacher 的多尺度语义特征，尤其是最终 decoder/pre-logit 特征；它不要求、也没有实现逐层逐码复原。** "
                "由于 QAD 训练了 127 epochs、STE 仅训练 16 epochs，而且 feature repair 与 source-level mIoU gain 无显著相关，当前仍不能把特征对齐或精度差异因果归因于蒸馏。"
            ),
        },
        {"id": "metrics", "type": "metric-strip", "cardIds": ["card-qif", "card-feature", "card-flip", "card-concentration"]},
        {
            "id": "code-result",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## 两种 reference 下，QAD 的全网 QIF 码保真度都更差\n\n"
                f"对共同 FP-QIF teacher，QAD 的宏平均 disagreement 为 **{pct(qif['qad_mean'])}**，STE 为 **{pct(qif['ste_mean'])}**；"
                f"QAD 高 **{pp(qif['qad_minus_ste'])}**，95% CI 为 [{pp(qif['qad_minus_ste_cluster_bootstrap_95ci'][0])}, {pp(qif['qad_minus_ste_cluster_bootstrap_95ci'][1])}]。"
                f"改用 element-weighted micro average 后，差距扩大到 **{pp(qif_micro['qad_minus_ste'])}**。在各自固定 master weights 的 FP-shadow/W4A4 开关测试中，QAD 同样高 **{pp(self_qif['qad_minus_ste'])}**。"
            ),
        },
        {"id": "qif-chart-block", "type": "chart", "chartId": "chart-qif-reference", "layout": "full"},
        {
            "id": "stage-result",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## QIF 反例不是单层异常：七个 stage 全部同向\n\n"
                "QAD 的 disagreement 在 Stem、三个 encoder 和三个 decoder stage 均高于 STE。差距最大的是 Encoder 1（+3.51 pp），其次是 Stem（+3.20 pp）和 Decoder 6（+2.26 pp）。"
                "逐层看，51/78 层显著更高，只有 10 层显著更低。这个分布不能支持“QAD 全局修复 code flipping”。"
            ),
        },
        {"id": "stage-chart-block", "type": "chart", "chartId": "chart-stage-delta", "layout": "full"},
        {
            "id": "feature-result",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## QAD 确实改善了 loss 直接约束的六个特征，但效应高度集中在 F6\n\n"
                f"六个特征的 normalized MSE 均下降，平均相对下降 **{pct(feature['relative_reduction_fraction'])}**；排除 F6 后仍下降 **{pct(feature_ex6['relative_reduction_fraction'])}**，说明并非完全由单点制造。"
                f"但按训练中实际使用的六项 raw MSE 求和，**{pct(primary['feature6_share_of_raw_mse_reduction'])}** 的 QAD-versus-STE 改善来自 F6。"
                "此外 F4 的 cosine distance 反而更差，说明“对齐改善”依赖 metric，最稳妥的表述应限定到训练采用的 MSE。"
            ),
        },
        {"id": "feature-chart-block", "type": "chart", "chartId": "chart-feature", "layout": "full"},
        {"id": "feature-table-block", "type": "table", "tableId": "table-features", "layout": "full"},
        {
            "id": "task-result",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## 输出层更稳定，但目前没有形成 representation–task 闭环\n\n"
                f"QAD 相对 teacher 的 prediction-flip rate 为 **{pct(task['qad']['prediction_flip_rate_vs_fp'])}**，STE 为 **{pct(task['ste']['prediction_flip_rate_vs_fp'])}**；"
                f"QAD 少 **{pp(task['ste']['prediction_flip_rate_vs_fp'] - task['qad']['prediction_flip_rate_vs_fp'])}**。QAD 的 mIoU 为 **{pct(task['qad']['miou'])}**，STE 为 **{pct(task['ste']['miou'])}**。"
                f"然而 source-level feature repair 与 mIoU gain 的 Spearman ρ 仅 **{primary['associations']['feature_repair_vs_source_miou_gain']['coefficient']:.3f}**，"
                "bootstrap CI 跨过零。因此当前只能说两个现象共存，不能说更强的 feature repair 导致更大的任务收益。"
            ),
        },
        {"id": "source-chart-block", "type": "chart", "chartId": "chart-source-closure", "layout": "full"},
        {"id": "class-table-block", "type": "table", "tableId": "table-classes", "layout": "full"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## 指标、样本与比较口径\n\n"
                "主分析覆盖 UDD6 validation 的 8,478 个 patch，并按 35 张原始 source image 聚类。QIF 指标覆盖 78 个实际触发节点；feature 指标覆盖 forward_qat 返回、且 QAD loss 实际使用的六个中间特征。"
                "共同-teacher 分析衡量训练后学生表示与 FP-QIF teacher 的偏离；self-sensitivity 分析固定每个 checkpoint 的 master weights 与 BN，仅在 QLayer 内关闭 weight/input fake quantization。"
                "后者衡量局部开关敏感性，不是另一个经过训练的 FP32 baseline。所有区间按 35 个 source cluster 重采样 10,000 次。"
            ),
        },
        {
            "id": "design",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## 实验设计与复现检查\n\n"
                "QAD、STE 和 FP teacher 使用相同 loader、输入顺序、T=1 和完整验证集。正式 probe 产生 1,322,568 条逐 patch/QIF 计数与 101,736 条逐 patch/feature 计数，行数与 2×8,478×78 和 2×8,478×6 完全一致。"
                "FP-QIF、QAD 与 STE 的 mIoU 分别精确复现为 67.8502%、61.9547% 和 58.1270%。macro 与 micro QIF 统计、common-teacher 与 self-shadow reference 都给出相同方向。"
            ),
        },
        {
            "id": "robustness",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## 稳健性检查揭示：QAD 更依赖量化后的执行点，而不是更接近自身 FP shadow\n\n"
                f"在 self-sensitivity probe 中，QAD 的 QIF disagreement 为 **{pct(self_qif['qad_mean'])}**，STE 为 **{pct(self_qif['ste_mean'])}**；QAD 的 feature NMSE 也高 **{pct(-self_feature['relative_reduction_fraction'])}**。"
                f"同时 QAD 的 W4A4 mIoU 比自身 FP shadow 高 **{pp(self_result['task']['qad_w4a4']['miou'] - self_result['task']['qad_fp']['miou'])}**。"
                "这说明 QAD 学到的是适应量化 forward 的解，而不是一个在开关量化时保持不变的解；因此不能把 FP shadow 当作理想目标，也不能用 self-distance 越小越好来替代任务证据。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "repair-analysis",
            "body": (
                "## 证据限制：当前最严重的问题仍是训练预算与方法特异性\n\n"
                "QAD checkpoint 经历 127 epochs，当前保留的 STE checkpoint 只有 16 epochs；因此精度、feature alignment 甚至 self-sensitivity 差异都可能部分来自优化时间。只有一个 seed，且没有 generic KD checkpoint。"
                "此外 QAD 本身就是 feature KD + STE-QAT，没有直接针对某个 Integer-LIF distortion 指标的 loss。即使 matched-budget 后结果仍成立，也更适合声称“distillation improves task-relevant feature alignment under W4A4”，"
                "而不是“a distortion-specific QAD repairs coupled Integer-LIF degradation”。"
            ),
        },
        {"id": "claims-heading", "type": "markdown", "body": "## 当前论文措辞的通过/阻断清单\n\n下表把观测事实、因果主张和方法特异性分开，避免用同一组数字支撑不同强度的结论。"},
        {"id": "claim-table-block", "type": "table", "tableId": "table-claims", "layout": "full"},
        {
            "id": "next",
            "type": "markdown",
            "body": (
                "## 最关键的下一步实验\n\n"
                "1. 重新生成与 QAD 完全同预算、同 LR horizon、同初始化的 127-epoch STE-QAT，并至少跑 3 seeds；这是解除当前最大 blocker 的最低要求。\n"
                "2. 在相同 teacher 下增加 generic feature-KD + QAT baseline。由于当前 QAD 没有 distortion-specific 组件，这个 baseline 实际上应与 QAD 定义重合；若确实重合，论文应主动把方法定位为训练框架/系统协同，而不是新型 loss。\n"
                "3. 做 feature-6 ablation 或只蒸馏 F6：因为 98.19% 的 raw KD-loss 改善来自 F6，这比继续扩大 layer-wise code 图更能解释方法作用。\n"
                "4. 预注册一个较窄的成功标准：matched-budget 下，QAD 同时降低 F6 MSE、prediction flips，并提高 mIoU；不要把全网 QIF code disagreement 设为预期修复终点，因为当前方向相反。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 仍待回答的问题\n\n"
                "- F6 alignment 是 mIoU 提升的必要条件，还是训练更久产生的伴随现象？\n"
                "- matched-budget STE 是否仍会出现更低的全网 code disagreement？\n"
                "- 只蒸馏 F6 能否复现完整 QAD，或前五个特征提供了训练稳定性而非最终精度？\n"
                "- 第二 seed/第二数据集上，feature alignment 与 output flip 的方向是否复现？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "QAD 对 W4A4 Integer-LIF 失真的修复程度与论文证据边界",
            "description": "Technical audit of whether QAD repairs observed W4A4 Integer-LIF distortion, using common-teacher and same-master-weight references.",
            "generatedAt": generated_at,
            "sources": [source],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": snapshot,
        },
        "sources": [source],
    }
    return artifact


if __name__ == "__main__":
    artifact = build()
    (OUTPUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT_DIR / "artifact.json")

