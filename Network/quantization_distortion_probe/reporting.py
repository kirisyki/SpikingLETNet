"""Human- and machine-readable result writers for the distortion probe."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .statistics import CLASS_NAMES, VARIANT_MODES


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "mode",
        "macro_disagreement",
        "macro_disagreement_ci_low",
        "macro_disagreement_ci_high",
        "macro_mae",
        "macro_signed_error",
        "miou",
        "pixel_accuracy",
        "prediction_flip_rate_vs_fp_qif",
        *[f"iou_{name}" for name in CLASS_NAMES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in ("fp_qif", *VARIANT_MODES):
            task = summary["task"][mode]
            row: dict[str, Any] = {
                "mode": mode,
                "miou": task["miou"],
                "pixel_accuracy": task["pixel_accuracy"],
                "prediction_flip_rate_vs_fp_qif": task.get(
                    "prediction_flip_rate_vs_fp_qif", 0.0
                ),
                **{
                    f"iou_{name}": task["per_class_iou"][name]
                    for name in CLASS_NAMES
                },
            }
            if mode != "fp_qif":
                rep = summary["representation"][mode]
                row.update(
                    {
                        "macro_disagreement": rep["macro_disagreement"][
                            "source_balanced_mean"
                        ],
                        "macro_disagreement_ci_low": rep["macro_disagreement"][
                            "cluster_bootstrap_95ci"
                        ][0],
                        "macro_disagreement_ci_high": rep["macro_disagreement"][
                            "cluster_bootstrap_95ci"
                        ][1],
                        "macro_mae": rep["macro_mae"]["source_balanced_mean"],
                        "macro_signed_error": rep["macro_signed_error"][
                            "source_balanced_mean"
                        ],
                    }
                )
            writer.writerow(row)


def write_report(path: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    decision = summary["preregistered_decision"]
    task_delta = summary["task_contrast"]
    lines = [
        "# Frozen-checkpoint Integer-LIF quantization distortion probe",
        "",
        "## Preregistered outcome",
        "",
        f"- Observable W4A4 distortion: **{decision['observable_w4a4_distortion']}**",
        f"- Coupled/superadditive distortion: **{decision['coupled_superadditive_distortion']}**",
        "",
        "The conclusion is restricted to the frozen SpikingLETNet-QIF checkpoint "
        "and the 35 UDD validation source images. It is not a cross-seed or "
        "cross-model generalization claim.",
        "",
        "## Representation results",
        "",
        "| Mode | Macro code disagreement | 95% cluster CI | Macro code MAE | Signed code error |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in VARIANT_MODES:
        rep = summary["representation"][mode]
        d = rep["macro_disagreement"]
        lines.append(
            f"| {mode} | {d['source_balanced_mean']:.6f} | "
            f"[{d['cluster_bootstrap_95ci'][0]:.6f}, "
            f"{d['cluster_bootstrap_95ci'][1]:.6f}] | "
            f"{rep['macro_mae']['source_balanced_mean']:.6f} | "
            f"{rep['macro_signed_error']['source_balanced_mean']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Task results",
            "",
            "| Mode | mIoU | Pixel accuracy | Prediction flip vs FP-QIF |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in ("fp_qif", *VARIANT_MODES):
        task = summary["task"][mode]
        lines.append(
            f"| {mode} | {task['miou']:.6f} | {task['pixel_accuracy']:.6f} | "
            f"{task.get('prediction_flip_rate_vs_fp_qif', 0.0):.6f} |"
        )

    lines.extend(
        [
            "",
            f"W4A4-input-QIF minus FP-QIF mIoU: "
            f"**{task_delta['w4a4_minus_fp_miou']:+.6f}**, 95% cluster CI "
            f"[{task_delta['cluster_bootstrap_95ci'][0]:+.6f}, "
            f"{task_delta['cluster_bootstrap_95ci'][1]:+.6f}].",
            "",
            "## Preregistered contrasts",
            "",
            "| Contrast | Mean | 95% cluster CI | Positive sources |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in summary["contrasts"].items():
        lines.append(
            f"| {name} | {result['source_balanced_mean']:+.6f} | "
            f"[{result['cluster_bootstrap_95ci'][0]:+.6f}, "
            f"{result['cluster_bootstrap_95ci'][1]:+.6f}] | "
            f"{result['positive_source_count']}/{result['source_count']} |"
        )

    lines.extend(
        [
            "",
            "## Protocol boundary",
            "",
            f"- Validation patches: {manifest['completed_patches']}",
            f"- Source-image clusters: {summary['source_count']}",
            f"- Declared QIF nodes: {manifest['declared_qif_count']}",
            f"- Active QIF nodes: {summary['active_layer_count']}",
            f"- Runtime time steps: {manifest['protocol']['time_steps']}",
            f"- Validation batch size: {manifest['protocol']['batch_size']}",
            "- All configurations retain the same QIF code range `{0,...,8}`.",
            "- `A4-input` refers to historical operator-input fake quantization, "
            "not replacement of QIF by a continuous A32 neuron.",
            "- This is a frozen-checkpoint perturbation probe, not a comparison of "
            "trained QAT and QAD models.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
