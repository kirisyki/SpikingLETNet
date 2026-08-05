#!/usr/bin/env python3
"""Evaluate new best checkpoints and compare against frozen historical QAD."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

NETWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NETWORK_DIR))
os.chdir(NETWORK_DIR)

import torch  # noqa: E402

from squat_comparison.audit import weight_and_storage_audit  # noqa: E402
from squat_comparison.model_factory import (  # noqa: E402
    build_fp32_lif_model,
    build_squat_from_fp32,
)
from squat_comparison.protocol import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_CONFIG,
    OUTPUT_ROOT,
    ExperimentProtocol,
    atomic_write_json,
    frozen_historical_results,
    prepare_new_directory,
)
from squat_comparison.training import create_loaders, validate  # noqa: E402


def load_state(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in payload:
        raise KeyError(f"missing model_state in {path}")
    return payload


def decision(delta: float, band: float) -> str:
    if delta > band:
        return "SQUAT 路线精度更高"
    if delta < -band:
        return "QAD 路线精度更高"
    return "两条路线精度相当"


def markdown_report(result: dict[str, Any]) -> str:
    qad = result["historical"]["qad"]
    fp = result["new_results"]["fp32_lif_snn"]
    squat = result["new_results"]["w4m4s1_squat"]
    lines = [
        "# QAD 与直接 QAT+SQUAT 路线对比结果",
        "",
        "## 结论",
        "",
        f"SQUAT mIoU 为 **{squat['miou']:.6f}**，历史冻结 QAD mIoU 为 "
        f"**{qad['miou']:.6f}**；差值为 **{result['comparison']['delta_route']:+.6f}**。"
        f"按预注册 ±0.005 解释带，结论为：**{result['comparison']['decision']}**。",
        "",
        "| 路线 | mIoU | pixel accuracy |",
        "|---|---:|---:|",
        f"| 历史 QAD（冻结） | {qad['miou']:.6f} | 未在本实验重跑 |",
        f"| 新 FP32 LIF-SNN（诊断基线） | {fp['miou']:.6f} | {fp['pixel_accuracy']:.6f} |",
        f"| 新 W4M4S1 QAT+SQUAT | {squat['miou']:.6f} | {squat['pixel_accuracy']:.6f} |",
        "",
        "## 逐类 IoU",
        "",
        "| 类别 | QAD | FP32 LIF-SNN | SQUAT | SQUAT-QAD |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, name in enumerate(CLASS_NAMES):
        qad_value = qad["per_class_iou"][name]
        fp_value = fp["per_class_iou"][index]
        squat_value = squat["per_class_iou"][index]
        lines.append(
            f"| {name} | {qad_value:.6f} | {fp_value:.6f} | {squat_value:.6f} | "
            f"{squat_value - qad_value:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 量化损失",
            "",
            f"- 新 SQUAT 相对自身 FP32 LIF-SNN：{result['comparison']['squat_quantization_loss']:+.6f}。",
            f"- 历史 QAD 相对历史 FP32 QIF：{result['comparison']['historical_qad_quantization_loss']:+.6f}。",
            "",
            "## 解释限制",
            "",
            "1. QAD 是历史冻结结果；本次没有训练、评估、转换或校准 QAD。",
            "2. 只有 seed 1234，不支持统计显著性或跨 seed 稳健性结论。",
            "3. 两条路线的神经元、状态/激活表示、蒸馏、码本和优化器不同，因此是完整路线比较，不是训练域的单因素因果消融。",
            "4. 历史 QAD 使用 feature distillation；SQUAT 不使用教师。",
            "5. SQUAT 保留 softmax、LayerNorm、ReLU、sigmoid 和插值等 FP32 连续模块，是混合 SNN。",
            "6. M4 使用 snnTorch 公开参考代码的固定阈值相对范围；论文正文存在动态范围表述差异。",
            "7. validation 同时用于模型选择和最终报告，没有独立 test split。",
            "8. 若 preflight 触发预算缩放，本结果只表示缩短预算下的路线表现。",
            "9. PyTorch fake quantization 的耗时不是低比特硬件加速或能耗证据。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    protocol = ExperimentProtocol()
    evaluation_dir = prepare_new_directory(OUTPUT_ROOT / "evaluation")
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(protocol.seed)
    _, validation_loader = create_loaders(
        protocol,
        physical_batch=1,
        generator=generator,
    )
    fp_payload = load_state(OUTPUT_ROOT / "fp32_snn/checkpoint_best.pth")
    fp_model, _ = build_fp32_lif_model(
        config=DEFAULT_CONFIG, classes=protocol.classes, seed=protocol.seed
    )
    fp_model.load_state_dict(fp_payload["model_state"], strict=True)
    fp_model.to(device)
    fp_result = validate(
        model=fp_model,
        loader=validation_loader,
        protocol=protocol,
        device=device,
        collect_audit=True,
    )
    fp_model.cpu()
    torch.cuda.empty_cache()

    squat_model, _ = build_squat_from_fp32(
        fp_model, expected_weight_layers=protocol.expected_weight_layers
    )
    squat_payload = load_state(OUTPUT_ROOT / "w4m4s1_squat/checkpoint_best.pth")
    squat_model.load_state_dict(squat_payload["model_state"], strict=True)
    squat_model.to(device)
    squat_result = validate(
        model=squat_model,
        loader=validation_loader,
        protocol=protocol,
        device=device,
        collect_audit=True,
    )
    storage = weight_and_storage_audit(squat_model)
    historical = frozen_historical_results(protocol)
    delta = squat_result["miou"] - historical["qad"]["miou"]
    result = {
        "historical": historical,
        "new_results": {
            "fp32_lif_snn": fp_result,
            "w4m4s1_squat": squat_result,
        },
        "comparison": {
            "delta_route": delta,
            "comparable_band": protocol.comparable_band,
            "decision": decision(delta, protocol.comparable_band),
            "squat_quantization_loss": squat_result["miou"] - fp_result["miou"],
            "historical_qad_quantization_loss": (
                historical["qad"]["miou"] - historical["fp32_qif"]["miou"]
            ),
        },
        "weight_and_storage_audit": storage,
        "preflight": json.loads((OUTPUT_ROOT / "preflight.json").read_text()),
        "training": {
            "fp32": json.loads((OUTPUT_ROOT / "fp32_snn/training_summary.json").read_text()),
            "squat": json.loads((OUTPUT_ROOT / "w4m4s1_squat/training_summary.json").read_text()),
        },
    }
    atomic_write_json(evaluation_dir / "comparison.json", result)
    (evaluation_dir / "report.md").write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()

