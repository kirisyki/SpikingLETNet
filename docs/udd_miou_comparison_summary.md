# UDD 对比实验 mIoU 汇总

更新日期：2026-08-20

## 口径与范围

本文档汇总 LETNet 项目中可归档的正式、完整 UDD6 validation mIoU 结果，方便论文引用、文件传输和仓库迁移。

- 默认评估集合：`/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt`。
- 默认评估规模：8,478 个 validation patch，来自 35 张原始图像。
- 默认 mIoU 口径：累计全部 patch 的一个六分类全局混淆矩阵，再计算六类 IoU 的算术平均值。
- 表中的 `pp` 表示绝对 mIoU 百分点，即 `100 × ΔmIoU`。
- 已排除 smoke、preflight、训练过程中的 `best_miou` 日志值和重复生成的报告镜像。
- `source-balanced mIoU` 是先按每张原始图像计算 mIoU，再对 35 张原图平均；该口径会单独标识，不能与全局 mIoU 直接混排。

## 总表

| 实验组 | 方法/配置 | T | mIoU | mIoU (%) | 组内变化（pp） | 参照与说明 |
|---|---|---:|---:|---:|---:|---|
| [W4A4 方法对比](../quantization_comparison_results/seed1234/report.md) | FP32 | 1 | 0.678502 | 67.8502 | 0 | 组内 FP32 基线 |
| W4A4 方法对比 | QAD W4A4 | 1 | 0.619547 | 61.9547 | −5.8955 | 相对 FP32 |
| W4A4 方法对比 | STE-QAT W4A4 | 1 | 0.581270 | 58.1270 | −9.7232 | 相对 FP32；QAD 比其高 3.8277 pp |
| W4A4 方法对比 | LSQ-QAT W4A4 | 1 | 0.537915 | 53.7915 | −14.0588 | 相对 FP32 |
| W4A4 方法对比 | EWGS-QAT W4A4 | 1 | 0.472837 | 47.2837 | −20.5666 | 相对 FP32 |
| [权重/激活位宽对比](../ternary_QAT_checkpoint/comparison_seed1234/report.md) | FP32，W32/A32 | 1 | 0.678502 | 67.8502 | 0 | 组内 FP32 基线 |
| 权重/激活位宽对比 | W4/A4 | 1 | 0.619547 | 61.9547 | −5.8955 | 与 QAD checkpoint 相同 |
| 权重/激活位宽对比 | W1.58/A4 | 1 | 0.591090 | 59.1090 | −8.7412 | 比 W4/A4 低 2.8457 pp |
| 权重/激活位宽对比 | W1.58/A1.58 | 1 | 0.223259 | 22.3259 | −45.5243 | 相对 FP32；低比特激活导致严重退化 |
| [QAD 与 SQUAT 路线对比](../squat_experiment_outputs/seed1234/evaluation/report.md) | 历史 FP-QIF | 1（QIF） | 0.678502 | 67.8502 | 0 | QAD 路线量化前基线 |
| QAD 与 SQUAT 路线对比 | 历史 QAD W4A4 | 1（QIF） | 0.619547 | 61.9547 | −5.8955 | 相对历史 FP-QIF |
| QAD 与 SQUAT 路线对比 | 新 FP32 LIF-SNN | 8 | 0.689462 | 68.9462 | 0 | 直接 SNN 路线量化前基线 |
| QAD 与 SQUAT 路线对比 | W4M4S1 QAT+SQUAT | 8 | 0.315514 | 31.5514 | −37.3948 | 相对新 FP32 LIF；比 QAD 低 30.4033 pp |
| [首层输入精度消融](../first_input_fp_results/full_existing_checkpoints_20260805/report.md) | QAD：首层输入 A4 | 1 | 0.619547 | 61.9547 | 0 | QAD 组参照 |
| 首层输入精度消融 | QAD：首层输入 FP32 | 1 | 0.622754 | 62.2754 | +0.3207 | 仅首层图像输入不量化 |
| 首层输入精度消融 | STE：首层输入 A4 | 1 | 0.581270 | 58.1270 | 0 | STE 组参照 |
| 首层输入精度消融 | STE：首层输入 FP32 | 1 | 0.586023 | 58.6023 | +0.4753 | 仅首层图像输入不量化 |
| [冻结 FP checkpoint 扰动](../quantization_distortion_results/full_20260805/report.md) | FP-QIF，W32/A32 | 1 | 0.678502 | 67.8502 | 0 | 冻结 FP checkpoint |
| 冻结 FP checkpoint 扰动 | W4/A32 | 1 | 0.405185 | 40.5185 | −27.3318 | 只施加权重量化 |
| 冻结 FP checkpoint 扰动 | W32/A4-input | 1 | 0.569789 | 56.9789 | −10.8714 | 只施加 operator-input 量化 |
| 冻结 FP checkpoint 扰动 | W4/A4-input | 1 | 0.345938 | 34.5938 | −33.2564 | 联合扰动；不是训练后的 QAD |
| [同 checkpoint FP-shadow 敏感性](../qad_distortion_repair_results/self_sensitivity_full_20260806/analysis_summary.json) | QAD FP shadow | 1 | 0.524277 | 52.4277 | 0 | 固定 QAD 权重，关闭 fake quant |
| FP-shadow 敏感性 | QAD W4A4 | 1 | 0.619547 | 61.9547 | +9.5270 | 相对自身 FP shadow |
| FP-shadow 敏感性 | STE FP shadow | 1 | 0.565157 | 56.5157 | 0 | 固定 STE 权重，关闭 fake quant |
| FP-shadow 敏感性 | STE W4A4 | 1 | 0.581270 | 58.1270 | +1.6113 | 相对自身 FP shadow |
| [共同 teacher 诊断](../qad_distortion_repair_results/qad_vs_ste16_full_20260806/analysis_summary.json) | QAD，source-balanced mIoU | 1 | 0.544429 | 54.4429 | +3.8209 | 相对 STE；非全局混淆矩阵口径 |
| 共同 teacher 诊断 | STE，source-balanced mIoU | 1 | 0.506220 | 50.6220 | 0 | 非全局混淆矩阵口径 |

## 建议用于论文主表的结果

若论文需要一张紧凑的主精度对比表，建议优先使用以下三个实验组：

1. W4A4 方法对比：FP32、QAD、STE-QAT、LSQ-QAT、EWGS-QAT。
2. 权重/激活位宽对比：FP32、W4/A4、W1.58/A4、W1.58/A1.58。
3. 完整工程路线对比：QAD 与 W4M4S1 QAT+SQUAT，同时分别报告各自的量化前 FP 基线。

首层输入精度、冻结 checkpoint 扰动和 FP-shadow 结果属于消融或诊断实验，更适合放在消融表、补充材料或机制分析部分。

## 数据质量与可比性说明

1. QAD、FP32 和 STE 的部分数值出现在多个报告中，但它们来自相同 checkpoint 和相同评估结果，不是独立重复实验。
2. 主 W4A4 方法对比的训练预算不相等：历史 QAD 观察了约 127 epochs，STE-QAT、LSQ-QAT 和 EWGS-QAT 使用 16-epoch 缩短预算。
3. 当前归档结果主要来自 seed 1234，不能直接声称跨 seed 统计稳定性。
4. QAD 与 SQUAT 比较是完整工程路线对比：QAD 使用单步 QIF compact-code 路线，直接 LIF-SNN/SQUAT 使用 `T=8`；两条路线的神经元、训练预算、蒸馏和状态量化设置并非单因素匹配。
5. `FP shadow` 是固定已训练权重后关闭 fake quant 的局部反事实，不是单独训练的 FP32 baseline。其正向 mIoU 差值不能解释为“量化天然提升精度”。
6. `A4-input` 表示历史 operator-input fake quantization，不表示用连续 A32 神经元替换 QIF。
7. 当前没有独立 test split；validation 同时承担 checkpoint 选择与最终报告，因此文中应称为 validation mIoU。
8. TensorRT 能耗实验记录了 `accuracy_evaluated=false`，本表数值不能称为 TensorRT deployed mIoU。

## 结构化结果来源

- W4A4 方法对比：[`quantization_comparison_results/seed1234/comparison.json`](../quantization_comparison_results/seed1234/comparison.json)
- 权重/激活位宽对比：[`ternary_QAT_checkpoint/comparison_seed1234/comparison.json`](../ternary_QAT_checkpoint/comparison_seed1234/comparison.json)
- QAD 与 SQUAT 路线对比：[`squat_experiment_outputs/seed1234/evaluation/comparison.json`](../squat_experiment_outputs/seed1234/evaluation/comparison.json)
- 首层输入精度消融：[`first_input_fp_results/full_existing_checkpoints_20260805/results.json`](../first_input_fp_results/full_existing_checkpoints_20260805/results.json)
- 冻结 checkpoint 扰动：[`quantization_distortion_results/full_20260805/summary.json`](../quantization_distortion_results/full_20260805/summary.json)
- FP-shadow 敏感性：[`qad_distortion_repair_results/self_sensitivity_full_20260806/analysis_summary.json`](../qad_distortion_repair_results/self_sensitivity_full_20260806/analysis_summary.json)
- 共同 teacher 诊断：[`qad_distortion_repair_results/qad_vs_ste16_full_20260806/analysis_summary.json`](../qad_distortion_repair_results/qad_vs_ste16_full_20260806/analysis_summary.json)

