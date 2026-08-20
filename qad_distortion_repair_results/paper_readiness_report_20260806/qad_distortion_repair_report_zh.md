# QAD 是否修复了 W4A4 Integer-LIF 量化失真？

**技术证据报告——中文版**  
**分析日期：** 2026 年 8 月 6 日  
**当前结论：** 现有实验支持对 QAD 作较窄的“蒸馏特征对齐”解释，但**不支持**“QAD 全局修复了 Integer-LIF code distortion”这一更强主张，也不能据此因果地断言 QAD 公平优于 STE-QAT。

## 技术摘要

量化失真在该模型中是明确可观测的。对冻结的 FP checkpoint 施加 W4A4 扰动，会引起显著的内部表示和任务输出变化。在已训练 checkpoint 的对比中，QAD 达到 **61.95% mIoU**，保留的 16-epoch STE-QAT checkpoint 为 **58.13%**。

但深入分析改变了对 QAD 作用机制的解释：

- 以同一个 FP-QIF teacher 为参照，QAD 在六个被直接蒸馏的特征上，相对 STE-QAT 将 normalized MSE 降低 **31.85%**，并将相对 teacher 的预测分歧率降低 **1.80 个百分点**。
- 特征效果高度集中：最终 decoder/pre-logit 特征 F6 贡献了六特征 raw KD MSE 总降幅的 **98.19%**。排除 F6 后，normalized MSE 的相对降幅为 **10.42%**。
- QAD **没有**提高全局 Integer-LIF code fidelity。其宏观 code disagreement 为 **11.64%**，STE-QAT 为 **10.08%**；QAD−STE 为 **+1.56 个百分点**，按原图像聚类 bootstrap 的 95% CI 为 **[+1.53, +1.59]**。35/35 个原始图像和全部七个网络 stage 的方向都相同。
- 同一 checkpoint 的 FP-shadow 敏感性实验得到同样结论：关闭 W4A4 量化时，QAD 的变化更大，而不是更小。这符合“模型适应了量化前向路径”，不符合“内部表示整体更接近 FP”的解释。
- 在 35 张原始验证图像之间，无论 QIF code repair 还是蒸馏特征 repair，都不能预测 QAD 相对 STE 的逐图 mIoU 增益。因此，现有结果没有建立“失真修复 → 精度恢复”的因果闭环。

目前最稳妥的解释是：**QAD 学到了对任务有用的 W4A4 表示，并使被直接蒸馏的多尺度特征——尤其 F6——更接近 teacher，但没有把全局逐层 Integer-LIF code 恢复到 FP teacher。** 由于 QAD 与保留的 STE-QAT 实验训练预算不相等，且只比较了一组 checkpoint/seed，这一结论仍属于描述性证据。

## 1. 研究问题与结论边界

本分析区分两个问题：

1. W4A4 量化失真是否能在 Integer-LIF 分割模型中稳定观测？
2. 现有证据能否证明 QAD 修复了该失真，并且这种修复解释了任务精度？

第一个问题已得到支持。第二个问题只在“被直接蒸馏的特征 target”这一层面得到支持；对全局 QIF code fidelity 和 mIoU 的因果解释均不成立。

这一边界对论文叙事很重要。当前方法实质上是高精度 teacher 的多尺度 feature distillation 与 STE-based QAT 的结合，并没有直接惩罚全局 QIF code disagreement 的 loss。因此，机制主张应与真实优化目标和观测结果一致。

## 2. 实验范围与指标定义

### 2.1 评估样本

- 数据集：完整 UDD6 validation split。
- 评估规模：**8,478 个 patch**，来自 **35 张原始图像**。
- 模型表示：**78 个活跃 QIF 节点**、六个蒸馏特征、运行时 `T = 1`、QIF code 范围 `[0, 8]`。
- 不确定性估计：**10,000 次原图像聚类 bootstrap**。统计单元是原始图像，而不是相互重叠的单个 patch。
- Checkpoint：共同的 FP-QIF teacher、历史 QAD best checkpoint、保留的 16-epoch STE-QAT best checkpoint。
- 已知混杂因素：QAD 训练共观测 127 epochs，best checkpoint 位于 epoch 119；保留的 STE-QAT 训练为 16 epochs，best checkpoint 位于 epoch 14。

### 2.2 两种参照设计

**共同 teacher 审计。** 将 QAD 和 STE-QAT 都与同一个独立训练的 FP-QIF teacher 比较，回答哪个 student 在 QIF 输出、KD 特征和最终预测上更接近 teacher。

**同 checkpoint FP-shadow 审计。** 深拷贝各自 checkpoint，固定 master weights 和 BN 状态，只关闭 QLayer 的 weight/input fake quantization，测量每个已学习解对自身 W4A4 operator 的敏感性。FP shadow 是局部反事实参照，**不是**独立训练的 FP 任务 baseline。

**冻结 checkpoint 量化扰动审计。** 在原始 FP checkpoint 上评估 W32A32、W4A32、W32A4 与 W4A4 扰动。它能够证明量化失真存在，但不能证明 QAD 修复了失真。

### 2.3 指标

- **QIF code disagreement：** student 与指定参照之间，离散 Integer-LIF 输出 code 不相等的元素比例；越低越接近参照。
- **QIF MAE：** student 与参照 QIF code 的平均绝对差；越低越接近。
- **Macro 统计：** 对原图像和层做平衡聚合，避免大 feature map 或 patch 数较多的原图像主导结果。
- **Micro 统计：** 对全部记录元素聚合，较大的 tensor 会按元素数获得更大权重。
- **Teacher-referenced normalized MSE：** 用 teacher 特征能量归一化的 feature MSE，适合比较不同尺度的特征。
- **Cosine distance：** 衡量特征方向差异，与 MSE 互补。
- **Prediction flip rate：** 输出像素预测类别与参照模型不一致的比例。
- **Source-balanced mIoU：** 先按每张原始图像计算 mIoU，再对原图像取平均；它不同于基于全局 confusion matrix 的 mIoU。

## 3. 量化失真明确存在，但尚未证明联合量化具有超加性

冻结 FP checkpoint 在 W4A4 扰动下发生了明显退化：

| 冻结 checkpoint 结果 | FP/W32A32 | W4A4 | 变化 |
|---|---:|---:|---:|
| 全局 mIoU | 67.85% | 34.59% | −33.26 pp |
| 相对 FP 的 prediction flip rate | 0% | 38.70% | +38.70 pp |
| Macro QIF code disagreement | 0% | 11.38% | +11.38 pp |

这组结果建立了一个可重复的 W4A4 dense-prediction failure mode：当高精度 checkpoint 暴露于联合低比特算子时，内部离散 code 和语义预测都会显著改变。

但它**没有**证明权重量化与激活量化之间存在超加性。现有 W4A32、W32A4、W4A4 分解能说明联合失真可观测，却没有可靠证明交互项大于两个单独效应之和。论文可以描述“joint quantization”或“coupled deployment constraints”，但不应宣称已经证明了 superadditive degradation mechanism。

## 4. QAD 改善了被直接蒸馏的特征

以共同 FP-QIF teacher 为参照，QAD 在其实际优化的 loss family 上比 STE-QAT 更接近 teacher。

| Teacher-referenced 特征结果 | QAD | STE-QAT | QAD 效果 |
|---|---:|---:|---:|
| F1–F6 平均 normalized MSE | 0.6946 | 1.0193 | 降低 31.85%，95% CI [30.77%, 32.95%] |
| F1–F6 平均 cosine distance | 0.3733 | 0.3930 | 降低 5.02% |
| F1–F6 平均 raw MSE | 0.3283 | 1.5318 | 降低 78.57% |
| 仅 F1–F5 平均 normalized MSE | 0.7962 | 0.8888 | 降低 10.42% |

六个蒸馏特征的 normalized MSE 都有所改善：

| 特征 | QAD 的 normalized-MSE 相对降幅 | 95% CI | 解释 |
|---|---:|---:|---|
| F1 | 4.30% | [3.30%, 5.26%] | 小幅改善 |
| F2 | 9.60% | [8.89%, 10.34%] | 中等改善 |
| F3 | 17.24% | [16.08%, 18.39%] | 明确改善 |
| F4 | 13.12% | [11.26%, 15.00%] | MSE 改善，但 cosine distance 变差 |
| F5 | 7.09% | [5.53%, 8.74%] | 中等改善 |
| F6 | 88.85% | [87.64%, 90.00%] | 占主导的最终 decoder/pre-logit 效应 |

因此，该效果真实存在，但并不均匀。F6 贡献了六特征 raw KD MSE 总降幅的 **98.19%**。此外，尽管 F4 normalized MSE 改善，其 cosine distance 对 QAD 反而更差 0.0212，95% CI 为 **[+0.0149, +0.0286]**。“特征对齐”必须绑定到明确指标，不能描述为所有几何意义上的普遍改善。

现有数据最支持的机制假设是：QAD 主要正则化了靠近任务输出端的后期 decoder 表示，同时在较早的 KD targets 上带来较小的尺度归一化改善。

## 5. QAD 没有修复全局 Integer-LIF code fidelity

全局 QIF 统计与“code repair”主张的方向相反。

| 共同 FP teacher 下的 QIF 结果 | QAD | STE-QAT | QAD − STE | 差值的 95% CI |
|---|---:|---:|---:|---:|
| Macro code disagreement | 11.64% | 10.08% | +1.56 pp | [+1.53, +1.59] pp |
| Micro code disagreement | 14.96% | 12.07% | +2.89 pp | [+2.84, +2.94] pp |
| Macro code MAE | 0.1446 | 0.1188 | +0.0258 | [+0.0253, +0.0264] |
| 绝对 signed-code bias | 0.0413 | 0.0153 | +0.0260 | [+0.0251, +0.0269] |
| 绝对 zero-rate 变化 | 1.96% | 1.11% | +0.85 pp | [+0.82, +0.89] pp |

QAD 在 **35/35 张原始图像**上的 macro disagreement 都更高；双侧精确符号检验为 `p = 5.82 × 10⁻¹¹`。七个 stage 的方向也全部一致：

| Stage | QAD − STE macro disagreement |
|---|---:|
| Stem | +3.20 pp |
| Encoder 1 | +3.51 pp |
| Encoder 2 | +0.71 pp |
| Encoder 3 | +1.21 pp |
| Decoder 4 | +0.46 pp |
| Decoder 5 | +0.88 pp |
| Decoder 6 | +2.26 pp |

在单个节点层面，78 个 QIF 节点中，QAD 有 11 个节点的平均 disagreement 更低、54 个更高、13 个相同。这排除了结果只由一个异常层或少数样本造成的解释。

这并不代表 QAD 无效，而是说明：**该 checkpoint 的收益并不是通过恢复 FP-like discrete codes 表达的。** 多种不同的内部 code 配置可能产生同样有用的下游特征和预测。

## 6. FP-shadow 实验表明量化适应，而不是关闭量化时更稳健

同一 master weight 的敏感性实验独立否定了全局修复解释。

| 自身 FP shadow → W4A4 结果 | QAD | STE-QAT | QAD − STE |
|---|---:|---:|---:|
| Macro QIF disagreement | 9.31% | 7.75% | +1.56 pp，95% CI [+1.46, +1.67] |
| Micro QIF disagreement | 11.08% | 9.33% | +1.75 pp |
| Macro QIF MAE | 0.1155 | 0.0913 | +0.0242 |
| Feature normalized MSE | 0.5615 | 0.5129 | QAD 高 9.48% |
| Feature cosine distance | 0.2788 | 0.2576 | QAD 更高 |

关闭 fake quantization 也会降低任务性能，而不是改善：

- QAD FP shadow：**52.43% mIoU**；QAD W4A4：**61.95%**。source-balanced 的 W4A4−shadow 增益为 **+8.44 pp**，95% CI **[+6.70, +10.21]**。
- STE FP shadow：**56.52% mIoU**；STE W4A4：**58.13%**。source-balanced 增益为 **+1.26 pp**，95% CI **[+0.14, +2.43]**。

这不能解释为“量化天然提高精度”，因为 FP shadow 并不是单独训练的 FP 模型。它说明已训练 weights、BN statistics、scales 与非线性工作点共同适应了量化前向路径；QAD 的这种协同适应比短训练预算的 STE checkpoint 更强。

## 7. QAD 相对 STE16 的任务精度更高，但因果链尚未闭合

| 完整验证集任务结果 | FP teacher | QAD | STE-QAT（16 epochs） |
|---|---:|---:|---:|
| 全局 mIoU | 67.85% | 61.95% | 58.13% |
| Pixel accuracy | 84.20% | 81.18% | 79.13% |
| 相对 FP teacher 的 prediction flip rate | — | 14.90% | 16.69% |

QAD 比保留的 STE checkpoint 高 **3.83 个 mIoU 点**，同时将相对 teacher 的 prediction flip rate 降低 **1.80 个百分点**，即相对减少 **10.77%**。在 source-balanced mIoU 上，QAD 在 35 张原始图像中的 29 张更高，平均差值为 **+3.82 个点**，95% CI **[+2.67, +5.00]**。

这些是有效的 checkpoint-level 描述性结果，但由于 127-versus-16-epoch 的训练预算不匹配，它们不能作为 QAD 方法因果收益的公平估计。

尝试建立“失真 → 修复 → 性能”的闭环也得到负结果：

| 逐原图像相关性，n = 35 | Spearman ρ | p-value | 95% 聚类 bootstrap CI |
|---|---:|---:|---:|
| QIF-code repair vs mIoU gain | −0.004 | 0.983 | [−0.318, +0.329] |
| KD-feature repair vs mIoU gain | −0.184 | 0.289 | [−0.472, +0.139] |

置信区间同时包含有意义的正相关和负相关。因此，当前 35 张原图像的研究没有证据表明“feature repair 更多的图像会获得更大的 mIoU 增益”。总体特征对齐改善与总体任务性能改善可以同时出现，但这并不能建立逐样本中介关系或因果性。

## 8. 论文主张证据审计

| 候选主张 | 状态 | 原因 |
|---|---|---|
| W4A4 会在冻结 FP 模型中造成可观测的 Integer-LIF 表示和分割失真。 | **支持** | 在完整验证集上出现大幅、可重复的 code 与任务变化。 |
| 以 normalized MSE 衡量，QAD checkpoint 的六个被直接蒸馏特征比 STE16 更接近 teacher。 | **描述性支持** | 六个特征均改善，但效果由 F6 主导。 |
| QAD 相对 STE16 产生更少的 FP-teacher 最终预测分歧。 | **描述性支持** | Prediction flip rate 为 14.90% vs 16.69%。 |
| QAD 修复或抑制了全局 Integer-LIF code distortion。 | **被当前测量反驳** | 两种参照设计下，QIF disagreement 和 MAE 都更高。 |
| QAD 解决了已被证明的权重–激活超加性退化。 | **不支持** | 联合失真存在，但没有证明超加性交互。 |
| Feature repair 导致了 mIoU 提升。 | **不支持** | 无逐原图相关性，也没有 targeted ablation 或 mediation experiment。 |
| QAD 公平且稳定地优于常规 QAT。 | **尚不支持** | 训练预算不等、仅一组 checkpoint/seed、没有可归档的 matched-budget STE 结果。 |
| QAD 是 distortion-specific 方法。 | **当前设计不支持** | 实现是通用多尺度 feature KD + STE-QAT，没有直接 QIF-distortion loss。 |

### 当前可安全写入论文的表述

> In the examined W4A4 checkpoint, quantization-aware feature distillation improves alignment with the FP teacher at the explicitly distilled multi-scale features, especially the final decoder/pre-logit feature, and is accompanied by fewer output prediction disagreements than the retained short-budget STE-QAT checkpoint.

在训练预算和多个 seed 公平匹配之前，必须将该比较明确标为 diagnostic 或 preliminary。

### 当前应避免的表述

- “QAD repairs global Integer-LIF quantization distortion.”
- “QAD resolves coupled or superadditive W4A4 degradation.”
- “The feature-alignment recovery explains or causes the mIoU gain.”
- 在不注明训练预算差异的情况下写 “QAD substantially outperforms conventional QAT”。

## 9. 局限性与稳健性检查

1. **训练预算不匹配是最主要的局限。** QAD 与 STE16 属于不同训练 regime，其差异混合了 objective、schedule 与 optimization maturity。
2. **只审计了一组 checkpoint/seed。** 原图像聚类 bootstrap 只量化验证集不确定性，不能替代独立训练 seed。
3. **缺少 generic KD baseline。** 现有证据无法区分普通 teacher feature supervision 的收益与任何特别的 quantization-aware 组件。
4. **F6 主导 raw KD 结果。** 这是有价值的定位结果，但也意味着六特征 aggregate 可能夸大广泛 representation repair。
5. **结果依赖指标。** F4 的 normalized MSE 改善，但 cosine distance 变差；raw MSE 还对特征尺度敏感。
6. **FP shadow 不是精度 baseline。** 它只是固定已训练 weights 后的局部 operator-sensitivity counterfactual。
7. **目前没有可归档的 matched-budget STE 结果。** 未归档的观测不能作为论文证据，必须按冻结协议重跑。
8. **分析只覆盖 UDD6 和一个 architecture。** 第二数据集或第二个 SNN segmentation model 上的泛化仍未验证。

已完成的稳健性工作包括：完整 validation split 推理、checkpoint hash 核验、以原图像为 cluster 的不确定性估计、macro 与 micro 指标、共同 teacher 与同 checkpoint 两种参照设计、逐 stage 定位，以及原始行数/计数一致性检查。相关统计测试全部通过（`8 passed`）。

## 10. 使机制叙事达到可投稿强度的最小实验

建议按以下顺序执行：

1. **Matched-budget STE-QAT，至少三个 seeds。** 与 QAD 统一初始化/预训练来源、127-epoch horizon、optimizer、scheduler、augmentation、batch size、resolution、checkpoint selection 和评估协议。
2. **Matched generic-KD baseline。** 在同一个 FP teacher 和 W4A4 student 下，比较 task-only STE-QAT、logit KD + STE-QAT、当前 multi-scale feature KD + STE-QAT，以及最终命名方法。
3. **F6 定位消融。** 比较无 feature KD、仅 F6 KD、仅 F1–F5 KD、F1–F6 KD；预先指定 F6 normalized MSE、prediction flip rate 与 mIoU 为 primary endpoints。
4. **跨 seeds 重复诊断。** 对每个 seed 重算共同 teacher 下的 feature 与 QIF 指标，不能从单个被选中的 checkpoint 推断方法级行为。
5. **增加一个泛化轴。** 在第二数据集或第二个可部署的 SNN segmentation architecture 上重复公平对照。

如果公平实验确认 QAD 具有更好的 F6 对齐、更少的输出 flip 和更高 mIoU，但全局 QIF disagreement 仍不降低，那么论文应将 QAD 描述为 **W4A4 训练中的 task-relevant feature preservation**，而不是 global integer-code repair。

## 11. 结论

现有实验足以说明 W4A4 失真存在，也足以说明被检查的 QAD checkpoint 相较保留的短训练 STE checkpoint，更好地保留了 teacher 的显式蒸馏特征与最终决策。但它们**不足以**证明 QAD 全局修复了 Integer-LIF code distortion，不足以证明 feature change 导致 accuracy gain，也不足以证明 QAD 公平优于 QAT。

对当前论文而言，最准确的现时结论是：

> QAD appears to improve task-facing, teacher-aligned feature representations—dominated by the final decoder/pre-logit target—while allowing internal discrete QIF codes to depart further from the FP teacher. Its role is better described as quantized representation adaptation and task-relevant feature preservation than as global code-distortion repair.

## 12. 可复现性与源文件

- [共同 teacher 分析汇总](../qad_vs_ste16_full_20260806/analysis_summary.json)
- [共同 teacher 实验 manifest](../qad_vs_ste16_full_20260806/manifest.json)
- [逐特征统计](../qad_vs_ste16_full_20260806/feature_statistics.csv)
- [逐层统计](../qad_vs_ste16_full_20260806/layer_statistics.csv)
- [逐 stage 统计](../qad_vs_ste16_full_20260806/stage_statistics.csv)
- [同 checkpoint 敏感性分析汇总](../self_sensitivity_full_20260806/analysis_summary.json)
- [同 checkpoint 敏感性实验 manifest](../self_sensitivity_full_20260806/manifest.json)
- [冻结 checkpoint 量化失真报告](../../quantization_distortion_results/full_20260805/report.md)
- [冻结 checkpoint 统计汇总](../../quantization_distortion_results/full_20260805/statistical_analysis/analysis_summary.json)
- [共同 teacher probe 实现](../../Network/run_qad_distortion_repair_probe.py)
- [Self-sensitivity probe 实现](../../Network/run_qad_self_distortion_probe.py)
- [统计分析实现](../../Network/qad_distortion_repair/statistics.py)
- [交互式 paper-readiness 报告](qad_distortion_paper_readiness_report.html)
