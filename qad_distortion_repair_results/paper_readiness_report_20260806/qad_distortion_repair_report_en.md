# Does QAD Repair W4A4 Integer-LIF Quantization Distortion?

**Technical evidence report — English**  
**Analysis date:** 6 August 2026  
**Current verdict:** The experiments support a narrow feature-alignment interpretation of QAD, but they do **not** support the stronger claim that QAD globally repairs Integer-LIF code distortion or causally outperforms a fairly matched STE-QAT baseline.

## Technical summary

Quantization distortion is clearly observable in this model family. Applying W4A4 perturbations to the frozen FP checkpoint produces substantial representation and task changes. In the trained-checkpoint comparison, QAD also reaches **61.95% mIoU**, compared with **58.13%** for the retained 16-epoch STE-QAT checkpoint.

The deeper analysis changes the mechanism interpretation, however:

- Against the same FP-QIF teacher, QAD reduces normalized MSE at the six explicitly distilled features by **31.85%** relative to STE-QAT, and reduces prediction disagreement with the teacher by **1.80 percentage points**.
- This feature result is highly concentrated: the final decoder/pre-logit feature, F6, accounts for **98.19% of the reduction in raw six-feature KD MSE**. Excluding F6, the normalized-MSE reduction is **10.42%**.
- QAD does **not** improve global Integer-LIF code fidelity. Its macro code disagreement is **11.64%**, versus **10.08%** for STE-QAT; the QAD-minus-STE difference is **+1.56 percentage points**, 95% source-cluster bootstrap CI **[+1.53, +1.59]**. The same direction appears for all 35 source images and all seven network stages.
- The within-checkpoint FP-shadow sensitivity test reaches the same conclusion: QAD is more sensitive, not less sensitive, to switching W4A4 quantization off. This is consistent with a model adapted to its quantized forward path, not with a globally FP-like representation.
- Across the 35 original validation images, neither QIF-code repair nor distilled-feature repair predicts the QAD-versus-STE source-level mIoU gain. Therefore the present analysis does not establish a causal “distortion repair → accuracy recovery” chain.

The defensible interpretation is: **QAD learns task-useful W4A4 representations that are better aligned with the teacher at the directly distilled multi-scale features—especially F6—without restoring global layer-wise Integer-LIF codes to the FP teacher.** This remains descriptive because the QAD run and retained STE-QAT run have unequal training budgets and only one checkpoint/seed pair is compared.

## 1. Question and decision boundary

The analysis asks two distinct questions:

1. Is W4A4 quantization distortion observable in the Integer-LIF segmentation model?
2. Does the existing evidence show that QAD repairs that distortion, and does this repair explain its task accuracy?

The first question is supported. The second is supported only for the explicitly distilled feature targets, not for global QIF-code fidelity or for a causal explanation of mIoU.

This distinction matters for the paper. The current method is high-precision-teacher feature distillation combined with STE-based QAT; it contains no loss that directly penalizes the measured global QIF code-disagreement statistic. The mechanism claim should therefore follow the actual optimization target and observed evidence.

## 2. Experimental scope and metric definitions

### 2.1 Evaluation cohort

- Dataset: full UDD6 validation split.
- Evaluation size: **8,478 patches** grouped into **35 original source images**.
- Model representation: **78 active QIF nodes**, six recorded distillation features, runtime `T = 1`, QIF code range `[0, 8]`.
- Uncertainty: **10,000 source-cluster bootstrap replicates**. The original source image, rather than an individual overlapping patch, is the statistical unit.
- Checkpoints: common FP-QIF teacher; historical QAD best checkpoint; retained 16-epoch STE-QAT best checkpoint.
- Known confound: the QAD training run covers 127 observed epochs and its best checkpoint is at epoch 119; the retained STE-QAT run covers 16 epochs and its best checkpoint is at epoch 14.

### 2.2 Reference designs

**Common-teacher audit.** QAD and STE-QAT are both compared with the same separately trained FP-QIF teacher. This asks which student is closer to the teacher at QIF outputs, KD features, and final predictions.

**Within-checkpoint FP-shadow audit.** Each trained checkpoint is deep-copied with its master weights and BN state fixed, while QLayer weight/input fake quantization is disabled. This asks how sensitive each learned solution is to its own W4A4 operators. The FP shadow is a local counterfactual and **not** a separately trained FP task baseline.

**Frozen-checkpoint perturbation audit.** The original FP checkpoint is evaluated under W32A32, W4A32, W32A4, and W4A4 perturbations. This establishes that distortion exists before quantization-aware retraining, but it cannot establish that QAD repairs the distortion.

### 2.3 Metrics

- **QIF code disagreement:** fraction of Integer-LIF output elements whose discrete codes differ from the selected reference. Lower is closer.
- **QIF MAE:** mean absolute difference between student and reference QIF codes. Lower is closer.
- **Macro statistic:** source-balanced and layer-balanced aggregation, preventing large feature maps or heavily patched source images from dominating.
- **Micro statistic:** aggregation over all recorded elements, allowing large tensors to contribute proportionally more.
- **Teacher-referenced normalized MSE:** feature MSE normalized by the teacher feature energy. It is preferred to raw MSE for comparing features with different scales.
- **Cosine distance:** difference in feature direction, complementary to MSE.
- **Prediction flip rate:** fraction of output pixels whose predicted class differs from the reference model.
- **Source-balanced mIoU:** mIoU calculated per original source image and then averaged. It differs from global confusion-matrix mIoU.

## 3. Quantization distortion exists, but coupled superadditivity is not established

The frozen FP checkpoint is substantially disrupted by W4A4 perturbation:

| Frozen-checkpoint result | FP/W32A32 | W4A4 | Change |
|---|---:|---:|---:|
| Global mIoU | 67.85% | 34.59% | −33.26 pp |
| Prediction flip rate vs FP | 0% | 38.70% | +38.70 pp |
| Macro QIF code disagreement | 0% | 11.38% | +11.38 pp |

These results establish a repeatable W4A4 failure mode in dense prediction: discrete internal codes and semantic predictions change substantially when a high-precision checkpoint is exposed to joint low-bit operators.

They do **not** establish that the weight and activation effects are superadditive. The existing W4A32, W32A4, and W4A4 decomposition indicates observable joint distortion, but not a reliable interaction larger than the sum of the separate effects. The paper should use “joint” or “coupled deployment constraints” descriptively, rather than claim a proven superadditive degradation mechanism.

## 4. QAD improves the directly distilled features

Against the common FP-QIF teacher, QAD is closer than STE-QAT on the loss family that QAD actually optimizes.

| Teacher-referenced feature result | QAD | STE-QAT | QAD effect |
|---|---:|---:|---:|
| Mean normalized MSE, F1–F6 | 0.6946 | 1.0193 | 31.85% lower, 95% CI [30.77%, 32.95%] |
| Mean cosine distance, F1–F6 | 0.3733 | 0.3930 | 5.02% lower |
| Mean raw MSE, F1–F6 | 0.3283 | 1.5318 | 78.57% lower |
| Mean normalized MSE, F1–F5 only | 0.7962 | 0.8888 | 10.42% lower |

The normalized-MSE improvement appears in all six distillation features:

| Feature | Relative normalized-MSE reduction for QAD | 95% CI | Interpretation |
|---|---:|---:|---|
| F1 | 4.30% | [3.30%, 5.26%] | Small improvement |
| F2 | 9.60% | [8.89%, 10.34%] | Moderate improvement |
| F3 | 17.24% | [16.08%, 18.39%] | Clear improvement |
| F4 | 13.12% | [11.26%, 15.00%] | MSE improves, but cosine distance worsens |
| F5 | 7.09% | [5.53%, 8.74%] | Moderate improvement |
| F6 | 88.85% | [87.64%, 90.00%] | Dominant final decoder/pre-logit effect |

The effect is therefore real but not uniform. F6 accounts for **98.19% of the reduction in raw six-feature KD MSE**. Moreover, F4's cosine distance is worse for QAD by 0.0212, 95% CI **[+0.0149, +0.0286]**, even though its normalized MSE improves. “Feature alignment” must be tied to a named metric; it should not be presented as a universal geometric improvement.

The strongest mechanism hypothesis supported by current data is that QAD mainly regularizes the late, task-facing decoder representation, while producing smaller scale-normalized improvements at earlier KD targets.

## 5. QAD does not repair global Integer-LIF code fidelity

The global QIF statistics point in the opposite direction from a code-repair claim.

| Common FP-teacher QIF result | QAD | STE-QAT | QAD − STE | 95% CI for difference |
|---|---:|---:|---:|---:|
| Macro code disagreement | 11.64% | 10.08% | +1.56 pp | [+1.53, +1.59] pp |
| Micro code disagreement | 14.96% | 12.07% | +2.89 pp | [+2.84, +2.94] pp |
| Macro code MAE | 0.1446 | 0.1188 | +0.0258 | [+0.0253, +0.0264] |
| Absolute signed-code bias | 0.0413 | 0.0153 | +0.0260 | [+0.0251, +0.0269] |
| Absolute zero-rate change | 1.96% | 1.11% | +0.85 pp | [+0.82, +0.89] pp |

QAD has higher macro disagreement in **35/35 source images**; the exact two-sided sign-test value is `p = 5.82 × 10⁻¹¹`. The direction is also consistent at stage level:

| Stage | QAD − STE macro disagreement |
|---|---:|
| Stem | +3.20 pp |
| Encoder 1 | +3.51 pp |
| Encoder 2 | +0.71 pp |
| Encoder 3 | +1.21 pp |
| Decoder 4 | +0.46 pp |
| Decoder 5 | +0.88 pp |
| Decoder 6 | +2.26 pp |

At individual-node level, QAD has lower mean disagreement in 11 of 78 QIF nodes, higher disagreement in 54, and equality in 13. This rules out an explanation based on one anomalous layer or a small number of samples.

The result does not mean QAD is ineffective. It means **FP-like discrete codes are not the representation property through which this checkpoint expresses its benefit**. Multiple internal code configurations can support useful downstream features and predictions.

## 6. The FP-shadow test indicates quantization adaptation, not robustness to removing quantization

The same-master-weight sensitivity test independently rejects the global-repair interpretation.

| Own FP-shadow → W4A4 result | QAD | STE-QAT | QAD − STE |
|---|---:|---:|---:|
| Macro QIF disagreement | 9.31% | 7.75% | +1.56 pp, 95% CI [+1.46, +1.67] |
| Micro QIF disagreement | 11.08% | 9.33% | +1.75 pp |
| Macro QIF MAE | 0.1155 | 0.0913 | +0.0242 |
| Feature normalized MSE | 0.5615 | 0.5129 | QAD is 9.48% higher |
| Feature cosine distance | 0.2788 | 0.2576 | QAD is higher |

Turning fake quantization off also lowers task performance rather than improving it:

- QAD FP shadow: **52.43% mIoU**; QAD W4A4: **61.95%**, a source-balanced W4A4-minus-shadow gain of **+8.44 pp**, 95% CI **[+6.70, +10.21]**.
- STE FP shadow: **56.52% mIoU**; STE W4A4: **58.13%**, a source-balanced gain of **+1.26 pp**, 95% CI **[+0.14, +2.43]**.

This should not be interpreted as “quantization inherently improves accuracy,” because the FP shadows were not trained FP models. It shows that the learned weights, BN statistics, scales, and nonlinear operating points are co-adapted to the quantized forward path. QAD is more strongly co-adapted than the short-budget STE checkpoint.

## 7. Task accuracy improves relative to STE16, but the causal chain remains open

| Full-validation task result | FP teacher | QAD | STE-QAT (16 epochs) |
|---|---:|---:|---:|
| Global mIoU | 67.85% | 61.95% | 58.13% |
| Pixel accuracy | 84.20% | 81.18% | 79.13% |
| Prediction flip rate vs FP teacher | — | 14.90% | 16.69% |

QAD exceeds the retained STE checkpoint by **3.83 mIoU points** and reduces teacher-relative prediction flips by **1.80 points**, equivalent to a **10.77% relative reduction** in flips. On source-balanced mIoU, QAD is higher for 29 of 35 source images, with a mean difference of **+3.82 points**, 95% CI **[+2.67, +5.00]**.

These are valid checkpoint-level descriptive results. They are not a fair causal estimate of the QAD method because of the 127-versus-16-epoch training mismatch.

The attempted “distortion → repair → performance” closure is also negative:

| Source-level association, n = 35 | Spearman ρ | p-value | 95% cluster-bootstrap CI |
|---|---:|---:|---:|
| QIF-code repair vs mIoU gain | −0.004 | 0.983 | [−0.318, +0.329] |
| KD-feature repair vs mIoU gain | −0.184 | 0.289 | [−0.472, +0.139] |

The confidence intervals include meaningful positive and negative relationships. The current 35-source study therefore provides no evidence that sources with more measured feature repair obtain more mIoU gain. Aggregate feature alignment and aggregate task performance can coexist without establishing source-level mediation or causality.

## 8. Evidence audit for manuscript claims

| Candidate claim | Status | Reason |
|---|---|---|
| W4A4 causes observable Integer-LIF representation and segmentation distortion in a frozen FP model. | **Supported** | Large, repeatable code and task changes on the complete validation split. |
| The QAD checkpoint aligns the six directly distilled features better than STE16 under normalized MSE. | **Supported descriptively** | All six improve; effect is dominated by F6. |
| QAD produces fewer final prediction disagreements with the FP teacher than STE16. | **Supported descriptively** | 14.90% vs 16.69% flip rate. |
| QAD repairs or suppresses global Integer-LIF code distortion. | **Contradicted by current measurements** | QIF disagreement and MAE are higher under both reference designs. |
| QAD resolves a proven superadditive weight–activation degradation. | **Not supported** | Joint distortion exists, but superadditive interaction is not established. |
| Feature repair causes the mIoU improvement. | **Not supported** | No source-level association; no targeted ablation or mediation experiment. |
| QAD fairly and stably outperforms conventional QAT. | **Not yet supported** | Unequal training budgets, one checkpoint/seed, no archived matched-budget STE result. |
| QAD is a distortion-specific method. | **Not supported by the current design** | The implemented loss is generic multi-scale feature KD plus STE-QAT, with no direct QIF-distortion term. |

### Manuscript-safe wording now

> In the examined W4A4 checkpoint, quantization-aware feature distillation improves alignment with the FP teacher at the explicitly distilled multi-scale features, especially the final decoder/pre-logit feature, and is accompanied by fewer output prediction disagreements than the retained short-budget STE-QAT checkpoint.

The comparison must be labeled as diagnostic or preliminary until training budgets and seeds are matched.

### Wording to avoid now

- “QAD repairs global Integer-LIF quantization distortion.”
- “QAD resolves coupled or superadditive W4A4 degradation.”
- “The feature-alignment recovery explains or causes the mIoU gain.”
- “QAD substantially outperforms conventional QAT” without the training-budget qualifier.

## 9. Limitations and robustness checks

1. **Training-budget mismatch is the dominant limitation.** QAD and STE16 are different training regimes, so their difference combines objective, schedule, and optimization maturity.
2. **Only one checkpoint/seed pair is audited.** Source-cluster bootstrap quantifies validation-set uncertainty; it does not replace independent training seeds.
3. **Generic KD is absent.** The current evidence cannot separate the effect of ordinary teacher feature supervision from any specifically quantization-aware component.
4. **F6 dominates the raw KD result.** This is a useful localization finding, but it also means the six-feature aggregate can overstate broad representation repair.
5. **Metric dependence is visible.** F4 improves in normalized MSE but worsens in cosine distance; raw MSE is sensitive to feature scale.
6. **The FP shadow is not an accuracy baseline.** It is a local operator-sensitivity counterfactual at fixed trained weights.
7. **No archived matched-budget STE result is available.** Unarchived observations must not be used as paper evidence; the experiment must be rerun under the frozen protocol.
8. **The analysis uses UDD6 and one architecture.** Generalization to another dataset or SNN segmentation model remains untested.

Robustness work already completed includes full-split inference, exact checkpoint hashing, source-clustered uncertainty, macro and micro metrics, common-teacher and within-checkpoint reference designs, per-stage localization, and raw-row/count consistency checks. The associated statistical tests pass (`8 passed`).

## 10. Minimum experiments that would make the mechanism story publishable

Run these in order:

1. **Matched-budget STE-QAT, at least three seeds.** Use the same initialization/pretraining source, 127-epoch horizon, optimizer, scheduler, augmentation, batch size, resolution, checkpoint selection, and evaluation protocol as QAD.
2. **Matched generic-KD baseline.** Under the same FP teacher and W4A4 student, compare task-only STE-QAT, logit KD + STE-QAT, the current multi-scale feature KD + STE-QAT, and the full named method.
3. **F6 localization ablation.** Compare no feature KD, F6-only KD, F1–F5-only KD, and F1–F6 KD. Pre-register F6 normalized MSE, prediction flip rate, and mIoU as the primary endpoints.
4. **Repeat the diagnostic across seeds.** Recompute the common-teacher feature and QIF metrics for every seed; do not infer method-level behavior from one selected checkpoint.
5. **Add one generalization axis.** Repeat the matched comparison on a second dataset or second deployable SNN segmentation architecture.

If the matched experiment confirms better F6 alignment, fewer output flips, and higher mIoU—but still not lower global QIF disagreement—the paper should frame QAD as **task-relevant feature preservation under W4A4 training**, not global integer-code repair.

## 11. Conclusion

The current experiments are sufficient to say that W4A4 distortion exists and that the examined QAD checkpoint preserves the teacher's explicitly distilled features and final decisions better than the retained short-budget STE checkpoint. They are **not** sufficient to say that QAD globally repairs Integer-LIF code distortion, that the feature change causes the accuracy gain, or that QAD fairly outperforms QAT.

For the paper, the most accurate present-tense conclusion is:

> QAD appears to improve task-facing, teacher-aligned feature representations—dominated by the final decoder/pre-logit target—while allowing internal discrete QIF codes to depart further from the FP teacher. Its role is better described as quantized representation adaptation and task-relevant feature preservation than as global code-distortion repair.

## 12. Reproducibility and source artifacts

- [Common-teacher analysis summary](../qad_vs_ste16_full_20260806/analysis_summary.json)
- [Common-teacher manifest](../qad_vs_ste16_full_20260806/manifest.json)
- [Per-feature statistics](../qad_vs_ste16_full_20260806/feature_statistics.csv)
- [Per-layer statistics](../qad_vs_ste16_full_20260806/layer_statistics.csv)
- [Per-stage statistics](../qad_vs_ste16_full_20260806/stage_statistics.csv)
- [Within-checkpoint sensitivity summary](../self_sensitivity_full_20260806/analysis_summary.json)
- [Within-checkpoint sensitivity manifest](../self_sensitivity_full_20260806/manifest.json)
- [Frozen-checkpoint distortion report](../../quantization_distortion_results/full_20260805/report.md)
- [Frozen-checkpoint statistical summary](../../quantization_distortion_results/full_20260805/statistical_analysis/analysis_summary.json)
- [Probe implementation](../../Network/run_qad_distortion_repair_probe.py)
- [Self-sensitivity implementation](../../Network/run_qad_self_distortion_probe.py)
- [Statistical analysis implementation](../../Network/qad_distortion_repair/statistics.py)
- [Interactive paper-readiness report](qad_distortion_paper_readiness_report.html)
