# W4A4 Quantization Training Comparison for SpikingLETNet_shallow_max

## Technical summary

The executed experiment supports the practical value of the existing
Quantization-Aware Distillation (QAD) model under the amended comparison
protocol. On the complete 8,478-sample UDD validation set, QAD achieved
`0.619547` mIoU, compared with `0.581270` for STE-QAT, `0.537915` for
LSQ-QAT, and `0.472837` for EWGS-QAT.

QAD exceeded STE-QAT by `3.8277` absolute mIoU points, passing the
pre-registered effectiveness threshold of `1.0` point. It exceeded the best
advanced baseline, LSQ-QAT, by `8.1633` points and therefore also passed the
pre-registered superiority threshold of `0.5` point.

These results are not, by themselves, a controlled proof that the algorithmic
difference caused the full accuracy gap. The historical QAD model was trained
for 127 epochs, whereas all three newly executed baselines were shortened to
16 epochs at the user's request. The evidence therefore supports the narrower
claim that the existing QAD model outperforms the three baselines under the
executed, shortened-budget protocol.

## QAD leads every W4A4 comparison baseline

The unified evaluation used the same checkpoint reconstruction, preprocessing,
validation loader, temporal setting, and confusion-matrix implementation for
all models. FP32 is included as a non-quantized reference.

**Unified validation mIoU by method**  
Scale: 0.00 to 0.70; each block represents approximately 0.02 mIoU.

```text
FP32       ██████████████████████████████████  0.678502
QAD W4A4   ███████████████████████████████     0.619547
STE W4A4   █████████████████████████████       0.581270
LSQ W4A4   ███████████████████████████         0.537915
EWGS W4A4  ████████████████████████            0.472837
```

| Method | Weight/activation precision | Best-checkpoint epoch | Unified mIoU | Gap vs. QAD |
|---|---:|---:|---:|---:|
| FP32 | 32/32 | 286 | 0.678502 | +5.8955 points |
| QAD | W4A4 | 119 | 0.619547 | — |
| STE-QAT | W4A4 | 14 | 0.581270 | -3.8277 points |
| LSQ-QAT | W4A4 | 10 | 0.537915 | -8.1633 points |
| EWGS-QAT | W4A4 | 8 | 0.472837 | -14.6711 points |

QAD retained more of the FP32 reference accuracy than any comparison method.
Its remaining gap to FP32 was `5.8955` absolute mIoU points.

## The QAD advantage appears in every semantic class

QAD exceeded each comparison baseline in all six classes. Against STE-QAT,
the smallest advantage was on roof (`1.7588` points) and the largest was on
background (`5.1747` points). This breadth reduces the likelihood that the
aggregate result is driven by only one class, although it does not remove the
training-budget confound.

| Method | Background | Facade | Road | Vegetation | Vehicle | Roof |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.544932 | 0.629335 | 0.639093 | 0.889751 | 0.545611 | 0.822292 |
| QAD W4A4 | 0.490170 | 0.576836 | 0.595894 | 0.874599 | 0.398562 | 0.781223 |
| STE-QAT W4A4 | 0.438424 | 0.532959 | 0.562018 | 0.839989 | 0.350598 | 0.763636 |
| LSQ-QAT W4A4 | 0.405871 | 0.465100 | 0.501322 | 0.864609 | 0.306708 | 0.683878 |
| EWGS-QAT W4A4 | 0.338681 | 0.358651 | 0.462159 | 0.828903 | 0.245401 | 0.603225 |

## Scope and metric definitions

- **Model:** `SpikingLETNet_shallow_max`.
- **Dataset and cohort:** all 8,478 samples in UDD `val_patches.txt`.
- **Task:** six-class semantic segmentation: background, facade, road,
  vegetation, vehicle, and roof.
- **Primary metric:** mean intersection over union (mIoU), calculated as the
  arithmetic mean of the six class IoUs from one accumulated confusion matrix.
- **Input and inference settings:** 400×400 input, validation batch size 20,
  and `T=1`.
- **Quantization scope:** signed per-tensor W4A4 quantization across all 71
  `Conv2d`, `Linear`, and `ConvTranspose2d` targets, with no first- or
  last-layer exemption.
- **Random seed:** 1234.
- **Baseline training budget:** 16 epochs after the approved runtime
  amendment.
- **Historical QAD budget:** 127 completed epochs; the existing QAD model was
  not retrained.
- **Learning-rate horizon:** 150 epochs for QAD and the shortened baselines,
  preserving the original polynomial schedule rather than compressing it into
  16 epochs.

An "absolute mIoU point" is `0.01` mIoU. For example, the QAD-minus-STE
difference of `0.0382768` mIoU is reported as `3.8277` points.

## Experimental design and method specifications

### Existing QAD reference

The historical QAD student used the repository's existing W4A4 straight-through
quantizer. Training combined segmentation cross-entropy with intermediate
feature distillation from a frozen FP32 teacher. The distillation term used a
weight of `0.1` across six intermediate feature tensors. Per the user's
instruction, this model was evaluated but not retrained.

### STE-QAT baseline

STE-QAT reused the existing W4A4 quantizer and its historical integer-activation
bypass behavior, but trained only on segmentation cross-entropy. It did not use
a teacher, feature matching, or any distillation loss. This is the primary
ablation for isolating the practical contribution of QAD relative to otherwise
similar W4A4 training.

### LSQ-QAT baseline

LSQ-QAT used signed per-tensor learned step sizes for both weights and
activations. Step sizes were trained jointly with the model, with the standard
LSQ gradient scaling based on tensor size and positive quantization range.
Quantizer step sizes used zero weight decay.

### EWGS-QAT baseline

EWGS-QAT used learned clipping bounds and output scales with element-wise
gradient scaling. Starting after the first epoch, backward scaling factors were
estimated once per epoch using 10 batches and up to 50 Hutchinson
Hessian-vector iterations per batch. `PA3.conv` remained part of the 71-layer
quantization scope but was recorded as inactive because the current model
forward path does not execute that layer.

### Shared optimization and selection

All new baselines used task loss only, Adam, model learning rate `1e-3`,
weight decay `1e-4`, seed 1234, the same UDD training augmentation, and the
same polynomial learning-rate schedule. EWGS quantizer-specific parameters
used learning rate `1e-5` and zero weight decay. Each method's reportable
checkpoint was the highest-mIoU checkpoint within its 16-epoch run. No
validation-driven hyperparameter search was performed.

## Validation and robustness checks

- The historical QAD result reproduced at `0.619547317`, only
  `0.000000317` from the registered reference value of `0.619547`, within the
  required `1e-5` tolerance.
- Model conversion confirmed exactly 71 quantized layers for STE, LSQ, and
  EWGS.
- Twenty-two comparison-specific tests passed, covering quantization ranges,
  gradients, temporal dimensions, non-mutating conversion, parameter-group
  isolation, protocol integrity, checkpoint/RNG restoration, and the
  127-to-16 epoch amendment.
- SHA-256 checks confirmed that the existing QAD scripts, historical W4A4
  quantizer, target model, and previous evaluation script were unchanged.
- The historical integer-activation bypass was audited rather than silently
  corrected. QAD produced 22,045 bypass calls, including 5,188 calls
  containing `+8`; STE produced 22,874 bypass calls, including 4,543 calls
  containing `+8`.
- All five checkpoints were reconstructed independently and evaluated on all
  424 validation batches.

## Limitations prevent a causal algorithm-only conclusion

1. **Training budgets differ materially.** QAD received 127 epochs, while the
   comparison baselines received 16. This is the most important limitation and
   can bias the result in favor of QAD.
2. **Only one seed was executed.** The experiment does not estimate
   between-seed variance or provide confidence intervals.
3. **Checkpoint selection and reporting use the validation set.** The best
   checkpoint was selected by validation mIoU and then re-evaluated on the
   same validation population. This is consistent across methods but is not
   an independent test-set estimate.
4. **Algorithm-specific implementations are not symmetric.** STE reuses the
   historical project quantizer, while LSQ and EWGS are isolated clean-room
   implementations. Differences may reflect implementation details in
   addition to the named algorithms.
5. **The result is descriptive and comparative, not causal.** The registered
   thresholds classify observed gaps; they do not prove that distillation
   alone caused those gaps.

## Recommended next steps

1. Train STE-QAT, LSQ-QAT, and EWGS-QAT to the same 127-epoch budget as the
   existing QAD model while preserving the 150-epoch learning-rate horizon.
2. Repeat the matched-budget comparison with at least three pre-registered
   seeds and report mean, standard deviation, and seed-level results.
3. Freeze model selection before evaluating once on an independent test split.
4. Preserve the current 16-epoch results as a runtime-constrained pilot, not as
   the sole paper-level proof of QAD's algorithmic superiority.

## Further questions

- Does QAD maintain its advantage when all baselines receive 127 epochs?
- How much of the QAD gain comes from intermediate feature distillation versus
  simply longer optimization?
- Are the relative rankings stable across seeds and an independent test set?
- Would a matched wall-clock or matched-compute comparison change the
  interpretation of the Hessian-intensive EWGS baseline?

## Reproducibility artifacts

- Machine-readable results: `comparison.json`
- Tabular results: `comparison.csv`
- Chinese summary report: `report.md`
- Protocol and checkpoint roots:
  `../../quantization_comparison_checkpoint/udd/seed1234/`
- Original protocol:
  `../../docs/w4a4_quantization_training_comparison_plan.md`
- Epoch-budget amendment:
  `../../docs/w4a4_quantization_training_comparison_execution_amendment.md`
