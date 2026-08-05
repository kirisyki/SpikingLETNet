# Quantization experiment handoff summary

> Purpose: give the next Agent enough verified context to start a new quantization task without re-running or accidentally modifying earlier experiments. Updated: 2026-08-02.

## 1. Project scope and constraints

- Model: `SpikingLETNet_shallow_max`; dataset: UDD six-class semantic segmentation.
- Existing QAD checkpoints must be reused and must **not** be retrained.
- Preserve the current architecture and QAD implementation. New quantization algorithms, training entry points, evaluation code, tests, and outputs must be placed in new files/directories and remain low-coupling.
- Do not overwrite `QAT_checkpoint/` or earlier reports. Inspect the working tree before editing.
- Before a materially new experiment, write a Markdown plan for review; execute only after approval.
- Report measured facts separately from interpretations, include exact metrics and limitations, and avoid causal claims that the protocol cannot support.

## 2. Shared evaluation protocol

- FP32 source checkpoint: `checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth`.
- Validation population: all 8,478 entries in UDD `val_patches.txt` (424 batches), input `400x400`, validation batch size 20.
- Runtime simulation steps: `T_time=1`; seed: 1234; primary metric: six-class mIoU from one accumulated confusion matrix.
- Quantization covers all 71 `Conv2d`, `Linear`, and `ConvTranspose2d` targets, with no first/last-layer exemption.
- FP32 reference mIoU: **0.678502**.

## 3. W4A4 training-method comparison

| Method | Training | Unified mIoU | Gap vs. QAD |
|---|---|---:|---:|
| QAD W4A4 | Task loss + intermediate feature distillation | **0.619547** | -- |
| STE-QAT W4A4 | Same historical W4A4 quantizer, task loss only | 0.581270 | -3.8277 points |
| LSQ-QAT W4A4 | Learned signed per-tensor step sizes | 0.537915 | -8.1633 points |
| EWGS-QAT W4A4 | Learned clipping/scales and element-wise gradient scaling | 0.472837 | -14.6711 points |

Observed result: QAD ranked first among the W4A4 methods and beat STE-QAT in every class. This is a runtime-constrained pilot, not an algorithm-only causal proof: the historical QAD run completed 127 epochs, while STE/LSQ/EWGS were shortened to 16 epochs at the user's request. All used a 150-epoch polynomial learning-rate horizon, and only one seed was evaluated.

The historical W4A4 activation quantizer normally uses signed integer codes `[-8, 7]`, but it bypasses requantization when the complete tensor is integral and lies in `[-8, 8]`. Consequently, `+8` can survive on QuantNeuron outputs; this behavior was audited and deliberately preserved.

Primary report: `quantization_comparison_results/seed1234/report_en.md`  
Machine-readable result: `quantization_comparison_results/seed1234/comparison.json`

## 4. Quantization-precision comparison

| Precision | Unified mIoU | Ternary-weight zero fraction |
|---|---:|---:|
| W4/A4 QAD | **0.619547** | n/a |
| W1.58/A4 QAT | 0.591090 | 86.2041% |
| W1.58/A1.58 QAT | 0.223259 | 80.7651% |

Key observations:

- W4/A4 exceeded W1.58/A4 by **2.8457 absolute mIoU points**, supporting the pre-registered W4 accuracy-value threshold of 2.0 points.
- Replacing A4 with nominal A1.58 caused a further **36.7831-point** mIoU loss.
- Ternary weights use codes `{-1, 0, +1}` with detached per-output-channel absolute-mean scales.
- Nominal A1.58 activations use codes `{-1, 0, +1}` with a dynamic per-tensor scale `mean(abs(x))`.

Primary report: `ternary_QAT_checkpoint/comparison_seed1234/report.md`  
Interactive report: `ternary_QAT_checkpoint/analysis_report/ternary_experiment_report.html`  
Machine-readable result: `ternary_QAT_checkpoint/comparison_seed1234/comparison.json`

## 5. QuantNeuron time-step and range semantics

Two unrelated values are both named `T`:

- Training/evaluation uses `T_time=1`, so the input time axis contains only `t=0` and temporal averaging is effectively an identity operation.
- The model YAML sets every `QIFNode.T` to `T_quant=8`. In this implementation it is a firing-code upper bound, not the runtime sequence length:

  `QIF(v) = round(clamp(v, 0, 8))`, giving output codes `{0,1,...,8}`.

All used QIF nodes have `bin=False`, so the optional binary temporal expansion is inactive.

Important caveat for future precision experiments: a raw QuantNeuron output is nonnegative. The current signed ternary activation quantizer does no centering or zero-point shift, so a directly QIF-fed tensor can only produce ternary codes `{0,+1}`; `-1` is unreachable. On those boundaries, nominal A1.58 activation therefore collapses to an effective binary activation. Do not describe it as a fully utilized three-state activation without disclosing this fact.

## 6. Implementation boundaries and useful entry points

- Existing QAD code: `Network/QAT_snn_STE.py`, `Network/quantization/int4_selfbuild.py` -- read/reuse only.
- Independent W4A4 baselines: `Network/train_quantization_baseline.py` and `Network/quantization_comparison/`.
- Independent ternary implementation: `Network/QAT_snn_ternary.py`, `Network/quantization/ternary_qat.py`.
- Unified evaluators: `Network/evaluate_quantization_comparison.py`, `Network/evaluate_precision_comparison.py`.
- Plans and protocol amendment: `docs/w4a4_quantization_training_comparison_plan.md`, `docs/w4a4_quantization_training_comparison_execution_amendment.md`.

## 7. Expected style for the next Agent

1. Start with the answer or proposed decision, then give only the evidence needed to audit it.
2. Use exact checkpoint/result paths and a compact metrics table.
3. State training budget, seed count, evaluation cohort, time steps, and quantization ranges explicitly.
4. Keep “observed comparison” distinct from “causal proof”; put material caveats next to the conclusion.
5. Preserve low coupling: add new files, tests, manifests, and output roots instead of modifying existing QAD or model code.
6. For a new nonnegative low-bit activation experiment, consider an explicitly unsigned three-level codebook such as `{0,1,2}`/`{0,s,2s}` in a new quantizer file; do not silently reinterpret the existing signed A1.58 result.
