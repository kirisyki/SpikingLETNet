# Frozen-checkpoint Integer-LIF quantization distortion probe

Status: approved for execution on 2026-08-05.

## Question

For the frozen FP-QIF `SpikingLETNet_shallow_max` checkpoint, does independently
enabling historical W4 weight quantization and A4 operator-input quantization
produce a reproducible perturbation of the paired Integer-LIF output codes, and
does the combined W4A4 mode affect semantic-segmentation accuracy?

This is a frozen-checkpoint perturbation probe. It does not compare trained QAT
and QAD checkpoints and cannot establish cross-seed or cross-model generality.

## Frozen sources

- Checkpoint: `checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth`
- Configuration: `Network/configs/SpikingLETNet_shallow/1.3M.yaml`
- Dataset: all 8,478 entries in UDD `val_patches.txt`
- Validation batch size: 20
- Runtime time steps: `T_time=1`
- QIF range: `{0,...,8}` with `T_quant=8`
- Bias: FP32
- Quantized operators: all 71 `Conv2d`, `ConvTranspose2d`, and `Linear` layers

## Factorized modes

All modes retain the same QIF neurons and integer outputs.

| Mode | Weight fake quantization | Noninteger operator-input fake quantization |
|---|---|---|
| `fp_qif` | off | off |
| `w4_qif` | W4 | off |
| `a4_input_qif` | off | A4 |
| `w4a4_input_qif` | W4 | A4 |

The historical quantizer is reproduced exactly: ordinary signed codes use
`[-8,7]`, while a complete integer tensor in `[-8,8]` bypasses input
requantization and can preserve `+8`. Dynamic activation scales are per-tensor
and recomputed on every forward. No strict-INT4 sensitivity analysis is run.

Because FP-QIF already has discrete QIF outputs, these modes must not be labeled
as a true W32A32/W32A4 neuronal-output comparison.

## Preregistered measurements

At every QIF node executed by the model, compare outputs at the same tensor
position against `fp_qif`:

\[
D_l = \Pr[q_l^{variant} \ne q_l^{fp}], \qquad
E_l = \mathbb{E}|q_l^{variant} - q_l^{fp}|.
\]

The primary representation metric is the equal-weight mean of the active-layer
disagreement rates. Original UDD images, not overlapping patches, are the
statistical units. The 8,478 patches map to 35 source images; confidence
intervals use a 10,000-replicate source-cluster bootstrap.

Auxiliary measurements are signed code error, stage-level disagreement, mIoU,
pixel accuracy, per-class IoU, and prediction-pixel flip rate. Boundary, small
object, and selected-example analyses are explicitly excluded from this round.

## Decision rule

An observable W4A4 distortion is declared only if:

1. the 95% cluster-bootstrap CI for `D(W4A4)-D(W4)` has a lower bound above zero;
2. the 95% CI for `D(W4A4)-D(A4-input)` has a lower bound above zero; and
3. the 95% CI for `mIoU(W4A4)-mIoU(FP-QIF)` has an upper bound below zero.

A coupled/superadditive distortion is declared only if the observable rule holds
and the 95% CI for

\[
E(W4A4)-E(W4)-E(A4\text{-input})
\]

has a lower bound above zero.

If only code perturbation is observed without task degradation, the result is
reported as representation perturbation rather than task-relevant distortion.

## Isolation and artifacts

The experiment only adds files under `Network/quantization_distortion_probe/`,
`tests/quantization_distortion_probe/`, this protocol, the standalone runner,
and a new `quantization_distortion_results/` output directory. Historical model,
training, quantization, checkpoint, and deployment files remain unchanged.
