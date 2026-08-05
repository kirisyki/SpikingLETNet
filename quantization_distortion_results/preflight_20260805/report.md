# Frozen-checkpoint Integer-LIF quantization distortion probe

## Preregistered outcome

- Observable W4A4 distortion: **True**
- Coupled/superadditive distortion: **False**

The conclusion is restricted to the frozen SpikingLETNet-QIF checkpoint and the 35 UDD validation source images. It is not a cross-seed or cross-model generalization claim.

## Representation results

| Mode | Macro code disagreement | 95% cluster CI | Macro code MAE | Signed code error |
|---|---:|---:|---:|---:|
| w4_qif | 0.120719 | [0.120719, 0.120719] | 0.144826 | -0.016360 |
| a4_input_qif | 0.101099 | [0.101099, 0.101099] | 0.116822 | 0.009729 |
| w4a4_input_qif | 0.121131 | [0.121131, 0.121131] | 0.145206 | -0.020207 |

## Task results

| Mode | mIoU | Pixel accuracy | Prediction flip vs FP-QIF |
|---|---:|---:|---:|
| fp_qif | 0.757116 | 0.882825 | 0.000000 |
| w4_qif | 0.411826 | 0.558058 | 0.431305 |
| a4_input_qif | 0.633545 | 0.802190 | 0.165321 |
| w4a4_input_qif | 0.400471 | 0.573349 | 0.411837 |

W4A4-input-QIF minus FP-QIF mIoU: **-0.356644**, 95% cluster CI [-0.356644, -0.356644].

## Preregistered contrasts

| Contrast | Mean | 95% cluster CI | Positive sources |
|---|---:|---:|---:|
| w4a4_minus_w4_disagreement | +0.000412 | [+0.000412, +0.000412] | 1/1 |
| w4a4_minus_a4input_disagreement | +0.020033 | [+0.020033, +0.020033] | 1/1 |
| mae_interaction_w4a4_minus_w4_minus_a4input | -0.116443 | [-0.116443, -0.116443] | 0/1 |

## Protocol boundary

- Validation patches: 40
- Source-image clusters: 1
- Declared QIF nodes: 84
- Active QIF nodes: 78
- Runtime time steps: 1
- Validation batch size: 20
- All configurations retain the same QIF code range `{0,...,8}`.
- `A4-input` refers to historical operator-input fake quantization, not replacement of QIF by a continuous A32 neuron.
- This is a frozen-checkpoint perturbation probe, not a comparison of trained QAT and QAD models.
