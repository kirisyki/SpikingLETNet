# Frozen-checkpoint Integer-LIF quantization distortion probe

## Preregistered outcome

- Observable W4A4 distortion: **True**
- Coupled/superadditive distortion: **False**

The conclusion is restricted to the frozen SpikingLETNet-QIF checkpoint and the 35 UDD validation source images. It is not a cross-seed or cross-model generalization claim.

## Representation results

| Mode | Macro code disagreement | 95% cluster CI | Macro code MAE | Signed code error |
|---|---:|---:|---:|---:|
| w4_qif | 0.108635 | [0.105637, 0.111788] | 0.127943 | -0.002764 |
| a4_input_qif | 0.087091 | [0.082803, 0.091219] | 0.099617 | 0.003699 |
| w4a4_input_qif | 0.113808 | [0.110680, 0.116984] | 0.135103 | -0.001386 |

## Task results

| Mode | mIoU | Pixel accuracy | Prediction flip vs FP-QIF |
|---|---:|---:|---:|
| fp_qif | 0.678502 | 0.841955 | 0.000000 |
| w4_qif | 0.405185 | 0.639980 | 0.333509 |
| a4_input_qif | 0.569789 | 0.770280 | 0.179692 |
| w4a4_input_qif | 0.345938 | 0.591629 | 0.386974 |

W4A4-input-QIF minus FP-QIF mIoU: **-0.332564**, 95% cluster CI [-0.366402, -0.296913].

## Preregistered contrasts

| Contrast | Mean | 95% cluster CI | Positive sources |
|---|---:|---:|---:|
| w4a4_minus_w4_disagreement | +0.005173 | [+0.004537, +0.005813] | 35/35 |
| w4a4_minus_a4input_disagreement | +0.026717 | [+0.025134, +0.028400] | 35/35 |
| mae_interaction_w4a4_minus_w4_minus_a4input | -0.092458 | [-0.097562, -0.087268] | 0/35 |

## Protocol boundary

- Validation patches: 8478
- Source-image clusters: 35
- Declared QIF nodes: 84
- Active QIF nodes: 78
- Runtime time steps: 1
- Validation batch size: 20
- All configurations retain the same QIF code range `{0,...,8}`.
- `A4-input` refers to historical operator-input fake quantization, not replacement of QIF by a continuous A32 neuron.
- This is a frozen-checkpoint perturbation probe, not a comparison of trained QAT and QAD models.
