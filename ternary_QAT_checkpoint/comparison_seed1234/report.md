# SpikingLETNet shallow max precision comparison

All models were re-evaluated with the same UDD validation loader, preprocessing, T, batch size, and confusion-matrix implementation.

| Model | Weight bits | Activation bits | mIoU | Weight zero fraction |
|---|---:|---:|---:|---:|
| FP32 | 32 | 32 | 0.678502 | n/a |
| W4/A4 | 4 | 4 | 0.619547 | n/a |
| W1.58/A4 | 1.585 | 4 | 0.591090 | 86.2041% |
| W1.58/A1.58 | 1.585 | 1.585 | 0.223259 | 80.7651% |

## Pre-registered decision

- W4/A4 minus W1.58/A4: 2.8457 mIoU points.
- Threshold: 2.0 points.
- Verdict: `supports_w4_accuracy_value`.
- Interpretation: W4/A4 exceeds W1.58/A4 by at least 2 absolute mIoU points.
- W1.58 activation penalty: 36.7831 mIoU points.

## Per-class IoU

| Model | background | facade | road | vegetation | vehicle | roof |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.544932 | 0.629335 | 0.639093 | 0.889751 | 0.545611 | 0.822292 |
| W4/A4 | 0.490170 | 0.576836 | 0.595894 | 0.874599 | 0.398562 | 0.781223 |
| W1.58/A4 | 0.454613 | 0.561850 | 0.539457 | 0.858834 | 0.365131 | 0.766655 |
| W1.58/A1.58 | 0.191041 | 0.121103 | 0.160260 | 0.529898 | 0.000032 | 0.337220 |
