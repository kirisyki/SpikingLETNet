# First image-input A4 inference ablation

Only the normalized image input to `init_conv.0.conv` is changed. The first convolution weight remains W4 and every later QLayer input remains A4.

This is a paired inference sensitivity test on existing checkpoints, not training with the alternative policy. Historical QAD observed 127 training epochs; STE-QAT used the shortened 16-epoch budget.

| Method | First image input | mIoU | Delta vs input A4 (points) |
|---|---|---:|---:|
| QAD | A4 | 0.620588 | +0.0000 |
| QAD | FP | 0.555916 | -6.4672 |
| STE | A4 | 0.610170 | +0.0000 |
| STE | FP | 0.559694 | -5.0475 |

## Decision boundary

A within-checkpoint absolute change of at least 0.5 mIoU points is treated as material enough to justify matched retraining. Smaller changes remain descriptive because this toggle was not present during training.

The QAD-versus-STE gap must not be interpreted causally here because the training budgets differ.
