# First image-input quantization ablation

## Question

Does historical A4 fake quantization of the normalized RGB image materially
affect the existing QAD and STE-QAT checkpoints?

## Controlled change

- Control: `init_conv.0.conv` uses W4 weights and A4 normalized-image input.
- Variant: `init_conv.0.conv` keeps W4 weights but consumes the FP32 normalized
  image directly.
- Every later QLayer retains the historical A4/integer-bypass behavior.
- No checkpoint parameter, BN statistic, dataset transform, time step, or
  validation rule changes.

## First-stage interpretation

The initial run is a paired inference-only sensitivity test on the full UDD
validation set. An absolute within-checkpoint change of at least 0.5 mIoU
points is considered material enough to justify matched retraining.

It is not a training comparison: QAD observed 127 training epochs, while the
available STE-QAT checkpoint comes from the shortened 16-epoch budget. The
cross-method gap therefore remains confounded until both variants are trained
under the same run and learning-rate-schedule budgets.
