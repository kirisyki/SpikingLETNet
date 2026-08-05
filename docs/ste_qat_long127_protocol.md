# STE-QAT W4A4 long-budget protocol

## Purpose

Remove the central fairness confound between the historical QAD checkpoint
(127 observed training epochs) and the shortened STE-QAT baseline (16 epochs).

## Frozen comparison

- Model: `SpikingLETNet_shallow_max`
- Dataset: UDD6 training patches and full validation patches
- Initialization: the same historical FP-QIF checkpoint used by QAD
- Seed: 1234
- Student: task-loss-only STE-QAT
- Weights: W4 for all 71 QLayers
- Inputs: historical A4/integer-bypass behavior for all 71 QLayers, including
  dynamic per-tensor A4 fake quantization of the normalized image
- Batch size: 64
- Batches per epoch: 401
- Run budget: 127 complete epochs
- Polynomial LR horizon: 150 epochs, exponent 0.9
- Optimizer: Adam, lr `1e-3`, weight decay `1e-4`
- Validation: complete UDD validation set after every epoch
- Selection: best validation mIoU across the 127 epochs

## Isolation

The run starts fresh from the shared FP checkpoint. It does not resume or
overwrite the shortened 16-epoch run. Checkpoint-last and checkpoint-best both
store model, optimizer, global iteration, RNG state, protocol, and manifest.

## Interpretation

This seed-1234 run can remove the training-budget confound for the central
QAD-versus-STE comparison. It cannot establish cross-seed significance.
