# PRO6000 TensorRT FP32 vs INT8 Energy

Status: **ready**. Gross board energy is primary; net energy is secondary.

| Engine | Gross J/image | CV | Net J/image | CV | Images/s | CV |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.268687200 | 0.478% | 0.195016245 | 0.577% | 1137.990354 | 0.154% |
| INT8 | 0.155802800 | 0.938% | 0.089325096 | 1.731% | 1260.673741 | 0.276% |

## Main comparison

- Gross energy reduction: **42.0133%**
- Net energy reduction: **54.1961%**
- Throughput ratio: **1.1078x**
- Median paired gross-energy reduction: 41.9037%

## Actual engine precision

- FP32 layers touching INT8: 0
- INT8 layers touching INT8: 285 / 383
- INT8 correlation: 74 / 74
- INT8 deconv: 3 / 3
- INT8 gemm: 15 / 15

## Historical PyTorch context

The existing T=1 QIF dense PyTorch FP32 result is 1.329127400 gross J/image,
0.543892035 net J/image, and 106.440745 images/s. It is context only.

## Scope

Fixed-shape batch-1 dense ANN-form TensorRT inference; engine build, preprocessing,
transfers, output copies, CUDA Graph, and accuracy evaluation are excluded. This does not
measure T=8 spike execution or the synchronous chip's temporal-reuse mechanism.
