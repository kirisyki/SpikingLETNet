# PRO6000 TensorRT FP32 vs INT8 Energy

Status: **ready**. Gross board energy is primary; net energy is secondary.

| Engine | Gross J/image | CV | Net J/image | CV | Images/s | CV |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 0.246228020 | 0.569% | 0.175577158 | 0.919% | 1228.924842 | 0.620% |
| INT8 | 0.159661480 | 1.083% | 0.091063720 | 1.612% | 1261.748228 | 0.346% |

## Main comparison

- Gross energy reduction: **35.1571%**
- Net energy reduction: **48.1346%**
- Throughput ratio: **1.0267x**
- Median paired gross-energy reduction: 35.2683%

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

## Long-window protocol

Each active trial executes 50000 batch-1 inferences by cycling the first 5000 preloaded validation images 10 times.
