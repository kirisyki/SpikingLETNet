# PRO6000 ANN FP32 Inference Energy

Measurement target: `SpikingLETNet_shallow_max`, FP32 checkpoint, ANN-mode dense forward (`T=1`, no time-step expansion).

| Metric | Value |
|---|---:|
| measured images | 500 |
| active elapsed (s) | 5.885850 |
| idle elapsed (s) | 5.886066 |
| active mean power (W) | 127.725096 |
| idle mean power (W) | 91.288857 |
| gross J/image | 1.503441802 |
| net J/image | 0.429651210 |
| images/s | 84.949504 |

Gross energy includes the full GPU power during active forward. Net energy subtracts an equal-duration idle baseline.
Data loading and preprocessing are excluded; images are preloaded to GPU before measurement.
