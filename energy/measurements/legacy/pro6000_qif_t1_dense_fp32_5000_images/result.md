# PRO6000 ANN FP32 Inference Energy

Measurement target: `SpikingLETNet_shallow_max`, FP32 checkpoint, ANN-mode dense forward (`T=1`, no time-step expansion).

| Metric | Value |
|---|---:|
| measured images | 5000 |
| active elapsed (s) | 59.248301 |
| idle elapsed (s) | 59.248506 |
| active mean power (W) | 132.108800 |
| idle mean power (W) | 88.738894 |
| gross J/image | 1.565453311 |
| net J/image | 0.514025529 |
| images/s | 84.390606 |

Gross energy includes the full GPU power during active forward. Net energy subtracts an equal-duration idle baseline.
Data loading and preprocessing are excluded; images are preloaded to GPU before measurement.
