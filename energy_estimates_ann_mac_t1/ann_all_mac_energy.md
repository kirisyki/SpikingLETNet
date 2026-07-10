# ANN-mode All-MAC Inference Energy

Assumption: activations are not expanded to T time steps; the whole network is evaluated as dense ANN MACs. Energy is compute-only and excludes memory, control, routing, and neuron-update costs.

| Network | Variant | MACs / image | E_MAC (pJ) | Compute energy (uJ/image) |
|---|---:|---:|---:|---:|
| FP32 | small | 9.320640e+09 | 4.6 | 42874.944 |
| FP32 | middle | 8.684560e+09 | 4.6 | 39948.976 |
| FP32 | max | 7.298880e+09 | 4.6 | 33574.848 |
| INT4_SetA_conservative | small | 9.320640e+09 | 0.23 | 2143.747 |
| INT4_SetA_conservative | middle | 8.684560e+09 | 0.23 | 1997.449 |
| INT4_SetA_conservative | max | 7.298880e+09 | 0.23 | 1678.742 |
| INT4_SetB_custom | small | 9.320640e+09 | 0.08 | 745.651 |
| INT4_SetB_custom | middle | 8.684560e+09 | 0.08 | 694.765 |
| INT4_SetB_custom | max | 7.298880e+09 | 0.08 | 583.910 |
