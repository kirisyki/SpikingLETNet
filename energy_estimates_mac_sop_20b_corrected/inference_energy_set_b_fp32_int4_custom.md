# Corrected Inference Compute Energy (set_b_fp32_int4_custom)

Energy includes only charged dense MACs and corrected spike-path AC/SOPs. Memory and neuron-update energy are excluded.

| Network | Variant | E_MAC (pJ) | E_AC (pJ) | MAC energy (uJ/img) | SOP energy (uJ/img) | Total compute (uJ/img) |
|---|---:|---:|---:|---:|---:|---:|
| FP | small | 4.6 | 0.9 | 58031.539 | 2717.255 | 60748.794 |
| FP | middle | 4.6 | 0.9 | 27184.749 | 2238.162 | 29422.911 |
| FP | max | 4.6 | 0.9 | 16930.944 | 1431.408 | 18362.352 |
| INT4 | small | 0.08 | 0.03 | 1097.431 | 72.644 | 1170.075 |
| INT4 | middle | 0.08 | 0.03 | 520.968 | 67.392 | 588.359 |
| INT4 | max | 0.08 | 0.03 | 324.188 | 54.723 | 378.911 |
