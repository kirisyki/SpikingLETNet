# Corrected Inference Compute Energy (set_a_fp32_int4_conservative)

Energy includes only charged dense MACs and corrected spike-path AC/SOPs. Memory and neuron-update energy are excluded.

| Network | Variant | E_MAC (pJ) | E_AC (pJ) | MAC energy (uJ/img) | SOP energy (uJ/img) | Total compute (uJ/img) |
|---|---:|---:|---:|---:|---:|---:|
| FP | small | 4.6 | 0.9 | 58031.539 | 2717.255 | 60748.794 |
| FP | middle | 4.6 | 0.9 | 27184.749 | 2238.162 | 29422.911 |
| FP | max | 4.6 | 0.9 | 16930.944 | 1431.408 | 18362.352 |
| INT4 | small | 0.23 | 0.03 | 3155.114 | 72.644 | 3227.759 |
| INT4 | middle | 0.23 | 0.03 | 1497.782 | 67.392 | 1565.174 |
| INT4 | max | 0.23 | 0.03 | 932.041 | 54.723 | 986.764 |
