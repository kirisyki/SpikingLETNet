# Corrected Spike Sparsity Summary

Source: corrected `layer_energy.csv`; spike-classified layers only. Experiment: UDD6 validation first 20 images, T=8, input 400x400.

| Network | Variant | Spike layers | Mean spikes/input | Expanded firing rate | Expanded spike sparsity | Aggregate zero sparsity | Corrected SOPs total |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP | small | 58 | 0.363937 | 0.045492 | 0.954508 | 0.722062 | 6.038345e+10 |
| FP | middle | 59 | 0.292775 | 0.036597 | 0.963403 | 0.777582 | 4.973694e+10 |
| FP | max | 58 | 0.222343 | 0.027793 | 0.972207 | 0.817459 | 3.180907e+10 |
| INT4 | small | 57 | 0.297497 | 0.037187 | 0.962813 | 0.771617 | 4.842952e+10 |
| INT4 | middle | 57 | 0.249000 | 0.031125 | 0.968875 | 0.812220 | 4.492784e+10 |
| INT4 | max | 58 | 0.256353 | 0.032044 | 0.967956 | 0.811670 | 3.648186e+10 |
