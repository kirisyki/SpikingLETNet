# Corrected MAC/SOP Summary

Source: corrected `summary_energy.csv`; 20 UDD6 validation images, T=8, input size 400x400.
SOPs use `(time-inclusive dense MACs / actual_timesteps) * mean cumulative spikes`, avoiding double-counting T.

| Network | Variant | Dense MACs / image | Charged dense MACs / image | SOPs / image | Mean spikes/input |
|---|---:|---:|---:|---:|---:|
| FP | small | 7.124544e+10 | 1.261555e+10 | 3.019172e+09 | 0.362713 |
| FP | middle | 6.864656e+10 | 5.909728e+09 | 2.486847e+09 | 0.285447 |
| FP | max | 5.507136e+10 | 3.680640e+09 | 1.590453e+09 | 0.257665 |
| INT4 | small | 7.124544e+10 | 1.371789e+10 | 2.421476e+09 | 0.341345 |
| INT4 | middle | 6.864656e+10 | 6.512096e+09 | 2.246392e+09 | 0.297459 |
| INT4 | max | 5.507136e+10 | 4.052352e+09 | 1.824093e+09 | 0.295684 |
