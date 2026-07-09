# MAC/SOP Summary

Source: `summary_energy.csv`; 20 UDD6 validation images, T=8, input size 400x400.
Energy columns are intentionally omitted here; this file reports operation counts only.

| Network | Variant | Dense MACs total | Dense MACs / image | Charged dense MACs / image | SOPs total | SOPs / image | Mean spikes/input |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP | max | 1.101427e+12 | 5.507136e+10 | 3.680640e+09 | 2.543512e+11 | 1.271756e+10 | 0.257665 |
| FP | middle | 1.372931e+12 | 6.864656e+10 | 5.909728e+09 | 3.978551e+11 | 1.989275e+10 | 0.285447 |
| FP | small | 1.424909e+12 | 7.124544e+10 | 1.261555e+10 | 4.828888e+11 | 2.414444e+10 | 0.362713 |
| INT4 | max | 1.101427e+12 | 5.507136e+10 | 4.052352e+09 | 2.917652e+11 | 1.458826e+10 | 0.295684 |
| INT4 | middle | 1.372931e+12 | 6.864656e+10 | 6.512096e+09 | 3.593910e+11 | 1.796955e+10 | 0.297459 |
| INT4 | small | 1.424909e+12 | 7.124544e+10 | 1.371789e+10 | 3.872816e+11 | 1.936408e+10 | 0.341345 |
