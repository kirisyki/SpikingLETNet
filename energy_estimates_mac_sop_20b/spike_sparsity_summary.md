# Spike Sparsity Summary

Source: `energy_estimates_mac_sop_20b/layer_energy.csv`.
Statistics cover only layers classified as `spike` by the mixed MAC/SOP rule; dense image and transformer paths are excluded.
Experiment setting: UDD6 validation first 20 images, batch size 1, input size 400x400, T=8.

## Definitions

For a QIF activation value `k`, the method treats it as `k` emitted spikes over `T` binary time slots.

```text
mean_spikes_per_aggregate_input = sum(k) / numel(k)
expanded_firing_rate = sum(k) / (T * numel(k))
expanded_spike_sparsity = 1 - expanded_firing_rate
aggregate_zero_sparsity = count(k == 0) / numel(k)
```

`expanded_spike_sparsity` is the preferred spike sparsity metric because it measures sparsity after expanding integer QIF outputs into T binary spike slots. `aggregate_zero_sparsity` is included only as an auxiliary diagnostic for zero-valued aggregate activations.

## Results

| Network | Variant | Spike layers | Mean spikes/input | Expanded firing rate | Expanded spike sparsity | Aggregate zero sparsity | SOPs total |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP | small | 58 | 0.363937 | 0.045492 | 0.954508 | 0.722062 | 4.828888e+11 |
| FP | middle | 59 | 0.292775 | 0.036597 | 0.963403 | 0.777582 | 3.978551e+11 |
| FP | max | 58 | 0.222343 | 0.027793 | 0.972207 | 0.817459 | 2.543512e+11 |
| INT4 | small | 57 | 0.297497 | 0.037187 | 0.962813 | 0.771617 | 3.872816e+11 |
| INT4 | middle | 57 | 0.249000 | 0.031125 | 0.968875 | 0.812220 | 3.593910e+11 |
| INT4 | max | 58 | 0.256353 | 0.032044 | 0.967956 | 0.811670 | 2.917652e+11 |

## Interpretation

The SOPs used by the current estimator already incorporate spike sparsity through `sum(k)`: zero aggregate activations contribute no synaptic operations, and larger integer activations contribute proportionally more events. This corresponds to an ideal event-driven hardware assumption where zero activations can be skipped completely. The table above quantifies the spike-path sparsity under that assumption.
