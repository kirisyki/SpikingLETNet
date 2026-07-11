# Quantized Spiking Convolution Temporal Reuse

This directory is self-contained and holds the implementation, tests, and generated results for the
`SpikingLETNet_shallow_max` QAT temporal-reuse statistic. No existing model or energy-tool source is
modified.

## Agreed accounting rule

- Run the complete INT4 QAT checkpoint with one software time frame and batch size 1.
- Capture the exact integer activation code produced inside every selected `QLayer.quantize_input()`.
- Include `Conv2d` and `ConvTranspose2d` layers classified as spike-input layers by
  `tools/energy_layer_policy_v2.yaml`; exclude RGB, Transformer, classifier, and other dense inputs.
- Expand code `q` as `q` leading ones followed by `8-q` zeros.
- Compare each timestep 2–8 only with timestep 1, independently for each input channel and output
  receptive-field position.
- Always compute timestep 1, even for an all-zero window.
- Use the real kernel, stride, padding, dilation, and groups. Convert transposed convolution to
  zero insertion followed by ordinary convolution.

For a window whose minimum positive integer code is `m`, the number of computed timesteps is
`9-m`. An all-zero window computes one timestep. This is exactly equivalent to explicit 8-step
expansion but avoids materializing every pulse tensor.

The primary network result is MAC-weighted:

```text
mean computed timesteps = total executed MACs / total one-timestep MACs
```

## Run

From the repository root:

```bash
python -m unittest temporal_reuse_analysis.test_temporal_reuse
python temporal_reuse_analysis/analyze_temporal_reuse.py
```

The default experiment uses the agreed checkpoint and the first 500 UDD6 validation patches with
`batch_size=1`. Results are written under `temporal_reuse_analysis/results/qat_max_val500/`:

- `report.md`: concise result and per-layer table.
- `summary.json`: aggregate metrics and full provenance.
- `layer_statistics.csv`: exact per-layer counts, histograms, geometry, and MAC totals.

