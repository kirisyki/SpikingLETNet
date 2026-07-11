# LETNet energy accounting v2

This document defines the version-2 theoretical arithmetic estimator and the
RTX PRO 6000 board-energy measurement protocol. Legacy scripts and result
directories remain unchanged and must not be mixed with v2 tables.

## Baselines and scope

- `T=8 mixed`: core-arithmetic SNN estimate. Layers are classified by explicit
  input semantics from `tools/energy_layer_policy_v2.yaml`.
- `T=1 all-dense`: QIF-enabled, one-step, dense-tensor PyTorch baseline. It is
  retained for compatibility but is **not** a continuous-activation ANN.
- Core arithmetic includes convolution, transposed convolution, Conv1d,
  linear layers, and both attention matrix products. Bias, normalization,
  softmax, pooling, interpolation, neuron update/reset, memory traffic, and
  data movement are excluded unless a future extended model states otherwise.

The theoretical arithmetic estimate and measured GPU board energy answer
different questions. They must be reported side by side, not treated as direct
measurements of one another.

## MAC and SOP definitions

For a layer with an explicit time dimension, define:

- `M_executed`: dense MACs for the tensor actually executed, including its
  observed time slices;
- `M_one_step`: `M_executed / observed_time_slices`;
- `s`: mean QIF cumulative spike count, obtained from the mean non-negative
  integer activation value;
- `r`: firing rate paired with `M_executed`,
  `s / observed_time_slices`.

The mean-rate approximation is:

```text
SOP = M_executed * r = M_one_step * s
```

The estimator evaluates both expressions per hook invocation and asserts that
they agree. `conv_macs()` uses `input.numel()` or `output.numel()`, so an
explicit `[T,N,C,H,W]` tensor already contributes all `T` dense executions.
The QIF `bin=False` value `k` already denotes a cumulative count in `0..T`;
therefore the executed MAC count must be scaled by `k/T`, or equivalently its
one-step MAC count must be scaled by `k`.

Time slices are recorded per layer because the network contains both explicit
`[8,N,...]` tensors and time-collapsed `[1,N,...]` tensors. The CSV therefore
contains `observed_time_slices`, `input_neuron_sites`,
`dense_macs_executed`, and `dense_macs_one_step_equivalent`.

Residual additions may produce integer event multiplicities above one. Each
positive integer is treated as that many event-driven additions. Raw and
derived outputs therefore report both `mean_spike_count_per_input` and
`sop_density_per_executed_mac`; they are not interchangeable when `T > 1`.

## Layer semantics and precision

Every supported compute layer must match exactly one policy rule. Unmatched or
multiply matched layers fail the run. `legacy-auto` is available only for
reproducing the former value-based classifier and is not the v2 default.

QAT precision is assigned per layer:

- a `QLayer` with both weight and activation quantization enabled and `k=4` is
  charged as int4;
- attention matrix products, Conv1d attention helpers, and unquantized layers
  are charged as FP;
- unsupported bit widths fail rather than silently reusing int4 parameters.

The efficient attention hook counts both `q @ k.T` and `attention @ v` for
every chunk produced by the implementation's four-way split.

## Recomputing theoretical results

T=8 FP and QAT, first 20 validation images:

```bash
python tools/estimate_spikingletnet_energy_v2.py \
  --variants small middle max \
  --checkpoint-kinds fp qat \
  --timestep 8 --max-batches 20 --batch-size 1 \
  --device cuda:0 \
  --output-dir energy/theoretical/v2/snn_t8_20_images
```

T=1 QIF dense-tensor baseline:

```bash
python tools/estimate_spikingletnet_energy_v2.py \
  --variants small middle max \
  --checkpoint-kinds fp qat \
  --timestep 1 --accounting-mode all-dense \
  --max-batches 20 --batch-size 1 --device cuda:0 \
  --output-dir energy/theoretical/v2/qif_t1_dense_20_images
```

Use `--max-batches 0` for the full split. Each run writes raw per-layer CSV,
summary CSV/JSON, hashes, sample identifiers, quantization inventory, policy
version, excluded operations, software versions, and a run manifest.

Derived Set A/Set B tables must be built from raw v2 outputs:

```bash
python tools/build_energy_reports_v2.py \
  --run qif_t1=energy/theoretical/v2/qif_t1_dense_20_images \
  --run snn_t8=energy/theoretical/v2/snn_t8_20_images \
  --baseline-run qif_t1 \
  --output-dir energy/theoretical/v2/derived_20_images
```

Generate operation, theoretical-energy, and measured-energy reconciliation
against the preserved legacy results:

```bash
python tools/compare_legacy_energy_v2.py
```

## RTX PRO 6000 measurement protocol

The measurement program binds CUDA and NVML by GPU UUID, disables TF32 by
default, preloads images on the GPU, excludes preprocessing and transfers, and
rejects foreign compute processes unless explicitly allowed.

Gross board energy from the NVML cumulative energy counter is the primary
metric. Bounded trapezoidal power integration is retained as a cross-check.
Net energy subtracting the stable mean of pre/post idle power is secondary and
is more sensitive to idle drift.

Formal run:

```bash
python tools/measure_pro6000_ann_energy_v2.py \
  --warmup-images 200 --measure-images 5000 --trials 5 \
  --idle-seconds 10 --sample-interval-ms 100 --tf32 off \
  --output-dir energy/measurements/v2/pro6000_qif_t1_dense_fp32_5x5000_images
```

Acceptance thresholds are throughput CV at most 2%, gross-energy CV at most
5%, and net-energy CV at most 10%. The result is marked `unstable` if any gate
fails; an unstable run must not be promoted as the reference measurement.

The measured model keeps QIF nodes enabled at `T=1`. Report it as the
"T=1 QIF dense FP32 PyTorch baseline", not as a conventional ANN.

## Validation

Run the deterministic tests with:

```bash
python -m unittest discover -s tests -p 'test_*_v2.py' -v
```

Tests cover time-bearing MACs, equivalent SOP definitions, attention products,
policy coverage for all variants, mixed-precision energy, strict JSON, CV, and
bounded power integration.
