# LETNet energy tools v2: implementation and validation report

Date: 2026-07-11

## Outcome

The legacy estimator's cumulative-spike SOP scaling is conceptually correct:
it divides time-bearing dense MACs by the actual number of time slices and then
multiplies by the mean QIF cumulative spike count. Legacy results are still not
the preferred final source because they classify layer semantics from runtime
values, omit attention matrix products and Conv1d, and apply one global QAT
precision.

The initial v2 implementation introduced the opposite error by multiplying
time-bearing dense MACs directly by the cumulative spike count. This inflated
most T=8 spike-layer SOPs by approximately eight. The formula, schema, tests,
raw results, derived reports, reconciliation tables, and conclusions have now
been corrected.

The current v2 pipeline adds an explicit semantic layer policy, observed
per-layer time slices, per-layer FP/int4 accounting, both attention matrix
products, Conv1d, strict manifests, derived reports, legacy reconciliation,
and a multi-trial PRO6000 board-energy protocol.

## Correct SOP definition

For a layer invocation:

```text
M_executed = dense MACs of the executed tensor, including observed time slices
M_one_step = M_executed / observed_time_slices
s          = mean non-negative QIF value = cumulative spikes per input
r          = s / observed_time_slices

SOP = M_executed * r = M_one_step * s
```

`QIFNode(bin=False)` quantizes each activation to an integer `k` in `0..T`.
Under the project's cumulative-count interpretation, `k` already represents
the number of spikes over the inference window; it is not a per-slot firing
rate. Therefore, when dense MACs already contain the explicit T dimension,
they must be multiplied by `k/T`. When MACs are one-step counts, they must be
multiplied by `k` directly.

The estimator records `mean_spike_count_per_input`,
`configured_firing_rate`, and `sop_density_per_executed_mac` separately. It
uses the observed per-layer time-slice count in the SOP formula and asserts the
two equivalent expressions for every hook call.

For max-QAT at T=8, spike-classified layers have 52.624640 G executed dense
MACs, 6.581440 G one-step MACs, and 2.004370 G SOPs per image. The layer-MAC
weighted effective firing rate is 3.8088%, so 52.624640 G × 3.8088% =
2.004370 G. The earlier 16.030482 G value was the erroneous cumulative-count
multiplier without division by T.

## Theoretical core-arithmetic results

The following values are mJ/image for the first 20 validation images. They are
core arithmetic estimates only; excluded costs include normalization, softmax,
pooling, interpolation, neuron update/reset, memory, transfers, and framework
overhead.

| Variant | Checkpoint | Set | T=1 QIF dense | T=8 SNN | T8 / T1 |
|---|---|---:|---:|---:|---:|
| max | FP | A/B | 33.832779 | 13.051399 | 0.385762 |
| max | QAT | A | 1.936674 | 0.880851 | 0.454827 |
| max | QAT | B | 0.841842 | 0.513843 | 0.610380 |
| middle | FP | A/B | 40.077943 | 26.359335 | 0.657702 |
| middle | QAT | A | 2.126416 | 1.398512 | 0.657685 |
| middle | QAT | B | 0.823732 | 0.617864 | 0.750079 |
| small | FP | A/B | 44.944948 | 48.004937 | 1.068083 |
| small | QAT | A | 4.213751 | 4.306464 | 1.022002 |
| small | QAT | B | 2.815655 | 2.907264 | 1.032536 |

The corrected max and middle T=8 estimates are lower than their matching T=1
baselines under both energy sets. The small T=8 model remains slightly higher:
6.81% for FP, 2.20% for QAT Set A, and 3.25% for QAT Set B. Thus sparsity alone
does not guarantee a reduction when dense attention and other non-spike paths
dominate, but the previous claim that all Set-B QAT models were higher was a
consequence of the SOP overcount.

The T=1 baseline keeps QIF nodes enabled and runs one time step. It is a
"T=1 QIF dense-tensor baseline", not a conventional continuous-activation ANN.

## Legacy reconciliation

- Adding attention matrix products and Conv1d increases T=1 MACs by `0.32%` to
  `4.83%`, depending on variant.
- Static semantic classification reduces charged dense T=8 MACs by `11.46%` to
  `38.24%` relative to legacy.
- Corrected v2 T=8 SOPs are `1.02x` to `1.25x` the legacy values. The remaining
  difference comes from semantic layer classification and expanded operator
  coverage, not from a global T factor.
- FP and QAT now have zero layer-mode mismatches for each variant. Their SOPs
  may still differ because their actual event activity differs.
- QAT is mixed precision. QLayer W4A4 arithmetic is int4, while unwrapped
  attention products, Conv1d, and unquantized layers remain FP. The former
  all-int4 legacy energy particularly under-estimates the small QAT model,
  whose attention cost is material.

Exact operation and energy differences are stored in
`energy/theoretical/v2/reconciliation_20_images/`.

## PRO6000 formal measurement

Target: `SpikingLETNet_shallow_max`, FP32 checkpoint, T=1 QIF dense-tensor
execution, batch size 1, 400x400 input, TF32 disabled.

Protocol: 200 warmup images; five trials of 5000 images; 10 s idle before and
after each trial; CUDA/NVML binding by UUID; foreign compute-process rejection;
NVML cumulative board-energy counter as primary; bounded power integration as
cross-check.

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition,
`GPU-4dacf618-7fd5-e149-1f34-0a5f872b1146`, PCI `00000000:A8:00.0`, 600 W
power limit, driver `580.95.05`.

| Metric | Median | CV | Acceptance |
|---|---:|---:|---:|
| gross J/image | 1.329127400 | 0.592% | <= 5% |
| net J/image | 0.543892035 | 0.473% | <= 10% |
| images/s | 106.440745 | 1.108% | <= 2% |

Status: `ready`. All five trials use the cumulative energy counter. Bounded
power integration differs from the counter by `-0.753%` to `+0.125%`, providing
an independent consistency check.

Gross board energy is the primary measured metric. Net energy is secondary
because it depends on the idle baseline. Neither is directly comparable to the
theoretical arithmetic-only value without modeling memory, neuron kernels,
normalization, launch overhead, and board idle power.

The legacy single-trial measurement reports 1.565453 J/image gross,
0.514026 J/image net, and 84.391 images/s. Version 2 differs by -15.10%, +5.81%,
and +26.13%, respectively. This is a combined protocol/environment difference:
the legacy run uses one power-integration trial and driver `595.58.03`; v2 uses
five cumulative-counter trials and driver `580.95.05`. It must not be presented
as a pure code-speedup result.

## Validation evidence

- 13 deterministic unit tests pass.
- FP and trusted QAT checkpoints run on the PRO6000.
- Semantic policy covers small/middle/max with exactly one rule per supported
  layer.
- The QAT attention wrapper regression is covered.
- All 1028 formal per-layer rows are finite and non-negative.
- All v2 JSON files parse under strict JSON; no NaN/Inf values are present.
- Manifests include input hashes, checkpoint/config/split hashes, policy hash,
  software versions, sample identifiers, and excluded operations.

## Deliverables

- `tools/estimate_spikingletnet_energy_v2.py`
- `tools/energy_layer_policy_v2.yaml`
- `tools/energy_accounting.py`
- `tools/build_energy_reports_v2.py`
- `tools/compare_legacy_energy_v2.py`
- `tools/measure_pro6000_ann_energy_v2.py`
- `tests/test_energy_tools_v2.py`
- `docs/energy_estimation_v2.md`
- `energy/theoretical/v2/qif_t1_dense_20_images/`
- `energy/theoretical/v2/snn_t8_20_images/`
- `energy/theoretical/v2/derived_20_images/`
- `energy/theoretical/v2/reconciliation_20_images/`
- `energy/measurements/v2/pro6000_qif_t1_dense_fp32_5x5000_images/`

Legacy scripts and result directories remain the historical reproduction
baseline. New claims should cite v2 raw outputs and manifests.
