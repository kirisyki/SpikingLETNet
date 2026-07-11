# Energy tools validation source notes

- Audience: technical.
- Delivery mode: self-contained HTML.
- Validation status: Needs revision.
- Primary code reviewed:
  - `tools/estimate_spikingletnet_energy.py`
  - `tools/measure_pro6000_ann_energy.py`
  - `Network/model/SpikingLETNet_shallow_{small,middle,max}.py`
  - `Network/model/module/transformer.py`
  - `Network/model/module/neuron.py`
- Primary result artifacts:
  - `energy_estimates_ann_mac_t1/`
  - `energy_estimates_mac_sop_20b_corrected/`
  - `measurements/pro6000_ann_fp32_max/`
  - `measurements/pro6000_ann_fp32_max_5000/`
- Reproducible checks performed:
  - CSV/JSON arithmetic reconciliation.
  - Independent trapezoidal reintegration of both PRO6000 power traces.
  - CPU-only AST/CLI checks.
  - CPU-only 32x32 synthetic FP estimator smoke test written to `/tmp/letnet_energy_audit_smoke`.
  - Static reconstruction of omitted attention matrix-multiplication MACs.
- Chart omission: exact audit lookup is the primary task, while the 45 nm operation model and Blackwell board energy are not a common-scale comparison. Tables avoid implying comparability. No chart was rendered.
- QAT checkpoint internals were not independently deserialized because complete model files require `torch.load(..., weights_only=False)`; the sandbox rejected that unsafe pickle execution. Existing generated QAT rows were validated arithmetically, but the per-layer INT4-enable state remains unverified.
- External references checked:
  - Horowitz, ISSCC 2014, rough 45 nm / 0.9 V operation energies.
  - ACE-SNN, Frontiers in Neuroscience 2022, 32-bit MAC/AC and zero-gating assumptions.
  - NVIDIA NVML API reference, board-power semantics, one-second averaging on Ampere-or-newer GPUs, and total-energy counter.
  - NVIDIA CUDA Programming Guide, `CUDA_VISIBLE_DEVICES` ordinal remapping.

