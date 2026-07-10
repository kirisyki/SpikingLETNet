# Hardware Energy References for MAC/AC Estimation

This note collects reference hardware numbers that can be used to instantiate
`energy_pj.fp.mac`, `energy_pj.fp.ac`, `energy_pj.int4.mac`, and
`energy_pj.int4.ac` in `tools/estimate_spikingletnet_energy.py`.

The values below are not universal constants. They depend on process node,
voltage, datapath width, accumulator width, memory hierarchy, dataflow, and
whether zero activations are actually gated. They should therefore be treated as
scenario parameters.

## 1. Horowitz-style arithmetic baseline

The frequently cited Horowitz ISSCC 2014 table gives rough operation energies
for CMOS arithmetic and memory access. Common values used in neural-network
energy estimates include:

```text
32-bit integer add       ~0.1 pJ
32-bit floating add      ~0.9 pJ
32-bit integer multiply  ~3.1 pJ
32-bit floating multiply ~3.7 pJ
8-bit integer add        ~0.03 pJ
8-bit integer multiply   ~0.2 pJ
32-bit SRAM/cache read   ~5 pJ
32-bit DRAM read         ~640 pJ
```

For a floating-point MAC, a practical estimate is:

```text
FP32 MAC ~= FP32 multiply + FP32 add ~= 3.7 + 0.9 = 4.6 pJ
FP32 AC  ~= FP32 add ~= 0.9 pJ
```

For integer/fixed-point SNN comparisons, many works use a cheaper integer
accumulation model:

```text
INT32 MAC ~= 3.1 to 3.2 pJ
INT32 AC  ~= 0.1 pJ
```

## 2. ACE-SNN / digital SNN reference

ACE-SNN reports an energy model for ANN/SNN comparison and lists:

```text
E_MAC = 3.1 pJ
E_AC  = 0.1 pJ
```

for a 32-bit operation reference, and explicitly notes that the spiking energy
assumes zero-gating logic: compute is activated only when an input spike is
received. This matches the assumption in our current SOP estimator.

This is a useful conservative digital SNN baseline:

```yaml
energy_pj:
  fp:
    mac: 4.6
    ac: 0.9
  int4:
    mac: 0.23
    ac: 0.03
```

Here `int4.mac=0.23` uses Horowitz's 8-bit integer multiply+add as a conservative
proxy for INT4, rather than claiming a more aggressive 4-bit custom datapath.

## 3. INT4 digital datapath estimate

For a custom INT4 datapath, the MAC energy can be lower than an 8-bit proxy.
A simple first-order estimate is:

```text
4-bit multiply ~= 8-bit multiply * (4/8)^2 ~= 0.2 * 0.25 = 0.05 pJ
4-bit or narrow add ~= 0.015 to 0.03 pJ
INT4 MAC ~= 0.06 to 0.10 pJ
INT4 AC  ~= 0.015 to 0.03 pJ
```

However, if accumulation is performed in a wider 16-bit or 32-bit accumulator,
the AC part may be closer to the accumulator width than the operand width.
For that reason, a safer custom-digital INT4 range is:

```yaml
energy_pj:
  int4:
    mac: 0.08
    ac: 0.03
```

Use this only if the target hardware truly implements narrow INT4 MACs and
zero-activation gating.

## 4. Bit-serial and CIM references

Recent bit-serial or compute-in-memory accelerators report much lower effective
energy per operation, but these are architecture-specific and should not be
mixed directly with a generic digital MAC model.

Examples:

```text
BitFair, 12 nm bit-serial accelerator: up to 117 BTOPS/W, ~0.07 pJ/SOP.
P-8T SRAM CIM, 28 nm, 4-bit input and 8-bit weight: 50.07 TOPS/W, about 0.02 pJ/op if TOPS is interpreted as primitive operations.
ACE-SNN PIM analog accumulation: E_BLP = 0.08 pJ per in-memory accumulation.
ODIN digital neuromorphic processor, 28 nm: 12.7 pJ/SOP, but includes a full online-learning neuromorphic processor context rather than only an arithmetic AC.
```

These numbers are useful as lower-bound or architecture-specific comparisons,
not as default parameters for our current estimator.

## Recommended parameter sets

### Set A: FP32 vs conservative INT4 digital

This is the safest starting point for reporting.

```yaml
energy_pj:
  fp:
    mac: 4.6
    ac: 0.9
    mem_read: 0.0
    mem_write: 0.0
    neuron_update: 0.0
  int4:
    mac: 0.23
    ac: 0.03
    mem_read: 0.0
    mem_write: 0.0
    neuron_update: 0.0
```

### Set B: FP32 vs custom INT4 digital

Use this if the target hardware has real INT4 MAC units and narrow/gated AC.

```yaml
energy_pj:
  fp:
    mac: 4.6
    ac: 0.9
    mem_read: 0.0
    mem_write: 0.0
    neuron_update: 0.0
  int4:
    mac: 0.08
    ac: 0.03
    mem_read: 0.0
    mem_write: 0.0
    neuron_update: 0.0
```

### Set C: ACE-SNN style integer digital baseline

Use this when aligning with papers that use integer MAC/AC rather than FP32 MAC.

```yaml
energy_pj:
  fp:
    mac: 3.1
    ac: 0.1
    mem_read: 0.0
    mem_write: 0.0
    neuron_update: 0.0
  int4:
    mac: 0.23
    ac: 0.03
    mem_read: 0.0
    mem_write: 0.0
    neuron_update: 0.0
```

## Practical choice for this project

For the current SpikingLETNet comparison, use Set A as the main result and Set B
as a sensitivity analysis. Report clearly that SOP energy assumes ideal
zero-activation gating, i.e. zero activations trigger no AC operation.

References:

- Mark Horowitz, "Computing's Energy Problem (and what we can do about it)",
  ISSCC 2014.
- Datta et al., "ACE-SNN: Algorithm-Hardware Co-design of Energy-Efficient &
  Low-Latency Deep Spiking Neural Networks for 3D Image Recognition", Frontiers
  in Neuroscience, 2022.
- Moons et al., "Minimum Energy Quantized Neural Networks", arXiv:1711.00215.
- Sharma et al., "Bit Fusion: Bit-Level Dynamically Composable Architecture for
  Accelerating Deep Neural Networks", arXiv:1712.01507.
- Li and Gao, "BitFair: A 12nm Bit-Serial CNN Accelerator...", arXiv:2607.05445.
- Frenkel et al., "A 0.086-mm2 12.7-pJ/SOP 64k-Synapse 256-Neuron
  Online-Learning Digital Spiking Neuromorphic Processor in 28nm CMOS",
  arXiv:1804.07858.
