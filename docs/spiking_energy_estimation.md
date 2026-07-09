# SpikingLETNet 计算能耗估计方案

本文档记录 `SpikingLETNet_shallow_small`、`SpikingLETNet_shallow_middle`、
`SpikingLETNet_shallow_max` 的理论计算能耗估计方案。配套脚本为
`tools/estimate_spikingletnet_energy.py`。

## 目标

目标不是直接给出不可验证的绝对硬件功耗，而是建立一个可复现、可替换硬件参数的
计算能耗估计流程：

- 比较 `small / middle / max` 三个结构。
- 同时比较普通 FP checkpoint 和 QAT int4 complete checkpoint。
- 主排序使用 core compute energy：`MAC + SOP/AC`。
- memory、neuron update 等能耗作为 optional extended 项，不参与默认主排序。

## 核心口径

本项目的 `QIFNode` 通过 `Quant` 输出 `0..T` 的整数值。我们采用累计 spike 口径：

```text
activation = k  等价于  k 个时间步脉冲事件
```

因此 SOPs 不按 `activation != 0` 计 1 次，而按输入激活的数值和估计：

```text
mean_spikes_per_input = sum(input_activation) / numel(input_activation)
SOP_layer ~= dense_MAC_layer * mean_spikes_per_input
```

这个公式用 dense MAC 的真实输出形状、kernel、groups、time/batch 维度作为基准，再乘以输入脉冲均值。它比简单的
`sum(input_spikes) * kernel_size * out_channels` 更稳，因为 stride、groups、输出尺寸已经体现在 dense MAC 中。

## 混合层分类

脚本不按“所有 Conv 都是 SNN”来算，而是按该层实际输入张量分类：

- 输入为非负整数、范围在 `0..T`，且不是 transformer/linear：按 spike/SOP/AC。
- 输入为原始图像、连续中间激活、transformer token、attention/MLP：按 dense MAC。
- `Linear` 默认按 dense MAC，因为当前三种模型的线性层来自 transformer。
- `pooling / shuffle / concat / reshape / repeat / interpolate` 默认不计入 core compute。
- `BatchNorm / QIFNode` 默认不计入 core compute，可通过 extended 参数表达近似开销。

这条规则很重要：网络中 transformer 前有 `output3.mean(0)`，后续 attention 和 MLP 是 dense token 计算，不应当按事件驱动 AC 处理。

## 能耗模型

主模型：

```text
E_core = MAC_charged * E_MAC + AC_charged * E_AC
```

FP 和 QAT/int4 使用分开的参数：

```yaml
energy_pj:
  fp:
    mac: 1.0
    ac: 1.0
  int4:
    mac: 1.0
    ac: 1.0
```

默认示例参数是 normalized 值，不代表某个工艺节点的真实 pJ。要做论文或硬件对齐实验时，应替换为目标硬件、工艺或仿真器给出的参数。

可选扩展项：

```text
E_total = E_core + E_mem_read + E_mem_write + E_neuron_update
```

脚本支持 `--include-extended`，并从同一个 energy config 中读取：

```yaml
mem_read: 0.0
mem_write: 0.0
neuron_update: 0.0
```

## 默认实验设置

输入和数据：

- dataset: UDD6
- split: `/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt`
- input size: `400,400`
- T: `8`
- default max batches: `20`
- `--max-batches 0` 表示跑完整 split

默认 FP checkpoint：

```text
small  checkpoint/udd/SpikingLETNet_shallow_smallbs64gpu1_trainval20260611-234750/model_best.pth
middle checkpoint/udd/SpikingLETNet_shallow_middlebs64gpu1_trainval20260611-222604/model_best.pth
max    checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth
```

默认 QAT checkpoint：

```text
small  QAT_checkpoint/udd/SpikingLETNet_shallow_smallbs64gpu1_trainval20260613-200147/model_q_best_complete.pt
middle QAT_checkpoint/udd/SpikingLETNet_shallow_middlebs96gpu1_trainval20260613-155726/model_q_best_complete.pt
max    QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260613-170955/model_q_best_complete.pt
```

QAT complete checkpoint 是可信本地 pickle 模型对象，脚本使用 `torch.load(..., weights_only=False)` 加载。不要对不可信文件使用这个加载方式。

## 运行方式

快速 smoke test：

```bash
python tools/estimate_spikingletnet_energy.py \
  --variants small \
  --checkpoint-kinds fp \
  --synthetic \
  --input-size 64,64 \
  --max-batches 1 \
  --output-dir /tmp/spiking_energy_smoke
```

默认比较：

```bash
python tools/estimate_spikingletnet_energy.py \
  --energy-config tools/energy_params.example.yaml \
  --output-dir energy_estimates
```

完整验证集：

```bash
python tools/estimate_spikingletnet_energy.py \
  --energy-config tools/energy_params.example.yaml \
  --max-batches 0 \
  --output-dir energy_estimates_full
```

## 输出

`layer_energy.csv`：逐层统计，包括：

- `dense_macs_total`
- `dense_macs_charged`
- `sop_total`
- `ac_charged`
- `mean_spikes_per_input`
- `nonzero_ratio`
- `core_energy_pj`
- `total_energy_pj`
- `classifications`

`summary_energy.csv` 和 `summary_energy.json`：模型级汇总，包括：

- `variant`
- `checkpoint_kind`
- `precision`
- `processed_batches`
- `dense_macs_total`
- `dense_macs_charged`
- `sop_total`
- `core_energy_pj`
- `total_energy_pj`

## 筛选标准

推荐报告中使用以下顺序筛选：

1. 主方案：混合 `MAC + SOP/AC`，FP 和 int4 使用分开的硬件参数。
2. 对照上界：`dense_macs_total`，表示完全不利用 spike 稀疏时的计算规模。
3. 诊断项：逐层 `sop_total / dense_macs_total`、`core_energy_pj` 占比，定位高能耗层。
4. 可选扩展：memory/neuron energy 只在硬件参数可靠时加入排序。
5. 后续模型选择应结合 mIoU 或 Pareto 指标，不只看最低能耗。

## 参考

- Lemaire et al., energy estimation for SNNs, arXiv:2210.13107.
- Chen et al., Eyeriss v2, arXiv:1807.07928.
- Horowitz, Computing's energy problem and what we can do about it, ISSCC 2014.
