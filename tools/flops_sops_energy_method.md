# FLOPs/SOPs 统计与计算能耗估计方法说明

本文档说明 `tools/estimate_spikingletnet_energy.py` 中采用的 FLOPs、MACs、SOPs 统计方法，以及如何由这些统计量推导 `SpikingLETNet_shallow_small`、`SpikingLETNet_shallow_middle`、`SpikingLETNet_shallow_max` 的理论计算能耗。本文采用研究方法描述口径，重点说明可复现的计算规则，而不将某一硬件平台的绝对功耗假定为普适结论。

## 1. 研究目标与基本假设

本文关注的问题是：在给定输入分辨率、时间步数、模型权重和硬件单次操作能耗参数后，如何估算 SpikingLETNet 系列模型一次推理的理论计算能耗。

估计过程遵循三个基本假设。

第一，常规 ANN 或连续值路径中的卷积、反卷积和线性层使用 dense MACs 描述计算量。这里的 MAC 表示一次乘加操作。若需要转换为 FLOPs，可按常见约定将一次 MAC 视为 2 FLOPs，但本文主计算统一以 MAC 为基本单位，避免在 SNN 的 AC/SOP 口径中引入歧义。

第二，脉冲路径中的卷积和反卷积使用 SOPs 描述计算量。SOPs 表示 synaptic operations，即由输入脉冲触发的突触累加操作。对于事件驱动硬件，这类操作通常对应加法累加 AC，而不是 dense MAC。

第三，能耗估计采用可替换硬件参数。脚本不内置某一工艺节点的固定 pJ 数值，而是从 `energy_pj.fp` 和 `energy_pj.int4` 读取 `mac`、`ac` 等参数。这样可以分别评估非量化模型和 QAT/int4 模型。

## 2. 模型输入与时间维度

脚本将 UDD6 图像预处理为 `[B, 3, H, W]`，再复制为多时间步输入：

```text
input_seq = image.unsqueeze(0).repeat(T, 1, 1, 1, 1)
shape(input_seq) = [T, B, 3, H, W]
```

默认设置为：

```text
T = 8
H = 400
W = 400
B = 1
```

所有逐层统计都基于真实前向传播时捕获的输入和输出张量形状。因此 stride、padding、dilation、groups、时间步和 batch 维度都由实际模型执行结果决定，而不是由手写网络表格近似推断。

## 3. FLOPs 与 MACs 的统计

本文主表中使用 MACs，不直接使用 FLOPs。原因是能耗模型通常以一次乘加或一次累加为基本事件，而 FLOPs 在不同文献中可能将乘法和加法拆成两个 floating-point operations。

对于 dense 路径，脚本统计 `dense_macs_total`。该值表示如果该层完全按 dense 方式执行，需要多少 MAC。

### 3.1 Conv2d

对于二维卷积层，单层 MACs 计算为：

```text
MACs_conv2d = numel(output) * (Cin / groups) * Kh * Kw
```

其中：

```text
numel(output) = T * B * Cout * Hout * Wout
```

若该层输入没有时间维，公式自然退化为：

```text
numel(output) = B * Cout * Hout * Wout
```

该公式同时适用于普通卷积、逐点卷积、深度可分离卷积中的 depthwise 卷积。`groups` 直接进入 `(Cin / groups)`，因此 depthwise 卷积不会被错误统计为 full convolution。

### 3.2 ConvTranspose2d

对于反卷积层，脚本以输入位置展开到输出通道和卷积核为近似：

```text
MACs_conv_transpose2d = numel(input) * (Cout / groups) * Kh * Kw
```

这个公式刻画每个输入元素对输出特征图的核展开贡献。对于本文目的，即比较同一网络族不同宽度和 FP/QAT 路径，该近似具有一致性。

### 3.3 Linear

对于线性层，脚本统计：

```text
MACs_linear = numel(output) * in_features
```

在当前三个 SpikingLETNet shallow 变体中，线性层主要位于 transformer 模块内。由于 transformer 前向中存在 `mean(0)`、token 展开、attention 和 MLP 等 dense 计算，脚本默认将 `Linear` 归为 dense MAC，而不按 SOP 处理。

## 4. SOPs 的统计

SOPs 的关键问题是如何解释 QIFNode 的输出。当前项目中的 `QIFNode` 通过 `Quant` 将膜电位量化为 `0..T` 范围内的整数。我们采用累计 spike 口径：

```text
activation = k  等价于该位置发放 k 个脉冲
```

因此，一个激活值是否非零不足以描述事件数。若仅用 `activation != 0` 计 1 次，会低估高强度激活位置的突触活动。脚本使用输入激活的非负数值和作为 spike 事件数：

```text
input_spike_sum = sum(clamp(input_activation, min=0))
mean_spikes_per_input = input_spike_sum / numel(input_activation)
```

对于被判定为脉冲路径的卷积或反卷积层，SOPs 由 dense MACs 按输入平均脉冲数缩放得到：

```text
SOP_layer = (dense_MAC_layer / actual_timesteps) * mean_spikes_per_input
```

这个表达式可以理解为：`dense_MAC_layer` 在多步 SNN 张量中已经包含时间维，因此需要先除以实际时间步数得到单步连接规模；而 QIF 整数激活 `k` 已经表示跨时间累计 spike 数。若平均每个输入位置累计发放 `r` 个脉冲，则该层突触操作数约为单步 dense 连接数乘以 `r`。

脚本同时记录两个诊断指标：

```text
input_sum      = 输入累计 spike 数
input_nonzero  = 非零输入元素数
nonzero_ratio  = input_nonzero / input_elems
```

其中 `nonzero_ratio` 仅作为稀疏性诊断，不作为 SOPs 主计算口径。

## 5. Dense 与 Spike 路径的混合分类

SpikingLETNet 并不是所有层都应按 SOPs 计算。脚本在每个可计算层的 forward hook 中检查实际输入张量，并根据输入语义分类：

```text
若输入为非负整数，且数值范围在 [0, T] 内，且该层不是 transformer/linear：
    classification = spike
否则：
    classification = dense
```

被统计的可计算层包括：

```text
Conv2d
ConvTranspose2d
Linear
QLayer 包装的 Conv2d / ConvTranspose2d / Linear
```

不计入 core compute 的操作包括：

```text
BatchNorm
QIFNode 本身
pooling
shuffle
concat
reshape
repeat
interpolate
softmax 的显式非 MAC 部分
```

这样处理的原因是网络中存在混合计算路径。例如第一层输入为归一化图像，属于连续值输入；transformer 前使用 `output3.mean(0)` 消去时间维，然后进入 attention 和 MLP，属于 dense token 计算。若将这些层强行按 SOPs 计算，会把非事件驱动的计算错误归入脉冲累加。

## 6. FP 与 QAT/int4 的区分

脚本分别加载两类模型。

FP 模型使用普通 checkpoint：

```text
checkpoint/udd/.../model_best.pth
```

QAT 模型使用完整量化模型对象：

```text
QAT_checkpoint/udd/.../model_q_best_complete.pt
```

QAT 模型中的可量化层被 `QLayer` 包装。统计时 hook 注册在 `QLayer` 上，并排除内部 `.layer` 子模块，避免同一计算被重复统计。对于 QAT 路径，能耗参数使用 `energy_pj.int4`；对于 FP 路径，能耗参数使用 `energy_pj.fp`。

因此，FP 与 QAT 的差异来自两个方面：

```text
1. 操作数量差异：量化模型可能改变激活分布，从而改变 SOPs。
2. 单次操作能耗差异：int4 MAC/AC 的硬件能耗参数可不同于 FP MAC/AC。
```

这使得实验能够区分“量化降低了操作能耗”与“量化改变了脉冲活动率”两种机制。

## 7. 能耗计算

主能耗模型只计算核心算术操作：

```text
E_core = MAC_charged * E_MAC + AC_charged * E_AC
```

其中：

```text
MAC_charged = 所有 dense 分类层的 dense MACs 之和
AC_charged  = 所有 spike 分类层的 SOPs 之和
```

对于 dense 层：

```text
MAC_charged_layer = dense_MAC_layer
AC_charged_layer = 0
E_layer = dense_MAC_layer * E_MAC
```

对于 spike 层：

```text
MAC_charged_layer = 0
AC_charged_layer = SOP_layer
E_layer = SOP_layer * E_AC
```

可选扩展能耗包括 memory read、memory write 和 neuron update：

```text
E_total = E_core + E_mem_read + E_mem_write + E_neuron_update
```

脚本只有在启用 `--include-extended` 时才加入这些项。默认参数中这些项为 0，因为没有给定具体硬件存储层级、数据复用策略和神经元电路实现时，强行加入固定值会产生伪精确结论。

## 8. 输出字段解释

逐层结果保存在 `layer_energy.csv`，主要字段如下：

```text
dense_macs_total      该层按 dense 执行时的理论 MACs
dense_macs_charged    实际纳入 MAC 能耗的 MACs
sop_total             该层按 spike 路径估计的 SOPs
ac_charged            实际纳入 AC 能耗的 SOPs
input_sum             输入累计 spike 数或连续输入正值和
input_nonzero         非零输入元素个数
mean_spikes_per_input input_sum / input_elems
nonzero_ratio         input_nonzero / input_elems
classifications       该层在前向中被判定为 dense 或 spike 的次数
core_energy_pj        核心计算能耗
total_energy_pj       core + optional extended 能耗
```

模型级结果保存在 `summary_energy.csv` 和 `summary_energy.json`，用于比较不同模型变体和 checkpoint 类型。

## 9. 方法边界与局限

本文方法是理论估计，不是板级功耗测量。它适合用于网络结构、量化路径和脉冲活动率之间的相对比较，但不应直接等同于真实芯片或 FPGA 的端到端能耗。

主要局限包括：

```text
1. 默认不计 memory hierarchy、cache/SRAM/DRAM 访问差异。
2. 默认不计 QIF 膜电位更新、reset、leak 的电路级开销。
3. softmax、LayerNorm、插值等非卷积主算子未完整建模。
4. ConvTranspose2d MACs 使用一致性近似，适合相对比较，不代表所有硬件实现。
5. SOPs 估计依赖数据分布，因此必须报告 processed_batches 和 split。
```

因此，正式报告中应同时给出：

```text
processed_batches
input_size
T
checkpoint
energy_pj 参数
dense_macs_total
MAC_charged
AC_charged
core_energy_pj
```

这样才能保证实验可复现，并允许其他研究者替换硬件参数后复算结果。

## 10. 推荐使用方式

默认 20 batch 估计：

```bash
python tools/estimate_spikingletnet_energy.py \
  --energy-config tools/energy_params.example.yaml \
  --output-dir energy_estimates
```

完整验证集估计：

```bash
python tools/estimate_spikingletnet_energy.py \
  --energy-config tools/energy_params.example.yaml \
  --max-batches 0 \
  --output-dir energy_estimates_full
```

若已有目标硬件的操作能耗，应复制 `tools/energy_params.example.yaml` 并替换其中的 `fp.mac`、`fp.ac`、`int4.mac`、`int4.ac`。只有在存储访问模型明确时，才建议启用 `--include-extended` 并设置 memory/neuron 相关参数。

## 11. Spike sparsity 统计

对于 `QIFNode` 输出的整数激活 `k`，本文将其解释为在 `T` 个二值时间槽中发放了 `k` 个 spike。因此 spike sparsity 使用时间展开后的定义：

```text
expanded_firing_rate = sum(k) / (T * numel(k))
expanded_spike_sparsity = 1 - expanded_firing_rate
```

这个指标不同于 `count(k == 0) / numel(k)`。后者只表示聚合整数激活为 0 的比例，不能区分 `k=1` 和 `k=7` 对时间展开后脉冲密度的影响。因此，进行 SOPs 和事件驱动硬件分析时，应优先使用 `expanded_spike_sparsity`。

基于 UDD6 validation 前 20 张图像、`T=8`、输入尺寸 `400x400` 的 FP/INT4 统计结果已经写入：

```text
energy_estimates_mac_sop_20b/spike_sparsity_summary.md
energy_estimates_mac_sop_20b/spike_sparsity_summary.csv
```

这份结果只聚合被混合规则分类为 `spike` 的层，排除了原始图像输入、transformer dense token、Linear 等 dense 路径。

