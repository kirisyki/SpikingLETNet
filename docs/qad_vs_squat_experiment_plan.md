# QAD 与直接训练 QAT+SQUAT 的量化路线对比实验方案

> 状态：待书面审批，未实现、未启动训练  
> 制定日期：2026-08-02  
> 目标模型：`SpikingLETNet_shallow_max`  
> 数据集：UDD 六类语义分割

## 1. 实验问题与结论边界

本实验比较两条完整量化路线在同一任务上的最终性能：

1. **历史 QAD 路线**：基于 Integer-LIF/QIF 的训练方案，在 ANN/QIF 域完成
   W4A4 QAD 后无缝转换为 SNN。
2. **直接 SNN 路线**：从头训练真实时序 LIF-SNN，再使用权重量化感知训练
   与 SQUAT 状态量化感知训练得到 W4M4S1 模型。

主问题是：

> 在各自完整训练方案下，直接训练的 QAT+SQUAT 路线与历史 QAD 路线在 UDD
> 验证集上的 mIoU 相差多少？

这是**路线级、单随机种子、历史结果对新实验**的比较，不是只改变训练域的因果
消融。最终报告不得把差距完全归因于“ANN 上训练”或“SNN 上训练”，也不得声称
具有跨随机种子的统计显著性。

## 2. 冻结参考与禁止事项

### 2.1 历史参考

- FP32 QIF 参考检查点：
  `checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth`
- FP32 QIF 参考 mIoU：**0.678502**
- QAD W4A4 检查点：
  `QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260612-135249/model_q_best.pth`
- QAD W4A4 参考 mIoU：**0.619547**
- 机器可读历史结果：
  `quantization_comparison_results/seed1234/comparison.json`

### 2.2 明确禁止

- 不重新训练、微调、校准或转换历史 QAD 模型。
- 不重新验证 QIF 代理与 binary-expanded SNN 的等价性；该等价性已由项目此前
  多次实验和硬件平台确认，本实验将其作为既定前提。
- 不重新运行 QAD 完整验证；最终比较直接复用冻结的 `0.619547`。
- 不修改以下现有实现：
  - `Network/QAT_snn_STE.py`
  - `Network/quantization/`
  - `Network/quantization_comparison/`
  - `Network/QAT_snn_ternary.py`
  - `Network/model/SpikingLETNet_shallow_max.py`
- 不写入 `checkpoint/`、`QAT_checkpoint/`、`ternary_QAT_checkpoint/` 或任何既有
  实验输出目录。

新流水线不得提供 QAD 训练入口。它只读取历史 JSON 中的冻结指标，并在 manifest
中记录来源。

## 3. 文献方法与项目适配

直接 SNN 量化采用 SQUAT（Stateful Quantization-Aware Training）。SQUAT 在普通
权重 QAT 之外，将 LIF 神经元跨时间保存的膜电位纳入训练期量化，并使用 STE
传播状态量化梯度。

主要参考：

- [SQUAT: Stateful Quantization-Aware Training in Recurrent Spiking Neural Networks](https://arxiv.org/abs/2404.19668)
- [snnTorch `state_quant` 参考实现](https://snntorch.readthedocs.io/en/latest/_modules/snntorch/functional/quant.html)
- [Brevitas QuantConv2d 权重量化说明](https://xilinx.github.io/brevitas/v0.12.1/tutorials/quant_tensor_quant_conv2d_overview.html)

正式代码不新增 snnTorch 或 Brevitas 运行依赖，而是在独立目录中复现已确认的
公式和公开参考实现语义。固定输入的参考等价性测试必须先通过。

### 3.1 项目适配及其限制

SQUAT 原论文使用分类网络和 spike-count readout；本实验将其适配到 UDD 像素级
语义分割：

- 原模型中的每个 QIF 原位替换为真实逐时刻 LIF。
- 在现有最终六类分类卷积后新增一个输出 LIF。
- 最终六类输出在 8 个时间步内的 spike count 作为像素级 logits。
- 除神经元替换和新增输出 LIF 外，不重写网络拓扑。
- 原模型中的时间平均、FP32 softmax attention、LayerNorm、ReLU MLP、sigmoid
  门控和双线性插值全部保留。

因此该模型是保持原 `SpikingLETNet_shallow_max` 混合拓扑的直接训练 SNN，不能
描述为端到端纯 spike-driven 或端到端全低比特网络。

## 4. 实验矩阵

主表只有两条最终路线：

| 路线 | 神经元/状态 | 权重 | 输出 | 指标来源 |
|---|---|---|---|---|
| 历史 QAD | QIF，`T_time=1`，整数码 `{0,...,8}` | 历史 W4 | 历史连续输出 | 冻结结果 `0.619547` |
| QAT+SQUAT | 真实 LIF，`T=8`，M4 状态、S1 spike | W4 | 8 步 spike count | 新完整验证 |

SQUAT 路线包含两个顺序阶段，但 FP32 LIF-SNN 只作为必要起点和诊断基线，不作为
第三个竞争方法：

```text
seed 1234 随机初始化
        ↓
T=8 FP32 LIF-SNN 训练
        ↓
最佳 FP32 LIF-SNN 检查点
        ↓
同时启用 W4 权重 QAT 与 M4 状态 SQUAT
        ↓
最终 W4M4S1 QAT+SQUAT
```

不得使用历史 ANN/QIF 检查点初始化 FP32 LIF-SNN，否则直接 SNN 路线会变成
ANN 预训练后的 SNN 微调。

## 5. LIF 与状态量化语义

### 5.1 时间步与输入编码

- 训练和评估均使用真实 `T=8`。
- UDD 静态图像在 8 个时间步直接重复输入，不做泊松编码。
- 所有 LIF 必须逐时刻递推；禁止把完整 `[T,B,...]` 张量一次送入 single-step
  神经元。
- 每个 batch 前将所有膜电位清零；一个 batch 的 8 个时间步之间保留状态。
- 对完整 8 步执行 BPTT，不截断时间梯度。

采用 `T=8` 是为了与 QIF 上限 `D=8` 的虚拟二值脉冲预算保持可解释的对应关系；
两者仍不是逐值数学等价。

### 5.2 神经元参数

所有内部 LIF 与新增输出 LIF 使用同一固定参数：

| 参数 | 值 |
|---|---:|
| leak `beta` | 0.5 |
| threshold `theta` | 1.0 |
| reset | subtractive soft reset |
| reset delay | 开启，遵循 snnTorch `Leaky` 默认语义 |
| initial membrane | 0 |
| spike | `{0,1}` |
| surrogate | threshold-shifted arctangent |
| surrogate `alpha` | 2.0 |
| learnable beta/threshold | 否 |

单步参考顺序为：

1. 根据上一时刻量化膜电位计算延迟复位信号。
2. `u_candidate = beta * u_prev + input - reset_prev * theta`。
3. FP32 阶段直接保存 `u_candidate`；SQUAT 阶段将其量化为 `u_q`。
4. 根据当前保存的膜电位与 `theta` 产生二值 spike。

状态复位判定遵循参考实现的 detached reset 语义；spike 阶跃函数使用 arctangent
surrogate 反向传播。

### 5.3 M4 阈值中心状态量化

正式 SQUAT 使用 4-bit、16 级、阈值中心的非均匀状态量化：

| 参数 | 值 |
|---|---:|
| `num_bits` | 4 |
| `uniform` | `False` |
| `thr_centered` | `True` |
| `threshold` | 1.0 |
| `lower_limit` | 0.0 |
| `upper_limit` | 0.2 |
| 状态范围 | `[-1.0, 1.2]` |
| exponential multiplier | 0.5 |

前向把膜电位映射到距离最近的合法状态级；超范围值映射到最近端点。反向采用：

```text
d(u_q) / d(u) = 1
```

论文正文关于动态上下界的描述与公开 snnTorch 参考实现的固定相对阈值范围存在
差异。本实验预注册采用公开代码语义，并在最终报告披露该差异。

## 6. W4 权重量化语义

量化全部 71 个 `Conv2d`、`Linear` 和 `ConvTranspose2d`，无首尾层豁免：

```text
qmin = -7
qmax = +7
scale = max(abs(W_fp)) / 7
W_code = clamp(round(W_fp / scale), -7, +7)
W_q = W_code * scale
```

- 粒度：signed symmetric per-tensor。
- scale 由当前 FP32 shadow weight 计算并停止梯度。
- 前向使用 `W_q`；反向通过 STE 更新 `W_fp`。
- bias 保持 FP32。
- 不使用现有 QAD、STE、LSQ、EWGS 或 ternary 量化器。
- 不对普通输入/输出张量添加额外 activation fake quant；S1 spike 与 M4 state 是
  SQUAT 路线的激活/状态低比特表示。

`[-7,7]` 与历史 QAD 的 `[-8,7]` 不同，属于两套完整方法自身的量化语义，最终
报告必须并列披露。

## 7. 输出、损失与模型选择

### 7.1 输出

新增输出 LIF 产生 `[T,B,6,H,W]` 二值 spike，沿时间维求和得到：

```text
logits_count.shape = [B,6,H,W]
logits_count ∈ {0,1,...,8}
```

训练使用总 spike count，不使用时间平均；评估 argmax 对总和与平均相同。

### 7.2 损失

- 使用项目现有像素级 `CrossEntropyLoss2d`。
- 不使用 QAD 教师、特征蒸馏或其它辅助损失。
- FP32 LIF-SNN 与 QAT+SQUAT 使用相同 spike-count 任务损失。

### 7.3 检查点选择

- 每个 epoch 后执行完整 UDD validation。
- 按六类 mIoU 选择阶段最佳检查点。
- FP32 最佳检查点是 SQUAT 阶段唯一合法起点。
- 同 mIoU 时保留更早的检查点，避免结果依赖覆盖顺序。

当前项目没有独立 test split；validation 同时用于模型选择和最终报告。该限制必须
写入结果报告。

## 8. 数据、随机性与评估协议

### 8.1 数据

- 数据集：UDD。
- 输入大小：`400x400`。
- 训练列表：现有 `train_patches.txt`。
- 验证列表：全部 `val_patches.txt`，8,478 条记录。
- 保留现有 random scale 与 random mirror 数据增强。

### 8.2 随机性

- 唯一正式 seed：1234。
- 固定 Python、NumPy、PyTorch CPU 和 CUDA RNG。
- `cudnn.deterministic=True`。
- DataLoader worker 使用从正式 seed 派生的确定性 seed。
- checkpoint 保存并恢复所有 RNG 状态、optimizer、scheduler、epoch、best metric
  和采样器进度。

### 8.3 最终评估

- SQUAT 最终检查点在全部 8,478 个验证样本上评估。
- validation batch size：20。
- `T=8`。
- 六类 mIoU 来自一个全局累计 confusion matrix。
- 同时报告 per-class IoU 与 pixel accuracy。

## 9. 优化器与训练预算

### 9.1 优化器

共同设置：

| 参数 | 值 |
|---|---:|
| optimizer | Adam |
| betas | `(0.9, 0.999)` |
| eps | `1e-8` |
| weight decay | `1e-4` |
| scheduler | cosine annealing，每 optimizer step |
| eta_min | `1e-6` |
| warmup | 无 |
| AMP | 关闭 |
| gradient clipping | 默认关闭 |

阶段设置：

| 阶段 | 初始 LR | 原始预算 |
|---|---:|---:|
| FP32 LIF-SNN | `1e-3` | 300 epochs |
| QAT+SQUAT | `3e-4` | 127 epochs |

SQUAT 使用 127 epoch，与历史 QAD 实际完成的 epoch 0--126 对齐，避免再次出现
历史 QAD 与极短基线比较的问题。

### 9.2 batch size

- 目标有效 batch size：64。
- 若 `T=8` 的物理 batch 64 OOM，则降低物理 batch 并使用梯度累积。
- 调整后必须保持每 epoch 的样本覆盖、有效 batch 和 optimizer update 口径一致。
- 最后一个不完整累积组按真实样本数归一化，不丢弃训练样本。

### 9.3 数值异常

每个 optimizer step 检查 loss、梯度和参数是否为有限值。出现 NaN/Inf 时：

1. 保存独立诊断快照和最后一个正常 checkpoint。
2. 终止当前正式 run。
3. 不在同一 run 中临时修改 LR、量化范围或梯度裁剪后继续。

任何恢复策略必须形成书面 amendment 并再次审批。

## 10. 48 小时预算备选方案

正式训练前仅运行不可报告的 correctness/performance preflight：

1. 排除 CUDA 初始化后的若干 warm-up batch。
2. 分别测量 FP32 和 SQUAT 的稳定训练 batch 时间。
3. 测量验证 batch 时间并按 8,478 个样本外推。
4. 把每 epoch 完整验证计入阶段 wall-clock 估计。

完整预算预计时间：

```text
H_full = 300 * t_fp32_epoch + 127 * t_squat_epoch
```

执行规则：

- 若 `H_full < 48h`，运行 300 + 127 epochs。
- 若 `H_full >= 48h`，以约 44 小时为目标，保留约 4 小时波动余量：

```text
s = 44 / H_full
E_fp32 = floor(300 * s)
E_squat = floor(127 * s)
```

- 最低预算为 FP32 100 epochs、SQUAT 40 epochs。
- 若最低 `100+40` 仍预计达到或超过 48 小时，停止并向用户报告，不启动正式训练。
- 缩短预算时，cosine 周期按实际 optimizer step 数同步缩短。
- 最终报告保存测速样本数、原始计时、估算公式、比例 `s` 和实际 epoch。

preflight 只用于正确性、显存和时间估计，不参与指标、不产生可选最佳模型。

## 11. 独立目录与依赖边界

计划新增：

```text
Network/squat_comparison/
├── __init__.py
├── protocol.py
├── state_quantizer.py
├── lif_neuron.py
├── weight_quantizer.py
├── model_factory.py
├── training.py
├── train_fp32.py
├── train_squat.py
├── evaluate.py
├── audit.py
└── pipeline.py

tests/squat_comparison/
├── test_state_quantizer.py
├── test_lif_dynamics.py
├── test_weight_quantizer.py
├── test_model_factory.py
├── test_checkpointing.py
└── test_evaluation.py

squat_experiment_outputs/
└── seed1234/
    ├── fp32_snn/
    ├── w4m4s1_squat/
    ├── evaluation/
    └── manifest.json
```

允许只读复用：

- 现有数据集 builder 与 UDD dataset。
- 现有 loss 与 confusion-matrix/metric 逻辑。
- 现有 `SpikingLETNet_shallow_max` 宏观拓扑及其非神经元子模块。

`model_factory.py` 必须创建独立模型副本，原位替换副本中的 QIF，并在副本分类器
末尾附加输出 LIF；不得修改原始模型实例或源文件。

所有输出目录默认拒绝覆盖非空目录。resume 只能写回 checkpoint 自身所属的新
SQUAT run 目录。

## 12. 实现前测试门

正式 preflight 前必须通过：

1. M4 精确生成 16 个参考状态级，范围和 multiplier 正确。
2. 最近状态映射、端点 clipping 和相同距离 tie 行为与参考实现一致。
3. M4 backward 是恒等 STE。
4. LIF 单步 charge、delayed subtractive reset、quantize、fire 顺序正确。
5. `T=8` forward 是真实递推；构造反例证明它不同于时间维并行 single-step。
6. 每个 batch 前状态清零，batch 内 8 步状态连续。
7. arctangent surrogate 的前向 spike 与解析/数值梯度检查通过。
8. W4 码本严格位于 `[-7,7]`，scale、clamp 与 STE 正确。
9. bias 始终保持 FP32。
10. 恰好 71 个目标权重层被量化，首尾层均包含。
11. 模型副本中的全部 QIF 被替换，且只新增一个输出 LIF。
12. 原始模型结构、参数值和 `state_dict` 键在 factory 调用后不变。
13. 输出 spike count 的形状为 `[B,6,H,W]`，取值为整数 `0..8`。
14. confusion matrix 小样本结果与手算一致。
15. checkpoint resume 恢复 optimizer、scheduler、RNG 和 best metric。
16. pipeline 图中不存在 QAD train/eval/convert 节点。
17. 任意旧输出目录和 `QAT_checkpoint/` 写入尝试被拒绝。

然后执行不进入结果的集成 smoke test：

- 一个 FP32 train batch + backward；
- 一个 SQUAT train batch + backward；
- 一个 validation batch；
- resume 后下一步 LR、输出和参数更新与不中断运行一致。

## 13. 审计与输出

每次正式 run 保存：

- 完整 CLI 参数与解析后的 protocol。
- Git commit、working-tree 状态和关键只读文件哈希。
- seed、软件/CUDA/cuDNN/GPU 信息。
- 数据列表哈希与样本数。
- QIF 替换数、LIF 数、量化目标数。
- W4 code/scale/饱和率与非法码计数。
- M4 每层状态占用、端点 clipping 率与阈值附近状态占比。
- 每层 firing rate。
- 输出全零像素率、最大 spike-count 并列率和每类平均输出 count。
- 每 epoch train loss、validation mIoU、per-class IoU、LR、耗时和峰值显存。
- 理论量化权重存储量、FP32 bias/BN/连续模块参数量。
- 最终完整 confusion matrix 与 machine-readable JSON。

PyTorch wall-clock throughput 仅描述当前软件实现，不解释为神经形态硬件速度或
能耗。FP32 softmax、sigmoid、插值等混合操作必须在资源报告中单列。

## 14. 主指标与解释规则

主差值：

```text
delta_route = mIoU_squat_new - 0.619547
```

预注册解释：

| 条件 | 单次实验描述 |
|---|---|
| `delta_route > +0.005` | SQUAT 路线精度更高 |
| `delta_route < -0.005` | QAD 路线精度更高 |
| `abs(delta_route) <= 0.005` | 两条路线精度相当 |

辅助报告：

- `mIoU_squat - mIoU_fp32_lif`：SQUAT 路线自身的量化损失。
- `0.619547 - 0.678502`：历史 QAD 路线的量化损失。
- 六类 IoU 与两个路线的逐类差值。
- 训练时长、峰值显存、firing rate 和理论存储。

`±0.005` 是实际解释带，不是置信区间或统计显著性阈值。

## 15. 最终报告的强制限制声明

最终结论旁必须同时写明：

1. QAD 是历史冻结结果，SQUAT 是本次新训练结果。
2. 只有 seed 1234，不支持统计显著性或跨 seed 稳健性结论。
3. 两条路线的神经元、状态/激活表示、蒸馏、码本和优化器不同。
4. QAD 使用历史 feature distillation；SQUAT 不使用教师。
5. SQUAT 保留原模型中的 FP32 连续模块，是混合 SNN。
6. SQUAT 采用公开参考代码的固定状态范围，论文正文存在不同表述。
7. validation 同时承担模型选择与最终报告，没有独立 test set。
8. 若触发 48 小时缩放，结果只能描述缩短预算下的路线表现。
9. 不把 PyTorch fake quantization 的 wall-clock 当作低比特硬件加速证据。

## 16. 审批门

在本文件获得用户明确批准前，不得：

- 创建 SQUAT 实现文件；
- 运行 smoke/preflight；
- 启动 FP32 LIF-SNN 或 QAT+SQUAT 训练；
- 创建正式实验输出目录。

批准后按以下顺序执行：

1. 新增独立实现与单元测试。
2. 通过所有静态、数值和隔离测试。
3. 运行不可报告的 smoke test。
4. 运行时间/显存 preflight 并应用 48 小时规则。
5. 训练 FP32 LIF-SNN。
6. 从其最佳检查点训练 QAT+SQUAT。
7. 执行完整验证、审计和最终报告。

