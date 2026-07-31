# SpikingLETNet_shallow_max W4A4 量化训练对比实验方案

## 文档状态

- 状态：**等待人工审查，禁止执行**
- 编写日期：2026-07-31
- 目标模型：`SpikingLETNet_shallow_max`
- 数据集：UDD
- 目标精度：W4A4
- 对比对象：现有 QAD、STE-QAT、LSQ-QAT、EWGS-QAT
- 当前阶段允许的项目改动：仅新增本方案文档

只有在用户完整审查本报告并明确批准后，才能进入代码实现、smoke test
或正式训练阶段。批准前不得创建本报告所规划的 Python 文件，不得启动 GPU
训练，也不得改动现有 QAD、模型、数据集或评估代码。

---

## 1. 执行摘要

本实验用于检验现有 QAD（Quantization-Aware-Distillation）对
`SpikingLETNet_shallow_max` 进行 W4A4 量化训练时的有效性。

实验复用现有 QAD 最佳检查点，不重新训练 QAD。新增三个对比模型：

1. **STE-QAT**：使用与 QAD 相同的 W4A4 前向量化方式，但只优化分割任务损失，
   不使用教师模型或中间特征蒸馏。这是隔离 QAD 蒸馏收益的核心消融。
2. **LSQ-QAT**：通过任务损失学习每个量化器的步长，代表可学习量化尺度方案。
3. **EWGS-QAT**：利用量化误差对反向梯度进行逐元素缩放，代表改进梯度估计方案。

工程实现遵循低耦合原则：

- 不修改任何现有 QAD 文件。
- 不修改目标模型、数据构建器、数据集实现或现有评估脚本。
- 所有新算法、训练入口、评估入口、流水线和测试均放在新文件中。
- 新产物写入独立目录，不覆盖 `QAT_checkpoint/` 或现有报告。
- 依赖方向只能是“新代码读取/调用现有稳定接口”；现有代码不得反向依赖新代码。

---

## 2. 研究问题与结论边界

### 2.1 核心研究问题

在模型、FP32 起点、数据、训练外层超参数、量化位宽和随机种子均受控时：

1. QAD 是否优于不带蒸馏的普通 STE-QAT？
2. QAD 与 LSQ-QAT、EWGS-QAT 相比是否具有竞争力？
3. 不同方法的总体 mIoU 差异来自哪些语义类别？

### 2.2 本实验能够支持的结论

- 在 seed 1234 下，QAD 相对 STE-QAT 的受控单次实验收益。
- 在相同实验协议下，QAD 相对 LSQ-QAT 和 EWGS-QAT 的描述性竞争力。
- 各方法在六个 UDD 类别上的误差差异。

### 2.3 本实验不能支持的结论

- 统计显著性或跨随机种子的稳定性结论。
- QAD 对其他模型、数据集、时间步或位宽的普遍有效性。
- 实际 INT4 硬件吞吐、功耗或部署收益。
- “所有量化训练算法中最优”等超出三个基线范围的结论。

原因是现有 QAD 仅有 seed 1234，且本方案明确不重新训练 QAD。

---

## 3. 已核验的现有实验事实

### 3.1 QAD 实现

现有 QAD 训练入口：

`Network/QAT_snn_STE.py`

现有 W4A4 量化器：

`Network/quantization/int4_selfbuild.py`

当前 QAD 的主要行为是：

- 从 FP32 检查点构建冻结教师模型。
- 构建可训练的 W4A4 学生模型。
- 任务损失为 UDD 六类语义分割交叉熵。
- 蒸馏损失为六个中间特征对之间的 MSE 之和。
- 总损失为：

  \[
  L_{\text{QAD}} = L_{\text{task}} + 0.1 L_{\text{feature-KD}}
  \]

- 权重和普通激活量化路径使用 signed W4A4，整数 clamp 范围为 `[-8, 7]`。
- 当前 W4A4 量化覆盖所有 `Conv2d`、`Linear` 和 `ConvTranspose2d`。
- 当前模型共识别出 71 个可量化层。
- 权重尺度为每层 per-tensor 动态尺度。
- 激活尺度为每次前向的 per-tensor 动态尺度。
- 舍入操作通过 STE 传递梯度。

现有激活量化器还有一项必须保留并披露的历史行为：如果某层的整个输入 tensor
已经是整数，且所有值均位于 `[-8, 8]`，代码会跳过重新量化。因此某些脉冲计数
输入可能保留 `+8`，而不是进入普通路径的 `[-8, 7]` clamp。该行为属于现有
QAD 的量化语义，本实验不得回改。STE-QAT 将通过只读复用原量化器继承同一行为；
最终审计会统计整数旁路及 `+8` 的实际触发情况。

### 3.2 现有 QAD 参考模型

FP32 起点：

`checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth`

QAD W4A4 最佳检查点：

`QAT_checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260612-135249/model_q_best.pth`

统一完整验证结果：

- mIoU：`0.619547`
- background：`0.490170`
- facade：`0.576836`
- road：`0.595894`
- vegetation：`0.874599`
- vehicle：`0.398562`
- roof：`0.781223`

### 3.3 训练轮数不一致及处理方式

QAD 参数记录声明 `max_epochs=150`，但实际日志只包含 epoch 0 到 126，
即共完成 127 个 epoch。最佳检查点记录的 epoch 为 119，对应日志中的
epoch 118。

本方案不重新训练 QAD，因此对比模型采用以下协议：

- 学习率调度总长度仍按 `150 × 401` iterations 计算。
- 正式训练在完成 127 个 epoch 后停止。
- 每个 epoch 均执行验证并保留此前最佳检查点。

这使对比模型拥有与现有 QAD 相同的实际 epoch 预算和相同形状的学习率曲线，
避免额外获得 23 个 epoch 的训练及择优机会。

### 3.4 现有证据的固有限制

现有 QAD 并不是与新增基线在同一次统一流水线中重新生成的结果。因此最终报告
必须明确写出：

- QAD 是历史检查点。
- 三个基线是本方案新增运行。
- 所有方法使用相同 FP32 起点和目标协议，但 QAD 与基线不属于同批重跑。
- 单种子结果不能用于统计显著性推断。

---

## 4. 对比算法选择

### 4.1 STE-QAT：核心因果消融

#### 目的

隔离“中间特征蒸馏”本身的收益。

#### 前向量化

STE-QAT 直接只读复用现有
`quantization.int4_selfbuild.quantize_model`，确保其量化层、量化范围、
动态尺度和 QAD 完全一致。

#### 训练损失

\[
L_{\text{STE-QAT}} = L_{\text{task}}
\]

不构建教师模型，不计算教师前向，不计算任何 KD 损失。

#### 解释

QAD 与 STE-QAT 的主要设计差异仅为蒸馏监督，因此：

\[
\Delta_{\text{KD}} =
\text{mIoU}_{\text{QAD}} - \text{mIoU}_{\text{STE-QAT}}
\]

是本实验最重要的有效性指标。

### 4.2 LSQ-QAT：可学习步长基线

依据：

- Steven K. Esser 等，Learned Step Size Quantization，ICLR 2020：
  <https://arxiv.org/abs/1902.08153>

#### 量化形式

对权重和激活均使用：

\[
\hat{x} =
s \cdot
\operatorname{clamp}
\left(
\operatorname{round}(x/s),
-8,
7
\right)
\]

其中 `s` 是每个量化器独立的、可训练的正步长。

#### 固定算法设置

- 权重：signed、per-tensor、4 bit。
- 激活：signed、per-tensor、4 bit。
- 所有 71 个可量化层均量化。
- 不保留 FP32 或 8-bit 首尾层。
- 步长初始化采用 LSQ 论文规则：

  \[
  s_0 =
  \frac{2\,\mathbb{E}(|x|)}
  {\sqrt{Q_P}},
  \quad Q_P=7
  \]

- 激活步长在首次有效 batch 上初始化一次，此后作为参数训练。
- 步长梯度按论文规则缩放：

  \[
  g_s =
  \frac{1}{\sqrt{NQ_P}}
  \]

  其中 `N` 为当前量化 tensor 的元素数。

- 步长通过数值下界保证为正，不允许出现零或负的有效量化尺度。
- 潜在模型权重使用共同 Adam 设置；LSQ 步长使用同一基础学习率和 poly
  multiplier，但处于 `weight_decay=0` 的独立参数组。
- 不使用蒸馏。

### 4.3 EWGS-QAT：量化误差感知梯度基线

依据：

- Junghyup Lee 等，Network Quantization with Element-wise Gradient
  Scaling，CVPR 2021：
  <https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Network_Quantization_With_Element-Wise_Gradient_Scaling_CVPR_2021_paper.html>
- 官方实现：
  <https://github.com/cvlab-yonsei/EWGS>

#### 梯度规则

EWGS 对连续归一化值与离散值之差进行保存。若上游梯度为 `g`，连续值与
离散值之差为 `d`，则反向传播为：

\[
g_{\text{EWGS}}
=
g \left(1 + \delta\,\operatorname{sign}(g)d\right)
\]

其中 `δ` 为每个权重和激活量化器独立的反向缩放因子。

#### 固定算法设置

- 权重：signed-capable、per-tensor、4 bit。
- 激活：signed-capable、per-tensor、4 bit。
- 量化级数为 16。
- 所有 71 个可量化层均量化。
- 不保留 FP32 或 8-bit 首尾层。
- 权重和激活分别维护上下量化边界。
- 权重边界首次初始化为 `l_W=-3σ(W)`、`u_W=3σ(W)`。
- 激活边界在首个有效 batch 初始化为 `l_A=min(A)`、
  `u_A=3σ(A)/sqrt(1-2/π)`。
- 每个量化层维护可学习的输出尺度，首次初始化为全精度输出平均绝对值与
  量化输出平均绝对值之比。
- 边界和输出尺度使用独立 Adam 参数组，基础学习率采用官方实现的
  `1e-5`、`weight_decay=0`，并使用与模型参数相同的 poly multiplier。
- 潜在模型权重仍使用共同的 Adam `lr=1e-3` 和 `weight_decay=1e-4`。
- 每个量化器维护独立的 EWGS 缩放因子。
- 启用 Hessian 信息更新缩放因子。
- 按官方实验设置，从 epoch 1 开始，每 1 个 epoch 更新一次缩放因子。
- 每次更新使用 10 个 batch，Hessian trace 使用 Rademacher-Hutchinson
  估计，单个 trace 最多 50 次迭代、相对收敛容差 `1e-3`。
- 每个量化器的缩放因子按平均 Hessian trace、tensor 元素数和
  `3 × gradient std` 计算，并 clamp 到非负范围。
- 不使用固定 `δ=0.001` 的简化版本。
- 不使用蒸馏。

EWGS 的 Hessian 估计将在独立的新文件中以论文公式重新实现，不直接复制
GPL-3.0 官方仓库代码，避免向现有项目引入许可证耦合。

如果 Hessian 更新在目标网络上出现无法接受的显存占用、非有限值或不可完成的
运行时间，smoke test 将判定 EWGS 未通过执行门。此时停止并提交诊断，不得擅自
切换到固定缩放因子版本。

### 4.4 未选择的算法

#### PACT / DoReFa-Net

未选原因：

- 现有激活统计显示模型同时包含非负脉冲输入和明显有符号的 Transformer、
  跳连及分类层输入。
- 原始 PACT/DoReFa 的非负或归一化激活假设不完全满足。
- 若改造成 bilateral/signed 版本，将引入非原始算法变体，方法身份与公平性
  更难解释。

#### PTQ

PTQ 不属于量化训练算法，无法直接回答 QAD 相对其他训练方案的有效性问题，
因此不纳入主比较。

---

## 5. 低耦合工程架构

### 5.1 不得修改的现有文件和目录

以下文件/目录在本实验中全部视为只读：

- `Network/QAT_snn_STE.py`
- `Network/QAT_snn.py`
- `Network/quantization/int4_selfbuild.py`
- `Network/model/SpikingLETNet_shallow_max.py`
- `Network/model/`
- `Network/builders/`
- `Network/dataset/`
- `Network/evaluate_precision_comparison.py`
- `checkpoint/`
- `QAT_checkpoint/`
- `ternary_QAT_checkpoint/`

不得通过 monkey patch、运行时覆盖全局函数、修改模块全局变量等方式绕过此限制。

### 5.2 计划新增的代码文件

批准后计划新增：

```text
Network/
├── quantization_comparison/
│   ├── __init__.py
│   ├── protocol.py
│   ├── layer_adapter.py
│   ├── ste_qat.py
│   ├── lsq_qat.py
│   ├── ewgs_qat.py
│   ├── ewgs_hessian.py
│   ├── model_factory.py
│   └── checkpointing.py
├── train_quantization_baseline.py
├── run_quantization_comparison_pipeline.py
└── evaluate_quantization_comparison.py

tests/
├── test_quantization_comparison_quantizers.py
├── test_quantization_comparison_model_factory.py
├── test_quantization_comparison_protocol.py
└── test_quantization_comparison_checkpointing.py
```

文件职责：

- `protocol.py`：保存冻结的实验协议和方法枚举，拒绝运行时静默覆盖关键设置。
- `layer_adapter.py`：统一支持 `Conv2d`、`Linear`、`ConvTranspose2d`
  及 `[T,B,...]` 多步输入。
- `ste_qat.py`：只读调用现有量化器，提供无 KD 的训练模型工厂。
- `lsq_qat.py`：独立实现 LSQ 量化函数和量化层。
- `ewgs_qat.py`：独立实现 EWGS 量化函数和量化层。
- `ewgs_hessian.py`：独立实现 EWGS 缩放因子更新所需的 Hessian 估计。
- `model_factory.py`：递归替换 71 个目标层，并校验数量、名称与位宽。
- `checkpointing.py`：保存和恢复模型、优化器、epoch、最佳指标及 RNG 状态。
- `train_quantization_baseline.py`：三个基线共享的训练入口。
- `run_quantization_comparison_pipeline.py`：顺序执行测试门、smoke test、
  正式训练及评估。
- `evaluate_quantization_comparison.py`：独立构建五种模型并执行统一完整验证。

### 5.3 依赖方向

```text
新增训练/评估入口
        │
        ▼
新增 quantization_comparison 包
        │
        ├──只读调用──► 现有 model builder
        ├──只读调用──► 现有 dataset builder
        └──STE 专用──► 现有 int4 quantize_model

现有项目代码 ──X──► 新增 quantization_comparison 包
```

现有模块不注册新方法，不添加条件分支，不感知本实验的存在。

### 5.4 独立产物目录

正式运行只允许写入：

```text
quantization_comparison_checkpoint/
└── udd/
    └── seed1234/
        ├── ste_qat_w4a4/
        ├── lsq_qat_w4a4/
        └── ewgs_qat_w4a4/

quantization_comparison_results/
└── seed1234/
    ├── comparison.csv
    ├── comparison.json
    ├── report.md
    ├── run_manifest.json
    └── figures/
```

每个方法目录至少包含：

- `best_quantized.pth`
- `latest_resume.pth`
- `metrics.csv`
- `run_manifest.json`
- `stdout.log`
- `loss_curve.png`
- `miou_curve.png`

如果目标目录非空，程序默认拒绝覆盖。继续训练必须显式使用
`--resume` 并通过 manifest 一致性校验。

---

## 6. 冻结的实验协议

### 6.1 共同设置

| 项目 | 固定值 |
|---|---|
| 模型 | `SpikingLETNet_shallow_max` |
| 配置 | `Network/configs/SpikingLETNet_shallow/1.3M.yaml` |
| 数据集 | UDD |
| 输入尺寸 | 400 × 400 |
| 类别数 | 6 |
| 训练类型 | `trainval` |
| 时间步 | T=1 |
| FP32 起点 | `...20260611-003821/model_best.pth` |
| 随机种子 | 1234 |
| batch size | 64 |
| workers | 6 |
| 正式训练 epoch | 127 |
| LR 调度总 epoch | 150 |
| 每 epoch 训练 batches | 401，禁止截断 |
| 潜在模型权重优化器 | Adam |
| 潜在模型权重初始学习率 | 0.001 |
| betas | (0.9, 0.999) |
| eps | 1e-8 |
| 潜在模型权重 decay | 1e-4 |
| LR 调度 | poly |
| poly exponent | 0.9 |
| random mirror | True |
| random scale | True |
| 任务损失 | CrossEntropyLoss2d |
| AMP | 关闭，与 QAD 保持一致 |
| 位宽 | W4A4 |
| 量化层 | 全部 71 层 |
| 量化粒度 | per-tensor |
| 量化级数 | 16 |
| 有符号输入支持 | 必须支持 |
| 精确编码/边界 | 各算法按第 4 节；QAD/STE 保留历史整数旁路 |
| 首尾层豁免 | 无 |

算法固有的量化尺度、边界、梯度缩放和 Hessian 更新按第 4 节设置，不视为
违反“外层训练超参数一致”。

### 6.2 学习率

第 `i` 个全局 iteration 的学习率为：

\[
\operatorname{lr}(i)
=
10^{-3}
\left(
1-\frac{i}{150 \times 401}
\right)^{0.9}
\]

正式训练在第 `127 × 401` 个训练 iteration 完成后结束，不将调度曲线重新
压缩到 127 epoch。

实现后必须使用 QAD 历史日志中的多个学习率锚点进行单元测试，防止调度器
调用顺序导致曲线偏移。

### 6.3 模型选择

- 每个 epoch 后在与 QAD 相同的验证 loader 上执行完整验证。
- 主选择指标为六类算术平均 mIoU。
- mIoU 相同时，保留较晚 epoch，与现有 `>= max(...)` 行为一致。
- 不根据单类别 IoU 选择模型。
- 正式结果只使用 `best_quantized.pth`。

### 6.4 断点恢复

- 每个 epoch 完成后保存 `latest_resume.pth`。
- 保存模型、优化器、epoch、最佳 mIoU、Python/NumPy/PyTorch/CUDA RNG 状态。
- 仅支持 epoch 边界恢复。
- 中途失败时从最后一个完整 epoch 恢复，不允许跳过剩余 batch。
- resume 前校验模型、方法、位宽、种子、FP32 SHA256 和协议版本。

---

## 7. 验证与报告协议

### 7.1 最终数据范围

最终比较使用：

- UDD `val_patches.txt`
- 完整 8,478 个 patch
- batch size 20
- workers 6
- T=1
- `max_val_iters=0`

### 7.2 统一评估

新的独立评估脚本一次性加载并评估：

1. FP32
2. 现有 QAD
3. STE-QAT
4. LSQ-QAT
5. EWGS-QAT

所有模型共享：

- 同一个 dataset builder。
- 同一个输入预处理。
- 同一个时间维处理。
- 同一个 6×6 confusion matrix 实现。
- 同一个类别顺序。

作为评估等价性门，新的评估脚本必须先重新计算现有 QAD，结果应与
`0.619547` 在数值容差内一致。若不一致，禁止生成最终比较结论，先定位评估
协议差异。

### 7.3 输出指标

主指标：

- mIoU

辅助指标：

- background IoU
- facade IoU
- road IoU
- vegetation IoU
- vehicle IoU
- roof IoU
- 最佳 epoch
- 最佳内部验证 mIoU
- 最终完整验证 mIoU
- 每 epoch 训练损失
- 每 epoch 学习率
- 墙钟训练时间
- 峰值 GPU 显存
- LSQ 步长分布
- EWGS 边界和缩放因子分布
- QAD/STE 整数激活旁路次数及 `+8` 触发次数

---

## 8. 预注册判定标准

设：

- \(M_Q\)：现有 QAD mIoU。
- \(M_S\)：STE-QAT mIoU。
- \(M_L\)：LSQ-QAT mIoU。
- \(M_E\)：EWGS-QAT mIoU。
- \(M_B=\max(M_L,M_E)\)。

### 8.1 QAD 有效性

\[
M_Q-M_S \ge 0.010
\]

即 QAD 至少领先 STE-QAT 1.0 个绝对 mIoU 百分点。

### 8.2 QAD 竞争力

\[
M_Q \ge M_B-0.005
\]

即 QAD 与最佳高级基线差距不超过 0.5 个百分点。

### 8.3 QAD 优越性

\[
M_Q-M_B \ge 0.005
\]

即 QAD 至少领先最佳高级基线 0.5 个百分点。

### 8.4 结论规则

- 阈值在正式训练前冻结。
- 不允许根据结果调整阈值。
- 不允许用个别类别优势替代总体 mIoU 结论。
- 未达阈值时必须写为“当前证据不支持相应主张”。
- 所有差值同时报告绝对 mIoU 和绝对百分点。
- 单种子结果不得使用“统计显著”等措辞。

---

## 9. 测试与执行门

### 9.1 静态和单元测试

正式训练前必须通过：

1. 三种工厂均精确替换 71 个量化层。
2. 目标层名称集合完全一致。
3. LSQ/EWGS 所有量化输出不超出 16 个 level。
4. 所有权重和激活量化器均支持有符号输入和 W4A4。
5. 首尾层不存在 FP32/8-bit 豁免。
6. STE-QAT 与现有 QAD 量化器在相同模型和输入上的前向输出逐元素一致。
7. STE/QAD 的整数旁路与 `+8` 触发行为被测试并记录。
8. STE-QAT 不实例化教师模型，不包含 KD 参数或 KD loss。
9. LSQ 步长初始化、正值约束和梯度缩放符合公式。
10. EWGS 在 `δ=0` 时反向梯度退化为 STE。
11. EWGS 梯度与手工公式的小 tensor 结果一致。
12. EWGS Hessian trace 和缩放因子用小 tensor 数值例验证。
13. `Conv2d`、`Linear`、`ConvTranspose2d` 均可前向和反向。
14. `[T,B,C,H,W]` 与 `[T,B,D]` 输入形状可以正确恢复。
15. checkpoint 保存后重载得到相同输出。
16. 断点恢复的下一 epoch 学习率和 RNG 状态连续。
17. 新评估器可以复现现有 QAD mIoU。
18. `git diff` 显示没有任何受保护现有文件被改动。

### 9.2 Smoke test

每个方法依次执行受限 smoke test：

- seed 1234
- 使用真实 FP32 检查点和真实 UDD loader
- 少量 train batches
- 少量 validation batches
- 至少完成一次 optimizer step
- 至少完成一次保存和恢复

通过条件：

- loss、输出、梯度和量化参数均无 NaN/Inf。
- 潜在 FP32 权重存在非零梯度。
- LSQ 步长存在有限非零梯度。
- EWGS 缩放因子更新成功且无显存溢出。
- EWGS 完整一次缩放因子更新的实测耗时可以外推。
- 模型 reset 后状态不会跨 batch 泄漏。
- 恢复后的前向输出与保存前一致。

smoke test 不用于比较 mIoU，也不允许据此选择表现更高的内部超参数。

### 9.3 正式训练门

只有以下条件全部满足才可启动正式训练：

- 本方案已获用户明确批准。
- 所有单元测试通过。
- 三个 smoke test 全部通过。
- 受保护文件 hash 未变化。
- 三个正式输出目录为空或不存在。
- GPU 环境和数据集完整。
- 每个方法的预运行 manifest 已生成并通过一致性校验。

任一门失败时停止，不自动降级算法或更换实验口径。

由于官方 EWGS 的 Hessian 协议需要对权重和激活量化器逐一估计 trace，若
smoke test 外推的 EWGS 单次完整更新不可完成，或完整 127-epoch EWGS 运行预计
超过 72 GPU 小时，则在 EWGS 正式训练前再次暂停并提交耗时诊断，等待用户决定。
不得自动减少 batch 数、Hutchinson 次数或更新频率。

---

## 10. 正式执行顺序

为避免单 GPU 资源竞争，按以下顺序串行执行：

1. STE-QAT，seed 1234，127 epochs。
2. LSQ-QAT，seed 1234，127 epochs。
3. EWGS-QAT，seed 1234，127 epochs。
4. 统一完整验证。
5. 生成 CSV、JSON、Markdown 报告和训练曲线。

推荐先运行 STE-QAT，是因为它最直接回答 QAD 的核心有效性问题，也最容易验证
新训练框架与原 QAD 协议的一致性。

---

## 11. 资源与时间预估

当前环境：

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition
- 显存：约 96 GB
- PyTorch：2.7.0+cu128
- CUDA：可用

历史 QAD 的 127-epoch 日志时间跨度约为 11 小时。新增框架和验证频率可能改变
耗时，因此普通 QAT 方法仅给出量级估计：

- STE-QAT：约 10–14 GPU 小时。
- LSQ-QAT：约 11–16 GPU 小时。
- EWGS-QAT：正式估时必须等待 Hessian smoke test。官方协议每轮涉及
  10 个 batch、最多 50 次 Hutchinson 迭代以及所有权重/激活量化器，
  其开销可能比普通训练高一个或多个数量级。
- 完整串行实验：在 EWGS smoke test 前无法给出可信总时长。

smoke test 后应使用真实吞吐更新 ETA，但不得据此改变训练预算或 EWGS
算法设置。若触发第 9.3 节的 72 GPU 小时门，则等待额外批准。

---

## 12. 风险与缓解措施

### 风险 1：历史 QAD 与新基线不是同批重跑

缓解：

- 固定相同 FP32 起点、seed、数据、更新预算和最终评估器。
- 在报告中显式标注历史检查点限制。
- 不声称统计显著性。

### 风险 2：EWGS Hessian 估计开销过高

缓解：

- 独立 smoke test 测量显存和耗时。
- Hessian 逻辑与训练逻辑分文件。
- 失败时停止并请求重新决策，不擅自改用固定缩放因子。

### 风险 3：LSQ/EWGS 对 SNN 时间维支持错误

缓解：

- 在统一 layer adapter 中集中处理时间维。
- 对 Conv、Linear、ConvTranspose 分别测试 T=1 和合成 T>1。
- 每个 batch 后调用 `functional.reset_net`。

### 风险 4：评估口径漂移

缓解：

- 新评估器先复现现有 QAD 的 `0.619547`。
- 所有模型在同一进程、同一 loader 和同一 confusion matrix 下顺序评估。
- 保存完整 protocol manifest。

### 风险 5：新代码意外影响现有架构

缓解：

- 只新增文件。
- 禁止 monkey patch。
- 新输出使用独立根目录。
- 测试前后核对受保护文件 hash 与 `git diff`。
- 未获额外批准不得修改现有文件以“方便接入”。

### 风险 6：单次运行中断

缓解：

- 每个 epoch 保存可恢复检查点。
- 保存优化器和 RNG 状态。
- 输出 manifest 记录已完成 epoch。
- 仅从最后完整 epoch 恢复。

---

## 13. 计划交付物

批准并完整执行后，交付：

1. 独立、低耦合的 STE/LSQ/EWGS W4A4 训练代码。
2. 量化器、模型转换、协议和 checkpoint 单元测试。
3. 三个正式最佳检查点及可恢复检查点。
4. 三套逐 epoch 指标与运行清单。
5. 五种精度模型的统一完整验证结果。
6. 最终 `comparison.csv` 和 `comparison.json`。
7. 最终中文 Markdown 报告。
8. QAD 有效性、竞争力和优越性判定。
9. 实验限制、异常和所有偏离项记录。

---

## 14. 审查清单

请在批准前重点审查：

- [ ] 是否接受不重新训练现有 QAD。
- [ ] 是否接受 STE-QAT、LSQ-QAT、EWGS-QAT 三个基线。
- [ ] 是否接受所有 71 层、per-tensor、signed-capable W4A4。
- [ ] 是否接受保留并审计 QAD/STE 的历史整数激活旁路及 `+8` 行为。
- [ ] 是否接受正式训练 127 epochs、调度总长 150 epochs。
- [ ] 是否接受仅使用 seed 1234。
- [ ] 是否接受算法内部参数按论文规则固定、不按验证结果调优。
- [ ] 是否接受 EWGS 量化边界/输出尺度使用独立 Adam `lr=1e-5` 参数组。
- [ ] 是否接受 EWGS 使用官方 Hessian 更新且失败时停止，而非自动降级。
- [ ] 是否接受 EWGS 预计超过 72 GPU 小时时再次暂停审批。
- [ ] 是否接受第 8 节预注册阈值。
- [ ] 是否接受所有实验代码只新增文件。
- [ ] 是否接受现有 QAD、模型、数据和评估文件全部只读。
- [ ] 是否接受独立输出目录和默认禁止覆盖策略。
- [ ] 是否接受单种子和历史 QAD 检查点带来的结论限制。

在上述项目全部获批前，本方案保持“等待人工审查，禁止执行”状态。
