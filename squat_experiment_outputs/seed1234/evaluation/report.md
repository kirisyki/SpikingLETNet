# QAD 与直接 QAT+SQUAT 路线对比实验总结

> 状态：**实验全部完成，结果完整性核验通过；结论可在明确限制条件下使用。**  
> 运行：seed 1234，T=8，UDD6，FP32 LIF-SNN 100 epochs + W4M4S1 SQUAT 40 epochs。  
> 完成时间：2026-08-06 19:20:00 UTC。

## 1. 执行状态

| 检查项 | 结果 | 证据 |
|---|---|---|
| tmux 流水线 | 完成 | `state=complete` |
| FP32 LIF-SNN | 100/100 epochs | epoch 连续，无缺失；最佳 epoch 83 |
| W4M4S1 SQUAT | 40/40 epochs | epoch 连续，无缺失；最佳 epoch 4 |
| 验证集覆盖 | 通过 | 每个 epoch 均为 8,478 个样本 |
| 最终评估覆盖 | 通过 | 8,478 个样本，1,356,480,000 个有效像素 |
| 历史 QAD 保护 | 通过 | 只读冻结 JSON；SHA256 与清单一致 |
| 数值一致性 | 通过 | 混淆矩阵重算的 mIoU、逐类 IoU、pixel accuracy 与保存值一致 |
| 后台进程 | 已退出 | tmux server 不再存在，GPU 无本实验计算进程 |

逐 epoch 记录汇总的累计时间约为 **81.87 小时**：FP32 阶段约 56.92 小时，SQUAT 阶段约 24.95 小时。由于服务器曾重启和续训，`training_summary.json` 中的单次进程 elapsed time 不是完整累计时长，因此本报告采用 CSV 各 epoch 时间之和。

## 2. 核心结论

**当前完整 SQUAT 配方不能替代 QAD。**

- 冻结历史 QAD mIoU：**0.619547**
- 直接 W4M4S1 QAT+SQUAT 最佳 mIoU：**0.315514**
- 路线差值（SQUAT − QAD）：**−0.304033**
- 预注册可比带：±0.005

SQUAT 比 QAD 低 **30.40 个 mIoU 百分点**，差距远大于可比带，判定为：**QAD 路线精度更高**。

更关键的是，新训练的 FP32 LIF-SNN 基线达到 **0.689462 mIoU**，而进入 W4M4S1 QAT+SQUAT 后降到 **0.315514**，量化阶段损失为 **−0.373948**。这说明当前直接 SNN 量化流程出现了严重退化，而不是 LIF-SNN 基线本身无法学习任务。

## 3. 对比口径

本实验比较的是两条完整路线：

1. **QAD 路线**：先在 ANN/QIF 代理上训练和量化，再无缝转换为 Integer-LIF SNN。此次严格读取历史冻结指标，不重新训练、评估、转换或校准。
2. **SQUAT 路线**：从随机初始化训练真实 T=8 LIF-SNN，再从最佳 FP32 checkpoint 直接执行 W4M4S1 QAT；量化误差在 SNN 时间展开和训练过程中被直接优化。

新路线保持既定网络结构，神经元改为 LIF，并在分类头后保留输出 LIF。物理 batch 为 16，梯度累积 4，有效 batch 为 64；FP32 阶段训练 100 epochs，SQUAT 阶段训练 40 epochs。

因此，本结果回答的是“**现有完整 QAD 工程路线与当前完整 SQUAT 工程路线谁表现更好**”，而不是“只改变训练域时的单因素因果效应”。

## 4. 路线级最终结果

最终结果均使用各自最佳 checkpoint。QAD 为历史冻结结果；新路线在完整验证集上重新评估。

| 路线 | 最佳 epoch | mIoU | Pixel accuracy | 说明 |
|---|---:|---:|---:|---|
| 历史 QAD（冻结） | 历史结果 | **0.619547** | 未提供 | 本次未重跑 |
| 新 FP32 LIF-SNN | 83 | **0.689462** | 0.848082 | 量化前诊断基线 |
| 新 W4M4S1 QAT+SQUAT | 4 | **0.315514** | 0.552441 | 本次直接 SNN 量化结果 |

量化损失对比：

| 路线 | 量化前 mIoU | 量化后 mIoU | 量化损失 |
|---|---:|---:|---:|
| 历史 QAD / Integer-LIF | 0.678502 | 0.619547 | **−0.058955** |
| 直接 LIF-SNN / SQUAT | 0.689462 | 0.315514 | **−0.373948** |

直接 SQUAT 的 mIoU 损失约为历史 QAD 量化损失绝对值的 **6.34 倍**。这个比值用于描述工程结果，不应被解释为严格的算法因果倍数，因为两条路线的蒸馏、码本、状态表示和优化细节并不完全相同。

## 5. 逐类 IoU

| 类别 | QAD | FP32 LIF-SNN | SQUAT | SQUAT − QAD |
|---|---:|---:|---:|---:|
| background | 0.490170 | 0.569946 | 0.271432 | −0.218738 |
| facade | 0.576836 | 0.644511 | 0.190729 | −0.386107 |
| road | 0.595894 | 0.638031 | 0.262074 | −0.333820 |
| vegetation | 0.874599 | 0.891443 | 0.777821 | −0.096778 |
| vehicle | 0.398562 | 0.552202 | **0.000002** | **−0.398559** |
| roof | 0.781223 | 0.840639 | 0.391027 | −0.390196 |

六个类别全部低于 QAD，说明退化不是少数类别的随机波动。vegetation 相对稳健，但 vehicle 几乎完全失效；facade、roof 和 road 也有大幅下降。

## 6. 训练动态

### 6.1 FP32 LIF-SNN

| Epoch | Train loss | mIoU | Pixel accuracy |
|---:|---:|---:|---:|
| 1 | 1.051477 | 0.387512 | 0.641016 |
| 25 | 0.485257 | 0.600364 | 0.799409 |
| 50 | 0.372528 | 0.664299 | 0.832580 |
| 75 | 0.304907 | 0.678890 | 0.844059 |
| **83（最佳）** | **0.291092** | **0.689526** | **0.848116** |
| 100 | 0.279719 | 0.685661 | 0.846091 |

FP32 LIF-SNN 正常收敛，最佳 checkpoint 的最终重评估 mIoU 为 0.689462，与 epoch 内记录的 0.689526 仅有正常的评估数值差异。

### 6.2 W4M4S1 SQUAT

| Epoch | Train loss | mIoU | Pixel accuracy |
|---:|---:|---:|---:|
| 1 | 1.010941 | 0.309121 | 0.559216 |
| 2 | 1.167519 | 0.307676 | 0.543855 |
| **4（最佳）** | **1.169166** | **0.315509** | **0.552441** |
| 10 | 1.348047 | 0.230863 | 0.453515 |
| 20 | 1.562349 | 0.168750 | 0.420894 |
| 30 | 1.604069 | 0.132156 | 0.356753 |
| 40 | 1.566412 | 0.130528 | 0.349507 |

SQUAT 从第 1 个 epoch 起就已显著低于 FP32 基线，并在第 4 个 epoch 后持续退化。最终路线对比使用的是**最佳 epoch 4 checkpoint**，而不是 epoch 40，因此当前差距不能归因于“误用了最后一个 checkpoint”。

## 7. 输出与量化审计

### 7.1 输出读出

| 指标 | FP32 LIF-SNN | SQUAT |
|---|---:|---:|
| 全零输出像素比例 | 0 | 0.045260% |
| 最大类并列比例 | 7.322198% | **29.732680%** |
| 最大脉冲计数 | 8 | 8 |
| background 平均计数 | 2.979410 | 1.989813 |
| facade 平均计数 | 2.384622 | 1.876667 |
| road 平均计数 | 1.731261 | 1.296132 |
| vegetation 平均计数 | 2.529837 | 1.663351 |
| vehicle 平均计数 | 0.291671 | **0.014350** |
| roof 平均计数 | 2.941824 | 2.051634 |

SQUAT 的 vehicle 平均输出计数只有 FP32 基线的约 1/20，同时最终最大类并列率从 7.32% 升到 29.73%。这与 vehicle 类输出饥饿及读出分辨率下降一致，但仍是诊断关联，不能单独证明最终输出 LIF 是唯一原因。

### 7.2 权重量化

| 指标 | 数值 |
|---|---:|
| 量化层数 | 71 |
| 量化权重元素 | 1,291,936 |
| 4-bit 理论权重存储 | 5,167,744 bit（约 0.616 MiB） |
| 非量化参数元素 | 11,124 |
| 量化层内 FP32 bias 元素 | 1,744 |
| 零码比例 | **56.465568%** |
| 聚合饱和比例 | 0.011301% |
| 权重码完全为零的量化层 | **11/71** |

完全归零的 11 个量化层为：

1. `DAB_Block_4.DAB_Module_4_0.conv1x1.conv`
2. `DAB_Block_4.DAB_Module_4_0.conv1x1_in.conv`
3. `DAB_Block_4.DAB_Module_4_0.conv1x3.conv`
4. `DAB_Block_4.DAB_Module_4_0.conv3x1.conv`
5. `DAB_Block_4.DAB_Module_4_0.dconv1x3.conv`
6. `DAB_Block_4.DAB_Module_4_0.dconv3x1.conv`
7. `DAB_Block_4.DAB_Module_4_0.ddconv1x3.conv`
8. `DAB_Block_4.DAB_Module_4_0.ddconv3x1.conv`
9. `DAB_Block_5.DAB_Module_5_0.dconv1x3.conv`
10. `DAB_Block_5.DAB_Module_5_0.dconv3x1.conv`
11. `transformer1.mlp.fc1`

全局饱和率很低，因此问题不像是简单的“所有权重都撞到 4-bit 端点”；相反，较高零码比例、局部层完全归零、输出并列率升高和 vehicle 类脉冲饥饿共同构成更值得优先检查的异常链。是否由权重量化、膜电位量化、梯度传播或输出读出中的某一项主导，仍需消融确认。

> 存储量为理论位宽估算。当前 PyTorch fake-quant 张量仍以 FP32 存储和执行，不能据此推断低比特硬件速度、显存或能耗收益。

## 8. 数据一致性核验

| 核验项 | 结果 |
|---|---|
| FP32 CSV 行数、epoch 连续性、summary 最佳值 | 通过 |
| SQUAT CSV 行数、epoch 连续性、summary 最佳值 | 通过 |
| 最终混淆矩阵重算 mIoU | 与保存值一致 |
| 最终混淆矩阵重算逐类 IoU | 与保存值一致 |
| 最终混淆矩阵重算 pixel accuracy | 与保存值一致 |
| 混淆矩阵像素数与 valid pixels | 一致 |
| 路线差值与量化损失复算 | 一致 |
| 冻结 QAD 文件 SHA256 | `67daba5d843da84b28006466b83cb303ec06d919468f4b74600732afa676e1a8`，一致 |

核验评级：**Share with caveats**。数值与执行完整性没有发现阻断问题，但单 seed 和完整路线差异限制了结论外推范围。

## 9. 解释限制

1. 只有 seed 1234，不能声称统计显著性或跨 seed 稳健性。
2. QAD 是历史冻结结果；此次没有重新训练、评估、转换或校准 QAD。
3. 两条路线在神经元实现、状态/激活表示、蒸馏、码本和优化器细节上不同，因此是完整路线比较，不是训练域的单因素因果消融。
4. 历史 QAD 使用 feature distillation，当前 SQUAT 不使用教师。
5. SQUAT 保留 softmax、LayerNorm、ReLU、sigmoid 和插值等 FP32 连续模块，属于混合 SNN。
6. M4 使用 snnTorch 公开参考实现的固定阈值相对范围；文献正文还存在动态范围表述。
7. validation 同时用于模型选择和最终报告，没有独立 test split。
8. 本次执行的是用户确认的 100+40 预算，不是预估约 247.7 小时的 300+127 全预算。
9. 结果足以否定“当前 SQUAT 配方与 QAD 精度可比”，但不足以证明所有 SQUAT 实现或直接 SNN 量化方法都必然较差。

## 10. 建议

1. **当前继续采用 QAD 作为精度基准**，不使用本次 SQUAT 配方替代它。
2. 下一轮直接复用已保存的最佳 FP32 LIF-SNN checkpoint，优先做低成本、可定位的消融：
   - 只量化权重（W4，膜电位保持 FP32）；
   - 只量化膜电位/状态（M4，权重保持 FP32）；
   - W4M4 下逐层检查全零码层在量化前后的权重、scale 与梯度；
   - 针对输出 LIF 做读出校准或损失权重消融。
3. 每个消融重点监控 vehicle 脉冲计数、最大类并列率、逐层 firing rate、膜电位码分布和全零权重层数量。
4. **无需重训 QAD**；历史 QAD 继续作为冻结对照即可。
5. 若目标升级为论文级算法结论，再补充多 seed 和独立 test split；在此之前，先定位 W4 与 M4 哪一侧贡献了主要退化。

## 11. 结果文件

- 最终对比与完整审计：`squat_experiment_outputs/seed1234/evaluation/comparison.json`
- 本报告：`squat_experiment_outputs/seed1234/evaluation/report.md`
- FP32 训练曲线：`squat_experiment_outputs/seed1234/fp32_snn/metrics.csv`
- SQUAT 训练曲线：`squat_experiment_outputs/seed1234/w4m4s1_squat/metrics.csv`
- 运行完成状态：`squat_experiment_outputs/seed1234/tmux_pipeline_status.json`
- 实验清单：`squat_experiment_outputs/seed1234/manifest.json`
- 100+40 协议修订：`squat_experiment_outputs/seed1234/protocol_amendment_100_40.json`
