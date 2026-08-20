# Integer-LIF + QAD 与直接 SNN 量化训练效率对比

## 结论

在 `NVIDIA GeForce RTX 5090`、400×400 输入、AMP 关闭的训练内核 benchmark 中，同 physical/effective batch=4 时，Integer-LIF + QAD 的峰值 allocated memory 为 **3.047 GiB**，Direct-LIF + QAT+SQUAT（T=8、full BPTT）为 **14.077 GiB**：QAD 降低 **78.35%**，吞吐提高 **4.545×**。

effective batch=64 时，QAD 采用 physical batch=32、SQUAT 采用 physical batch=4，吞吐分别为 **108.54** 与 **10.75 images/s**（**10.095×**）；这组峰值显存因 physical batch 不同而不作倍率比较。

## 主结果

| 口径 | 路线 | physical/effective B | T | Peak allocated GiB | Median s/update | images/s |
|---|---|---:|---:|---:|---:|---:|
| 同 physical batch | Integer-LIF + QAD | 4/4 | 1 | 3.047 | 0.081555 | 49.047 |
| 同 physical batch | Direct-LIF + QAT+SQUAT | 4/4 | 8 | 14.077 | 0.370642 | 10.792 |
| 同 effective batch | Integer-LIF + QAD | 32/64 | 1 | 24.006* | 0.589642 | 108.540 |
| 同 effective batch | Direct-LIF + QAT+SQUAT | 4/64 | 8 | 14.082* | 5.952199 | 10.752 |

*physical batch 不同，不用于显存倍率结论。

## 机制证据

在同一 SQUAT 路线内，T=8 相比 T=1 的峰值显存为 **5.853×**，更新时间为 **4.522×**，直接支持时间展开和 full BPTT 是额外开销的重要来源。完整 QAD 相对无教师/KD 的 QIF-STE 多用 **0.578 GiB**，pooled median 更新时间比为 **1.001×**。

## 精度上下文

冻结 seed-1234 记录：QAD mIoU **0.619547**，SQUAT mIoU **0.315514**，差 **+0.304033**。这是不同训练预算与完整路线设置下的支持性证据，不能视为受控因果比较。

## 论文推荐表述

> On a single RTX 5090 with identical physical and effective batch sizes of 4, Integer-LIF + QAD reduced peak PyTorch-allocated training memory from 14.08 to 3.05 GiB (78.4%) and improved training throughput by 4.54× over the T=8 Direct-LIF QAT+SQUAT baseline with full BPTT. At an effective batch size of 64, QAD supported an 8× larger physical batch and achieved 10.09× higher throughput.

## 方法与边界

- 主比较：5 个独立进程轮次，每轮 10 warmups + 20 measured updates；轮次中位数 block bootstrap 95% CI。
- 时间步消融：3 轮，每轮 5 warmups + 10 measurements。
- 固定 64 个真实 UDD 样本反复测量；数据加载、验证与 checkpoint I/O 排除，epoch 时间仅为投影。
- 主比较改变了神经元、T、BPTT、量化对象和蒸馏，只能解释完整路线差异。
- `torch.cuda.max_memory_allocated` 不含 CUDA context 与非 PyTorch 分配；fake-quant 训练速度不代表低比特硬件推理速度或能耗。
