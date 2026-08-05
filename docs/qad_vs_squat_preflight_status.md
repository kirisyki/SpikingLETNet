# QAD 与直接 QAT+SQUAT 实验执行状态：preflight 阻断

> 日期：2026-08-02  
> 状态：正式训练未启动；按已批准的 48 小时规则停止

## 结论

不可报告的 GPU smoke/preflight 已通过正确性、显存和计时检查，但测得最低预注册
预算仍明显超过 48 小时，因此没有启动 FP32 LIF-SNN 或 QAT+SQUAT 正式训练。

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition（97,887 MiB）。
- 物理 batch 64、32：OOM。
- 可行物理 batch：16；梯度累积 4 次；有效 batch 保持 64。
- FP32 LIF-SNN：每个完整 epoch 预计 2,030.756 秒。
- W4M4S1 QAT+SQUAT：每个完整 epoch 预计 2,224.422 秒。
- 完整 300+127 epoch：预计 247.702 小时。
- 原方案最低 100+40 epoch：预计 81.126 小时。

已批准方案规定：若最低 `100+40` 仍预计达到或超过 48 小时，应停止并报告，不能
擅自启动更短的正式 run。该条件已经触发。

## 测速口径

两阶段都使用真实 `T=8`、400×400 输入、完整 BPTT、AMP 关闭。每阶段排除一个
warm-up 有效 batch 后测量两个有效 batch，并排除两个 validation warm-up batch 后
测量五个 validation batch。估算把 25,700 个训练样本的 402 次 optimizer update
以及全部 8,478 个验证样本的 424 个 validation batch 计入每个 epoch。

| 阶段 | train/effective-batch 中位数 | val/batch 中位数 | train/epoch | val/epoch | 总计/epoch |
|---|---:|---:|---:|---:|---:|
| FP32 LIF-SNN | 4.6473 s | 0.3834 s | 1,868.208 s | 162.548 s | 2,030.756 s |
| W4M4S1 SQUAT | 5.0283 s | 0.4789 s | 2,021.368 s | 203.054 s | 2,224.422 s |

完整机器可读结果在
`squat_experiment_outputs/seed1234/preflight.json`；环境、源文件哈希、冻结 QAD
来源和工作树状态在同目录的 `manifest.json`。

## 已完成的隔离与测试

- 独立实现位于 `Network/squat_comparison/`，没有导入现有 QAD/STE/LSQ/EWGS/
  ternary 量化器。
- 17 个新测试全部通过：M4 参考语义、LIF 时序、W4、71 层覆盖、模型复制隔离、
  checkpoint/RNG、指标手算、旧目录写保护和无 QAD compute node。
- 历史 QAD 没有被训练、评估、校准或转换；流水线只读取冻结 JSON 指标。
- 当前正式输出目录只有 `manifest.json` 与 `preflight.json`，没有训练 checkpoint。
- preflight 完成后 GPU 占用已释放为 0 MiB。

全仓无选择的 `pytest -q` 会在收集既有
`FPGA_test_package/Network/tools/flops_counter/ENet_Flops_test.py` 时因其自身无法导入
`model` 而失败；这不是新实现回归。新实验的定向测试为 17/17 通过。

## 如需继续所需的协议修订

按原比例公式但取消 `100+40` 最低门槛时，预算为 `53+22` epoch，实测外推约
43.491 小时。这是最贴近已批准 44 小时目标的备选，但会显著缩短两阶段训练，必须
先书面修订实验协议，最终结果也必须明确标为缩短预算结果。

