# Integer-LIF + QAD 与直接 QAT+SQUAT 训练效率对比

> 此文件由 benchmark 自动生成；最终论文表述需经过统计与方法学验证。

- GPU：`NVIDIA GeForce RTX 5090`
- 输入：UDD 真实样本，400×400，AMP 关闭
- 计时：CUDA 同步 wall clock；不含数据读取、验证和 checkpoint I/O

## 汇总

| 模式 | 方法 | physical/effective batch | T | 峰值显存 GiB | update 中位时间 s | 吞吐 img/s |
|---|---|---:|---:|---:|---:|---:|
| capacity | qad | 32/64 | 1 | 24.004 | 0.580272 | 110.293 |
| capacity | squat | 4/64 | 8 | 14.082 | 5.821449 | 10.994 |
| matched | qad | 4/4 | 1 | 3.047 | 0.085424 | 46.825 |
| matched | squat | 4/4 | 8 | 14.077 | 0.360608 | 11.092 |
| qad_ablation | qif_ste | 4/4 | 1 | 2.469 | 0.064409 | 62.163 |
| timestep_ablation | squat | 4/4 | 1 | 2.405 | 0.065267 | 61.352 |
| timestep_ablation | squat | 4/4 | 2 | 4.189 | 0.093753 | 42.684 |
| timestep_ablation | squat | 4/4 | 4 | 7.735 | 0.171037 | 23.389 |
| timestep_ablation | squat | 4/4 | 8 | 14.077 | 0.363779 | 10.996 |

## 主比较倍率

### capacity

- QAD 峰值显存降低：`-70.46%`
- SQUAT/QAD 显存倍率：`0.587×`
- QAD 吞吐加速：`10.032×`
- QAD update 时间降低：`90.03%`

### matched

- QAD 峰值显存降低：`78.35%`
- SQUAT/QAD 显存倍率：`4.620×`
- QAD 吞吐加速：`4.221×`
- QAD update 时间降低：`76.31%`

## 结论边界

主比较同时改变 QIF/LIF、T=1/T=8、BPTT、状态/激活量化与教师蒸馏，
因此只能解释为完整训练路线的效率差异，不能把全部差异因果归因于 QAD。
PyTorch fake quantization 的训练耗时也不代表低比特硬件推理速度或能耗。
