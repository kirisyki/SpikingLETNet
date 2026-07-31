# W4A4 量化训练对比执行修订：缩短正式训练预算

## 决策

在 STE-QAT 完成 16 个完整 epoch 后，用户要求减少训练 epoch 以降低总训练时间。
三种对比方法的正式训练预算因此统一从 127 epoch 调整为 16 epoch。

不变项：

- 模型：`SpikingLETNet_shallow_max`
- 量化精度：W4A4、per-tensor、全部 71 个目标层、无首尾层豁免
- 数据、增强、batch size、seed=1234、优化器及其算法特有参数
- 学习率调度 horizon：150 epoch
- 任务损失及统一验证口径
- EWGS Hessian：每次更新 10 个 batch、每 batch 最多 50 次 Hutchinson 迭代

## 执行处理

- STE-QAT 第 16 epoch 完成验证和 checkpoint 保存后停止。
- 第 17 epoch 的未完成更新不保存，并从第 16 epoch checkpoint 作为最终运行状态。
- LSQ-QAT 与 EWGS-QAT 均执行 16 个完整 epoch。
- 旧的 127-epoch 协议 checkpoint 只允许修改协议与 manifest 元数据；
  模型、优化器和 RNG 状态不变。

## 时间影响

EWGS 的 1 batch × 1 Hutchinson 迭代实测耗时为约 23.06 秒。按 16 epoch
中 epoch 1–15 的 15 次 Hessian 更新外推，正式 Hessian 部分约需：

`23.06 × 15 × 10 × 50 / 3600 ≈ 48.0 GPU 小时`

## 解释限制

现有 QAD 模型训练了 127 epoch，而新的对比基线只训练 16 epoch。因此最终结果可以描述
“既有 QAD 相对于缩短预算基线的表现”，但不能把所有差异严格归因于 QAD 算法本身。
若需要无训练预算混杂的论文级结论，仍应恢复三种基线的 127-epoch 正式实验。
