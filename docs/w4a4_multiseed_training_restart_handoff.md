# W4A4 三随机种子训练：执行与恢复记录

## 当前状态

- 记录日期：2026-08-20
- 项目目录：`/root/autodl-tmp/LETNet`
- 状态：**正式 multi-seed训练进行中**
- 正式训练进度：**0/8 个新增运行**
- 当前执行：QAD seed2345与seed3456分别在两张GPU上并行训练；完成后各自串行执行
  STE、LSQ、EWGS和全量验证。
- 正式 tmux 会话：`w4a4_seed2345_gpu0`、`w4a4_seed3456_gpu1`。
- 所有已生成文档和 smoke 日志位于 `/root/autodl-tmp/LETNet` 数据盘目录，服务器重启后
  应从该目录继续。

本文是当前唯一有效的 W4A4 multi-seed训练与重启方案；此前的 multi-seed草案已删除，
不得再引用。

---

## 1. 已冻结的训练矩阵

随机种子统一预注册为 `1234`、`2345`、`3456`。

| 方法 | Epochs/seed | seed 1234 | 重启后待补 |
|---|---:|---|---|
| QAD | 127 | 复用历史结果 | seed 2345、3456 |
| STE-QAT | 16 | 复用现有结果 | seed 2345、3456 |
| LSQ-QAT | 16 | 复用现有结果 | seed 2345、3456 |
| EWGS-QAT | 16 | 复用现有结果 | seed 2345、3456 |

- 待新增正式训练：8 个进程。
- 待新增训练预算：`2 × 127 + 6 × 16 = 350 epoch-equivalents`。
- QAD-127 与三个16-epoch基线属于不等预算路线对比，论文不得表述为同预算算法消融。
- 不得根据中途结果替换 seed、单独延长 epoch 或针对某个 seed 调参。

---

## 2. 固定训练协议

| 项目 | 固定值 |
|---|---|
| 模型 | `SpikingLETNet_shallow_max` |
| 数据集 | UDD，6 类 |
| 输入 | `400 × 400` |
| 训练 batch size | 64 |
| 验证 batch size | 20 |
| 时间步 | `T_time=1` |
| 量化 | W4A4、signed、per-tensor、71 层、无首尾层豁免 |
| AMP | 关闭 |
| 优化器 | Adam，基础学习率0.001，weight decay `1e-4` |
| LR schedule | polynomial，horizon 150 epoch，exponent 0.9 |
| checkpoint选择 | QAD：127轮内最佳；其余方法：16轮内最佳 |
| 主指标 | 全量 UDD validation 六类 mIoU |
| 运行方式 | 每个 smoke 和正式训练均在独立 tmux 会话中运行 |

固定 FP32 起点：

`checkpoint/udd/SpikingLETNet_shallow_maxbs64gpu1_trainval20260611-003821/model_best.pth`

SHA-256：

`2729d27dad82b9ec5124cc6474bcaad5bb3813de30b1875cd39765be1f46c3fb`

固定配置：

`Network/configs/SpikingLETNet_shallow/1.3M.yaml`

SHA-256：

`ab1008283b7f2f539d17654ae30861c5a794145ac3aaa1d4b5f02fc1589becb8`

---

## 3. 换卡前显存测试结果

测试设备：NVIDIA GeForce RTX 5090，`32,607 MiB` 总显存。测试使用 QAD、W4A4、
`400×400`、`T=1`、AMP关闭，并完成或尝试一个训练 batch和一个验证 batch。

| 测试 | 结果 | nvidia-smi采样峰值 |
|---|---|---:|
| batch 64，默认分配器 | OOM；学生前向阶段还需申请2.44 GiB | 29,894 MiB |
| batch 64，expandable segments | OOM；接近满卡时还需申请158 MiB | 31,956 MiB |
| batch 32，expandable segments | 成功完成前向、反向、优化器更新和验证 | 25,450 MiB |

结论：当前32GB卡不能执行计划中的 QAD batch 64。batch 32虽然可运行，但会改变训练协议，
不能未经修订直接用于正式三-seed对比。

smoke 产物：

- `quantization_multiseed_results/udd/w4a4_v1/smoke/qad_memory_rtx5090_seed1234_v1/`
- `quantization_multiseed_results/udd/w4a4_v1/smoke/qad_memory_rtx5090_seed1234_expandable_v1/`
- `quantization_multiseed_results/udd/w4a4_v1/smoke/qad_memory_rtx5090_seed1234_bs32_v1/`

换卡后设备为两张 NVIDIA RTX PRO 6000 Blackwell Server Edition，每张`97,887 MiB`。
QAD batch64在GPU0完成完整教师/学生前向、KD、反向、optimizer step和验证，峰值
`57,321 MiB`（约58.6%），退出码0。产物位于：

- `quantization_multiseed_results/udd/w4a4_v1/smoke/qad_memory_pro6000_gpu0_bs64_v1/`

---

## 4. 当前双卡执行配置

1. GPU0只运行seed2345队列，GPU1只运行seed3456队列；两个进程不共享单卡。
2. 每个进程仍是单卡batch64，没有使用DDP、DataParallel、梯度累积或AMP。
3. 两张卡分别设置`CUDA_VISIBLE_DEVICES=0/1`和`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
4. QAD运行时单卡实测约54–57 GiB，低于97,887 MiB容量。
5. 队列日志：
   - `quantization_multiseed_results/udd/w4a4_v1/run_logs/seed2345/`
   - `quantization_multiseed_results/udd/w4a4_v1/run_logs/seed3456/`
6. 正式checkpoint：
   - `quantization_multiseed_checkpoint/udd/w4a4_v1/seed2345/`
   - `quantization_multiseed_checkpoint/udd/w4a4_v1/seed3456/`

---

## 5. 重启后的恢复检查清单

### 5.1 只读环境检查

在 `/root/autodl-tmp/LETNet` 中执行：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
git status --short
```

检查要求：

- PyTorch 能识别全部新 GPU；
- QAD 指定的单卡没有其他进程占用；
- 项目目录、主方案和三组 smoke 日志仍存在；
- FP32 checkpoint/config hash 与本文一致。

### 5.2 先重做 QAD batch-64显存 smoke

正式训练前必须在新卡上重复：

- QAD；
- batch size 64；
- 输入 `400×400`；
- `T=1`；
- W4A4；
- AMP关闭；
- 一个完整训练 batch，必须覆盖教师前向、学生前向、KD loss、反向和 optimizer step；
- 一个验证 batch；
- 约0.2秒间隔记录 `nvidia-smi` 显存。

测试必须放在新的 tmux 会话和新的 smoke 目录中，不得覆盖本次三组日志。通过条件：

1. 退出码为0；
2. 完整前向、反向、optimizer step 和验证全部完成；
3. 无 OOM/NaN/Inf；
4. 峰值显存建议不高于单卡容量的85%，为 DataLoader波动、allocator和长期训练保留余量。

### 5.3 代码实施状态

多 seed训练、评估和聚合入口已经创建并通过26项相关测试：

- `Network/train_qad_multiseed.py`
- `Network/train_quantization_baseline_multiseed.py`
- `Network/evaluate_quantization_multiseed.py`
- `Network/aggregate_quantization_multiseed.py`
- `tests/test_multiseed_training_entrypoints.py`

历史 QAD入口保持只读；正式新 seed 只使用`Network/train_qad_multiseed.py`。

---

## 6. 当前正式队列与恢复顺序

1. 两个tmux队列均先执行QAD 127 epochs；每轮原子保存`checkpoint_last.pth`和最佳checkpoint。
2. QAD完成后，同一seed队列依次执行STE、LSQ、EWGS各16 epochs。
3. 四种方法完成后，队列执行该seed的完整424-batch validation。
4. 两个seed均验证完成后，运行`Network/aggregate_quantization_multiseed.py`生成最终
   Markdown/CSV/JSON。
5. 若SSH断开，tmux继续运行；重新连接后用`tmux attach -t w4a4_seed2345_gpu0`或
   `tmux attach -t w4a4_seed3456_gpu1`查看。
6. 若服务器重启导致QAD中断，使用对应输出目录的`checkpoint_last.pth`和同一seed恢复；
   不得换seed或更改协议。

当前有正式训练进程运行，**不要重启或释放服务器**。

