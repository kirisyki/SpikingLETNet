# Codex 本地测试执行说明

本文件是交给本地 Codex 的执行契约。目标是在原生 Windows 11、Intel Core i9-14900K 上完成 `SpikingLETNet_shallow_max` 的 FP32 与 W8A8 ONNX CPU 推理、性能测试，并使用已明确标注的 PCM 或 HWiNFO 后端完成 CPU Package 能耗测试。

## 1. 不可变测试口径

- 正式测试必须在原生 Windows 11 中执行；WSL 只能用于辅助检查，不得生成正式能耗结论。
- Python 固定为 3.11 x64，依赖版本以 `requirements.txt` 为准。
- 默认后端为 ONNX Runtime `CPUExecutionProvider`；`OpenVINOExecutionProvider` 仅作为可选的独立后端。
- 只能在同一 provider 内比较 FP32 与 W8A8，不得把不同 provider 的结果组合成一组结论。
- 固定 batch=1、输入 `image: float32[1,3,400,400]`、输出 `logits: float32[1,6,400,400]`。
- 主延迟边界仅为预加载输入后的 `session.run()`；PNG 解码和预处理只作为辅助端到端指标。
- 本任务不做 mIoU、精度或量化误差验收。数值比较只能用于确认输出形状正确且全部有限。
- 能耗允许两种互斥口径：Intel PCM/RAPL Package 计数器差值，或 HWiNFO `CPU Package Power [W]` 时间积分；不得混称、拼接或改名。
- 不要修改或覆盖 `models/*_portable.onnx`、`provenance/model_best.pth`、测试数据和 `checksums.sha256`。
- `models/local_optimized/` 和 `results/` 是目标机生成目录，可以重新生成。

## 2. 模型选择

CPU EP 必须使用：

- `models/SpikingLETNet_shallow_max_fp32_portable.onnx`
- `models/SpikingLETNet_shallow_max_w8a8_qdq_portable.onnx`

OpenVINO EP 必须使用：

- `models/SpikingLETNet_shallow_max_fp32_openvino_portable.onnx`
- `models/SpikingLETNet_shallow_max_w8a8_qdq_openvino_portable.onnx`

OpenVINO 兼容图将原图中的 `Col2Im` 改写为标准 ONNX 子图。不要让 OpenVINO 直接加载 CPU EP 的 portable 图，否则可能静默回退到 CPU EP。

## 3. 执行前检查

Codex 应先读取：

- `README_CN.md`
- `PCM_SETUP_CN.md`
- `HWINFO_SETUP_CN.md`
- `artifact_manifest.json`
- `config/benchmark.json`

确认当前目录是解压后的 `windows_cpu_deployment` 根目录。建议路径较短且不含中文，例如 `D:\LETNetBench\windows_cpu_deployment`。

检查系统状态：

1. 确认操作系统为原生 Windows 11，CPU 为 Intel Core i9-14900K。
2. 确认电脑接通电源且散热稳定。
3. 关闭浏览器、同步软件、游戏启动器和杀毒扫描；PCM 路径关闭其他 PMU 监控，HWiNFO 路径只保留一个用于日志采集的 HWiNFO Sensors 实例。
4. 不得要求用户关闭 Secure Boot、内存完整性或驱动签名验证。
5. PCM 必须来自 Intel 官方仓库；HWiNFO 必须来自官方渠道并通过 Authenticode 签名检查。

## 4. 安装环境

在普通 PowerShell 中执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1 -InstallPython
```

如果已经有 Python 3.11 x64，可执行：

```powershell
.\setup_windows.ps1
```

若安装失败，Codex 应记录完整命令、退出码和错误信息，修复环境问题后重新运行；不得通过放宽依赖版本或改写模型规避失败。

## 5. 先执行无能耗推理验证

在普通 PowerShell 中运行默认 CPU EP：

```powershell
.\run_all.ps1 -SkipEnergy
```

这一步必须完成模型审计、本机图优化、核心/线程扫描、正式延迟测试和 provider profile。检查：

- `results/model_audit.json` 中 `passed` 为 `true`。
- FP32/W8A8 输出固定为 `[1,6,400,400]` 且全部有限。
- W8A8 图存在 INT8 initializer 和 QDQ 节点。
- `results/provider_profile_cpu.json` 中 W8A8 存在 `observed_quantized_operator_events`，且数值大于 0。
- `results/latency_cpu.json` 至少包含 200 次正式测量，并报告 median/P90/P95/P99。
- FP32/W8A8 使用同一个 `selected_config`。

如需同时验证 OpenVINO，但暂不测能耗：

```powershell
.\run_all.ps1 -SkipEnergy -IncludeOpenVino
```

额外检查：

- `results/provider_profile_openvino.json` 中 FP32/W8A8 都有 `OpenVINOExecutionProvider` 事件。
- 不得出现 `CPUExecutionProvider` 节点事件；出现即视为静默回退并停止引用该后端结果。
- `results/openvino_runtime_precision.json` 中 W8A8 的 `integer_runtime_nodes` 和 `integer_implementations` 均大于 0。
 
## 6. 能耗后端决策

先读取 `ENERGY_TEST_BLOCKERS_CN.md`。不得为了 PCM 关闭 Secure Boot、内存完整性/HVCI、驱动签名验证或启用测试签名模式。

- 只有官方 PCM 驱动能在当前安全策略下正常加载且 `check_pcm.ps1` 通过时，才可使用 PCM 路径。
- 当前机器若仍存在该文档记录的 PCM 证书/权限阻塞，采用 `HWINFO_SETUP_CN.md` 的 HWiNFO 正式路径。
- HWiNFO 结果必须命名为 HWiNFO CPU Package 功率积分能耗；它不是 PCM/RAPL 计数器差值。
- 两种后端只能择一生成本轮正式结论。保留完整原始证据，不得用估算值填补失败结果。

## 7. HWiNFO 正式 CPU EP 测试（当前机器推荐）

延迟和调优已经完成时，确认 `results/tuning_cpu.json` 存在，不要重跑。先验证官方 HWiNFO 签名：

```powershell
.\check_hwinfo.ps1 -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

由用户启动 HWiNFO Sensors-only，设置 1000 ms 轮询并开始连续 CSV 日志。日志开始至少 10 秒后运行：

```powershell
.\run_hwinfo_energy.ps1 -Mode Record -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

Record 结束后停止 HWiNFO 日志，确认 CSV 不再增长，然后分析：

```powershell
.\run_hwinfo_energy.ps1 -Mode Analyze -HwinfoLog "D:\logs\hwinfo_energy.csv" -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

正式协议：3 个 180 秒空闲窗；每模型 5 个 180 秒基础窗并交替；每模型另连续预采集 3 个补测储备窗。分析只按采集顺序使用补测，禁止事后挑选。样本完整率至少 99%，中位间隔 0.8–1.2 秒，最大缺口 2.5 秒，三种积分相对跨度不超过 1%；后台 CPU 不超过 10%，热余量大于 2°C，最高温度低于 98°C且不得热降频；每图能耗和均值延迟 CV 均不超过 5%。

## 8. 可选 PCM 正式路径

仅在驱动满足当前 Windows 签名策略时，在管理员 PowerShell 中执行：

```powershell
.\check_pcm.ps1 -PcmExe "C:\Tools\pcm\pcm.exe"
.\run_all.ps1 -PcmExe "C:\Tools\pcm\pcm.exe"
```

PCM 继续采用 3 次空闲基线、每模型至少 5 次交替的 60 秒窗口以及最多 3 次补测。PCM 专用字段和 `raw_pcm` 只用于 PCM，不得写入 HWiNFO 结果。

## 9. 可选 OpenVINO 正式测试

只有用户要求双后端时才加入 `-IncludeOpenVino`。HWiNFO 的 Record 与 Analyze 两阶段必须同时加入该参数；PCM 在 `run_all.ps1` 中加入。CPU EP 和 OpenVINO EP 分别汇总。OpenVINO 结果只有在 provider profile 无 CPU EP 回退且 runtime precision 观察到整数内核时才可引用。

## 10. 结果检查与汇报

至少检查并保留：

- `results/system_info.json`
- `results/model_audit.json`
- `results/local_model_preparation.json`
- `results/tuning_cpu.json`
- `results/latency_cpu.json`
- `results/provider_profile_cpu.json`
- PCM 路径：`results/energy_cpu.json`、`energy_trials_cpu.csv`、`raw_pcm/cpu/`
- HWiNFO 路径：`results/energy_hwinfo_cpu.json`、`energy_trials_hwinfo_cpu.csv`、`raw_hwinfo/cpu/`、`hwinfo_trial_markers_cpu.json`
- `results/result_summary.md`

如果测试 OpenVINO，还要保留对应 provider 的能耗 JSON/CSV、原始采集日志 和 `openvino_runtime_precision.json`。

Codex 的最终汇报必须包含：

1. Windows 版本、CPU、Python、ORT、OpenVINO 和所选能耗采集工具版本。
2. 每个 provider 选中的 CPU affinity profile、逻辑 CPU ID 和线程数。
3. FP32/W8A8 的 median、P90、P95、P99、吞吐量和辅助端到端中位延迟。
4. FP32/W8A8 的 Package J/image、Dynamic J/image、平均 Package W、有效试验数和 CV。
5. 同一 provider 内 W8A8 相对 FP32 的延迟加速比和能耗变化。
6. QDQ/量化算子、provider 接管和 OpenVINO 整数运行时检查结果。
7. 所有无效试验、补测、热余量或后台负载问题。
8. 明确声明本任务没有精度验收，且 CPU Package 能耗不等于整机墙上功耗。

禁止只摘取最快一次或最低能耗一次作为结论。优先引用 `result_summary.md`，但必须结合 JSON、CSV 和对应 raw_pcm 或 raw_hwinfo 文件核对。

## 11. 推荐给本地 Codex 的指令

用户可在解压目录中直接对 Codex 说：

> 请完整读取 instruction.md、README_CN.md、HWINFO_SETUP_CN.md 和 PCM_SETUP_CN.md，先检查当前 Windows 安全状态与能耗后端，再严格按 instruction.md 执行 CPU EP 正式测试；保持 Secure Boot 和内存完整性开启；若 PCM 仍被证书/权限阻塞，使用 HWiNFO 正式路径完成能耗测试。不要修改 portable 模型和测试口径。完成后核对 JSON、CSV 与对应原始采集日志，生成中文结果摘要。若我要求双后端，再启用 OpenVINO，并单独报告。

