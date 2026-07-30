# SpikingLETNet_shallow_max：Windows 11 / Intel 14900K CPU 部署包

本目录用于在原生 Windows 11 上对固定输入 `[1,3,400,400]` 的 `SpikingLETNet_shallow_max` 单步 ANN/QIF 等价图进行 FP32 与 W8A8 CPU 推理、延迟统计和 CPU Package 能耗测试。

默认后端是 ONNX Runtime `CPUExecutionProvider`；`OpenVINOExecutionProvider` 是可选后端。两套后端使用不同优化器和内核，报告会分别统计。

## 包内模型

- `models/SpikingLETNet_shallow_max_fp32_portable.onnx`：经过形状推导、常量折叠和冗余图消除的可移植 FP32 ONNX。
- `models/SpikingLETNet_shallow_max_w8a8_qdq_portable.onnx`：S8S8、权重按通道、激活按张量的静态 QDQ 混合精度 ONNX。
- `models/*_openvino_portable.onnx`：与上述两图在 CPU EP 上逐元素等价，但将 OpenVINO 不支持的 `Col2Im` 改写为标准 `Range/ScatterND/Reshape` 子图。OpenVINO 模式必须使用这组文件，避免静默回退到 CPU EP。

W8A8 不表示每一个 ONNX 节点都使用 INT8。适合量化的 Conv/MatMul/Gemm 路径带 QDQ；Softmax、Resize、形状操作和不受支持的路径可保留 FP32。脚本会输出图覆盖率、provider 分配和 OpenVINO runtime precision，不能仅凭文件名宣称 INT8 内核已生效。

`models/local_optimized/` 初始为空。首次在 14900K 上运行时，CPU EP 使用默认 portable 图并在这里生成目标机器专用的 ORT 优化图；OpenVINO 使用 `*_openvino_portable.onnx` 并生成编译缓存。这些文件不能复制到其他 CPU 后直接用于正式比较。

## 快速开始

在原生 Windows 11 PowerShell 中解压本目录；路径尽量不含中文或过长字符。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1 -InstallPython
```

能耗提供两条互斥路径：HWiNFO 功率日志积分适用于保持 Secure Boot/HVCI 开启的机器；Intel PCM 路径继续保留。两者的测量方法和字段名称不同，不能混合。

### HWiNFO 路径（当前受限 Windows PC 推荐）

详见 [HWINFO_SETUP_CN.md](HWINFO_SETUP_CN.md)。先完成延迟测试：

```powershell
.\run_all.ps1 -SkipEnergy
```

在 HWiNFO Sensors-only 中以 1000 ms 周期开始 CSV 日志，然后依次执行：

```powershell
.\run_hwinfo_energy.ps1 -Mode Record -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
# Record 完成后停止 HWiNFO 日志
.\run_hwinfo_energy.ps1 -Mode Analyze -HwinfoLog "D:\logs\hwinfo_energy.csv" -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

### Intel PCM 路径

详见 [PCM_SETUP_CN.md](PCM_SETUP_CN.md)。驱动满足当前 Windows 签名策略时，在管理员 PowerShell 中运行：

```powershell
.\check_pcm.ps1 -PcmExe "C:\Tools\pcm\pcm.exe"
.\run_all.ps1 -PcmExe "C:\Tools\pcm\pcm.exe"
```

双后端测试在相应命令加入 `-IncludeOpenVino`。只测试延迟、跳过能耗：

```powershell
.\run_all.ps1 -SkipEnergy
```

## 正式测试口径

- 原生 Windows 11、Python 3.11 x64。
- batch=1，固定 400×400，100 张验证图像预加载到内存并循环使用。
- 主延迟仅覆盖 `session.run()`，不含 PNG 解码、缩放、归一化、argmax 和文件保存。
- 同时输出 20 次辅助端到端计时，覆盖 PNG 解码、RGB 转换、缩放、归一化和 `session.run()`；该结果不作为主性能口径。
- 先扫描 P 核/E 核、SMT 和线程数；同一后端内 FP32/W8A8 使用同一个主配置。
- 正式延迟测试预热 30 秒并至少执行 200 次，报告 median/P90/P95/P99。
- PCM 使用 60 秒计数器差值窗口；HWiNFO 使用连续 1000 ms 功率日志和服务端 epoch 时间标记，对 180 秒窗口做梯形积分。
- 每个模型至少 5 次能耗试验并交替执行，另有 3 次空闲基线；HWiNFO 还连续预采集每模型 3 个补测储备窗口。
- PCM 指标保持原字段；HWiNFO 指标必须保留 `HWiNFO`/`hwinfo_` 标签，并报告积分 Package J/image、平均 Package W 和扣除空闲功率后的 Dynamic J/image。
- CV 超过 5%、热余量不大于 2°C、温度达到 98°C、发生热降频或测试前后台 CPU 超过 10% 时标记不稳定/无效；HWiNFO 另要求完整率至少 99%、中位采样间隔 0.8–1.2 秒、最大缺口 2.5 秒、积分方法跨度不超过 1%。
- 脚本临时切换 Windows“高性能”电源计划，并在正常或异常退出时恢复原计划。
- 本任务不设 mIoU 或量化精度门槛；只检查可加载、固定形状、有限输出和量化/内核覆盖。

## 结果文件

- `results/model_audit.json`：ONNX、QDQ、initializer 和 CPU 冒烟审计。
- `results/local_model_preparation.json`：14900K 本机优化产物信息。
- `results/tuning_<provider>.json`：核心/线程扫描及共同配置选择。
- `results/latency_<provider>.json`：正式延迟统计。
- `results/provider_profile_<provider>.json`：ORT provider 节点分配。
- `results/openvino_runtime_precision.json`：OpenVINO CPU 实际执行精度与实现。
- `results/energy_<provider>.json`、`energy_trials_<provider>.csv`：PCM 能耗结果。
- `results/energy_hwinfo_<provider>.json`、`energy_trials_hwinfo_<provider>.csv`：HWiNFO 功率积分结果。
- `results/raw_pcm/`：原始 PCM 证据。
- `results/raw_hwinfo/`：原始 HWiNFO CSV 及其 SHA-256。
- `results/result_summary.md`：最终中文摘要。

## 数据与追溯

- `data/calibration/`：跨 106 个训练场景确定性分层抽样的 300 张图像，用于复现静态量化。
- `data/validation/`：跨 35 个验证场景抽样的 100 对图像/标签；正式性能测试只读取图像。
- `provenance/`：原始 FP32 checkpoint、YAML 和导出/量化脚本。Windows 推理不需要安装 PyTorch。
- `provenance/server_validation/`：打包服务器上的模型、provider 与整数内核非性能验证证据；不能当作 14900K 性能结果。
- `checksums.sha256`：传输后可用 `Get-FileHash` 或其他 SHA-256 工具核验不可变文件。

## 注意

CPU Package 能耗包含测量窗口内整颗 CPU package 上的其他活动，不等于墙上插座的整机能耗。关闭浏览器、同步软件和游戏启动器；PCM 路径关闭其他 PMU 监控工具，HWiNFO 路径只保留用于采集的单个 HWiNFO Sensors 实例。保持散热稳定后再运行。不要把 CPU EP 的 FP32 结果与 OpenVINO EP 的 W8A8 结果拼成一组对比。
