# HWiNFO（Windows 11）CPU Package 功率积分方案

本方案用于 Intel PCM 驱动无法在保持 Secure Boot 与内存完整性（HVCI）开启的条件下加载时。它保留原生 Windows 11、相同 ONNX Runtime 会话、相同 CPU affinity/线程配置、batch=1、FP32/W8A8 交替和重复试验协议，但将采集后端改为 HWiNFO 的 `CPU Package Power [W]` 连续日志。

该结果必须命名为 **HWiNFO CPU Package 功率积分能耗**。它不是 Intel PCM/RAPL 能量计数器差值，也不是整机墙上插座能耗，不能与 PCM 数值混合。

## 1. 安全边界

- 不关闭 Secure Boot。
- 不关闭 Windows“内存完整性”或 HVCI。
- 不启用测试签名模式，不加载来源不明的 MSR/WinRing0 驱动。
- HWiNFO 必须从官方渠道下载，并由 `check_hwinfo.ps1` 验证 Authenticode 签名。
- HWiNFO 由用户在桌面会话中手动启动；若官方 HWiNFO 自身请求 UAC，可由用户单独确认，但 Codex、PowerShell 推理脚本和分析器保持非管理员运行。
- 正式采集在原生 Windows 11 中完成，WSL 不参与计时或能耗采集。

## 2. HWiNFO 传感器日志设置

1. 启动 `HWiNFO64.exe`，选择 **Sensors-only**，界面/传感器名称切换为 English。
2. 将全局传感器轮询周期设为 **1000 ms**。
3. 在 Sensor Settings 的 Custom 页使用 **Logged** 只记录本实验所需项目，减少日志和轮询扰动。
4. 确认日志同时含以下列：
   - `Date`
   - `Time`
   - `CPU Package Power [W]`
   - `Distance to TjMAX` 热余量
   - `CPU Package [°C]` 或 `Core Max [°C]`
   - `Thermal Throttling`
5. 开始 CSV 日志后，先等待至少 10 秒，再运行 Record。
6. 整个 Record 阶段只保留一个连续日志，结束后再停止日志。
7. 停止日志并确认文件大小不再变化后，才能运行 Analyze。

如果日志里有多个同名 Package Power 列，Analyze 会拒绝猜测。请从 CSV 表头复制完整列名，并通过 `-PowerColumn "完整列名"` 指定。

## 3. 首次环境和签名检查

在部署目录的普通 PowerShell 中执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
.\check_hwinfo.ps1 -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

脚本只检查文件签名、版本和日志结构，不修改安全设置。

正式实验前先录制至少 2 分钟试验日志，停止日志后执行：

```powershell
.\check_hwinfo.ps1 -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe" -HwinfoLog "D:\logs\pilot.csv"
```

确认输出的实际中位采样间隔在 0.8–1.2 秒。Windows 不是实时系统，HWiNFO 实际间隔会包含传感器轮询和写盘开销；若超出范围，应禁用不需要且耗时的传感器或微调轮询周期后重新做 pilot，不能放宽正式分析阈值。

## 4. 已经完成延迟测试时的正式流程

当前 PC 已有 `results\tuning_cpu.json` 和延迟结果时，不要重跑调优。按以下顺序执行。

先在 HWiNFO 中开始连续日志，然后：

```powershell
.\run_hwinfo_energy.ps1 `
  -Mode Record `
  -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

Record 默认采集：

- 3 个 180 秒空闲窗口；
- FP32/W8A8 各 5 个 180 秒基础窗口，按轮次交替顺序；
- FP32/W8A8 各 3 个 180 秒预采集补测窗口。

全部窗口约需 57 分钟/后端。补测窗口预先连续采集，Analyze 只按采集顺序使用所需窗口，不能事后挑选最低能耗样本。

Record 完成后停止 HWiNFO 日志，再执行：

```powershell
.\run_hwinfo_energy.ps1 `
  -Mode Analyze `
  -HwinfoLog "D:\LETNetBench\logs\hwinfo_energy.csv" `
  -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

若 Windows 日期为日/月/年且日志日期存在歧义：

```powershell
.\run_hwinfo_energy.ps1 -Mode Analyze -HwinfoLog "D:\logs\hwinfo.csv" -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe" -DateOrder dmy
```

## 5. 从头完成 CPU EP 测试

推荐先完成无能耗阶段：

```powershell
.\run_all.ps1 -SkipEnergy
```

然后按第 4 节单独执行 HWiNFO Record/Analyze。也可以在 HWiNFO 已经开始日志后一次运行：

```powershell
.\run_all.ps1 -EnergyBackend Hwinfo -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe"
```

该命令结束时只完成时间窗记录；必须停止 HWiNFO 日志并运行 Analyze 才会生成能耗结论。

## 6. 可选 OpenVINO

只有用户要求双后端时才加入 `-IncludeOpenVino`，Record 和 Analyze 两个阶段必须保持一致：

```powershell
.\run_hwinfo_energy.ps1 -Mode Record -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe" -IncludeOpenVino
.\run_hwinfo_energy.ps1 -Mode Analyze -HwinfoLog "D:\logs\hwinfo_energy.csv" -HwinfoExe "C:\Tools\HWiNFO64\HWiNFO64.exe" -IncludeOpenVino
```

CPU EP 与 OpenVINO EP 单独汇总，不得跨后端组合 FP32/W8A8 结果。

## 7. 自动质量门槛

每个窗口必须同时满足：

- 样本完整率至少 99%；
- 中位采样间隔在 0.8–1.2 秒；
- 最大采样缺口不超过 2.5 秒；
- 左矩形、右矩形、梯形三种积分的相对跨度不超过 1%；
- 测试前后台 CPU 占用不超过 10%；
- 热余量大于 2°C、最高温度低于 98°C、未观察到 Thermal Throttling；
- 每个模型至少 5 个有效窗口；
- 每图能耗 CV 与推理均值延迟 CV 均不超过 5%。

Analyze 会拒绝没有热遥测、缺少 3 个有效空闲基线或未被日志前后样本完整包围的窗口。仍不稳定时必须报告 `complete_unstable`，不得选择性删除结果。

## 8. 计算和输出

每个推理窗口使用服务端记录的 epoch 起止时间，在 HWiNFO 相邻样本间线性插值边界，并按梯形法积分：

```text
HWiNFO total CPU package energy (J) = integral(CPU Package Power, time)
HWiNFO CPU package J/image = total energy / completed inference count
HWiNFO dynamic CPU package J/image =
    (total energy - idle baseline power * window duration)
    / completed inference count
```

主要文件：

- `results\hwinfo_trial_markers_<provider>.json`
- `results\energy_hwinfo_<provider>.json`
- `results\energy_trials_hwinfo_<provider>.csv`
- `results\raw_hwinfo\<provider>\`
- `results\energy_status_hwinfo_<provider>.json`
- `results\result_summary.md`

原始 HWiNFO CSV 会复制到 `raw_hwinfo` 并记录 SHA-256。最终报告中的字段保留 `hwinfo_` 前缀，避免被误认为 PCM/RAPL。

## 9. 常见失败

- `window_not_bracketed`：日志开始过晚、停止过早，或 Analyze 不在采集 PC 的相同时区运行。
- `median_sampling_interval` / `maximum_sampling_gap`：轮询周期不是 1000 ms，或日志丢样。
- `thermal_headroom_telemetry_missing`、`temperature_telemetry_missing` 或 `thermal_throttling_telemetry_missing`：日志未选择温度、热余量或降频列。
- `background_cpu`：关闭后台任务后整轮重测。
- `complete_unstable`：检查散热、电源计划和后台负载；不得删除不利样本后重算。
- 找不到 Package Power：确认列名含 `CPU Package Power` 和 `[W]`，必要时使用 `-PowerColumn`。


## 10. 参考依据

- HWiNFO 官方下载：<https://www.hwinfo.com/download/>
- HWiNFO 作者关于日志实际间隔会受传感器读取与写盘影响：<https://www.hwinfo.com/forum/threads/inconsistent-polling-period-logging.7948/>
- HWiNFO 作者关于用 Custom 页的 Logged 选项选择 CSV 项目：<https://www.hwinfo.com/forum/threads/question-about-csv-logging.6417/>
- HWiNFO 官方公司/发布者信息（REALiX, s.r.o.）：<https://www.hwinfo.com/privacy-policy/>
