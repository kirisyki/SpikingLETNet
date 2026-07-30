# Windows CPU 能耗测试阻塞说明

更新时间：2026-07-23  
适用项目：`SpikingLETNet_shallow_max` Windows 11 / i9-14900K / ONNX Runtime CPU EP

## 1. 当前结论

FP32 与 W8A8 的模型部署、ORT CPU 推理、线程调优和正式延迟测试已经完成。Intel PCM/RAPL CPU Package 能耗测试没有启动，状态为：

```text
blocked/not measured
```

没有生成或估算以下指标：

- CPU Package J/image
- Dynamic J/image
- 平均 CPU Package W
- FP32/W8A8 能耗变化
- 有效能耗试验数和能耗 CV

当前存在两个相互独立的阻塞：

| 阻塞 | 当前状态 | 单独解决后能否完成 PCM 测试 |
|---|---|---|
| Codex 工具进程没有管理员令牌 | `IsAdministrator=False` | 不能，仍受驱动签名阻塞 |
| PCM `MSR.sys` 只有测试签名 | Secure Boot、HVCI 均开启 | 不能通过普通管理员权限解决 |

## 2. 管理员权限问题

### 2.1 实际检查结果

在当前 Codex 工具进程内执行：

```powershell
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
```

结果：

```text
User            : 4090LAB\93211
IsAdministrator : False
Process         : C:\Program Files\WindowsApps\Microsoft.PowerShell_7.6.4.0_x64__8wekyb3d8bbwe\pwsh.exe
```

`Confirm-SecureBootUEFI` 在该工具进程中也返回“访问被拒绝”。

### 2.2 为什么外层管理员 PowerShell 没有解决

可能原因包括：

- Codex 复用了此前启动的非提升后台进程。
- Codex 的命令执行代理与外层 CLI 进程隔离。
- 执行代理出于安全原因剥离了 Windows 管理员令牌。
- Codex 的命令审批只解除工作区沙箱限制，不等同于 Windows UAC 提升。

因此，不能根据终端标题中的“管理员”推断 Codex 工具子进程也具有管理员权限，必须在实际执行命令的进程中检查。

### 2.3 权限问题可解决的范围

真正提升后的管理员进程可以：

- 安装或启动已经满足 Windows 签名策略的 PCM 驱动。
- 创建或控制 PCM Windows 服务。
- 访问 PCM 暴露的 MSR 设备。
- 执行 `check_pcm.ps1`。

管理员权限不能：

- 把测试签名驱动变成 Microsoft 正式签名驱动。
- 绕过 Secure Boot 或 HVCI 的内核代码完整性检查。
- 证明当前 `MSR.sys` 与 Windows 11 25H2/HVCI 兼容。

## 3. PCM 驱动证书问题

### 3.1 PCM 来源

使用的 PCM 来自 Intel `intel/pcm` 官方 AppVeyor 项目：

| 字段 | 值 |
|---|---|
| AppVeyor build | `1.0.2317` |
| Build ID | `54345897` |
| Job ID | `jbjxpe3mh6delrs3` |
| Git commit | `6cee474147e0e70827656a3e26aac668b42938f4` |
| Artifact | `pcm-all.zip` |
| Artifact 大小 | `18493201` bytes |
| Artifact SHA-256 | `EFF649FC3C28336D11D158EEE09168B6E456E592C7A6022BBC10AD43E31C8C23` |

本地目录：

```text
tools\pcm-appveyor-1.0.2317\extracted
```

### 3.2 二进制和驱动签名

```text
pcm.exe SHA-256:
34320D0E404374BC3A842C019C9D2D2E8B85BF0EBCEA01584D88FF886255CE08

MSR.sys SHA-256:
AE5E766535F51515C8EFBF1B3F8289681074D3BEB2DD95535E5A1EADD6A02C53
```

`Get-AuthenticodeSignature` 结果：

```text
SignatureStatus : UnknownError
StatusMessage   : 已处理证书链，但是在不受信任提供程序信任的根证书中终止。
Subject         : CN="WDKTestCert appveyor,134280125769083826"
Issuer          : CN="WDKTestCert appveyor,134280125769083826"
Thumbprint      : 47B70993F339111CE93164474F2E91E722E087A7
```

这是 AppVeyor 构建环境生成的自签名 WDK 测试证书，不是 Microsoft attestation 或 WHQL 生产签名。

### 3.3 主机安全状态

已记录的主机状态：

```text
Secure Boot             : Enabled
HVCI / Memory Integrity : Enabled
HVCI registry value     : 1
```

在这种配置下，单纯将 `WDKTestCert` 导入 Windows 用户或计算机的受信任根证书存储区，不能使该驱动满足现代 Windows 内核加载策略。重新使用本地自签名证书签名也不能替代 Microsoft 内核签名。

### 3.4 现有 PCM 检查结果

执行：

```powershell
.\check_pcm.ps1 -PcmExe "<official-appveyor-path>\pcm.exe"
```

脚本在测量前停止，原因为当前工具进程没有管理员权限：

```text
请在管理员 PowerShell 中运行本脚本。
```

由于签名检查已经确认当前 `MSR.sys` 不受系统信任，即使先解决管理员令牌问题，仍预计会在驱动安装或加载阶段受阻。为了避免改变系统安全策略，没有尝试加载该驱动。

## 4. 现有方案禁止采用的规避方法

`instruction.md` 要求不得为了能耗测试执行以下操作：

- 关闭 Secure Boot。
- 关闭 HVCI 或 Windows“内存完整性”。
- 启用 `TESTSIGNING`。
- 临时关闭驱动签名强制。
- 从不明来源下载或替换 `MSR.sys`。
- 使用估算值冒充 Intel PCM/RAPL 实测值。
- 将墙上插座功耗标记为 CPU Package 功耗。

这些限制应继续保留。MSR 驱动能够读写处理器模型专用寄存器，加载不受信任或存在漏洞的驱动会扩大内核攻击面。

## 5. 严格保留 Intel PCM 口径时的解除条件

若新方案仍要求 Intel PCM/RAPL CPU Package 能量，必须同时满足：

1. 获得与 `pcm.exe` 配套的 Windows x64 `MSR.sys`。
2. 驱动具有 Microsoft attestation 或 WHQL/WHCP 认可的内核签名。
3. 驱动通过 Windows 11 25H2 的 HVCI 兼容性要求。
4. 使用 Windows SDK 验证内核签名：

   ```powershell
   signtool verify /kp /v "C:\Tools\pcm\MSR.sys"
   ```

5. 在实际管理员进程中运行 `check_pcm.ps1`。
6. `pcm.exe` 输出必须包含并可解析 `Proc Energy (Joules)`。
7. 不改变 Secure Boot、HVCI 或驱动签名策略。

如果由驱动维护组织自行解决签名，需要注册 Microsoft Hardware Developer Program、持有 EV 代码签名证书、准备驱动提交包，并通过 Partner Center 获取 Microsoft 签名。该流程通常应由 Intel 或拥有驱动源码和签名资质的组织完成。

参考资料：

- Intel PCM：<https://github.com/intel/pcm>
- Intel PCM Windows HOWTO：<https://github.com/intel/pcm/blob/master/doc/WINDOWS_HOWTO.md>
- Microsoft 内核驱动签名要求：<https://learn.microsoft.com/windows-hardware/drivers/install/kernel-mode-code-signing-requirements--windows-vista-and-later->
- Microsoft attestation signing：<https://learn.microsoft.com/windows-hardware/drivers/dashboard/code-signing-attestation>

## 6. 修改能耗测试方案时需要重新定义的内容

若不再强制使用 PCM，不能只替换采集工具名称。新方案至少应明确以下内容。

### 6.1 测量边界

必须选择并固定一种口径：

- CPU Package 能量。
- CPU 插槽或主板 EPS 输入能量。
- 整机 AC 墙上插座能量。
- 软件估计的处理器能耗。

不同边界的数值不能直接比较，也不能沿用相同字段名称。

### 6.2 采集工具和可信度

需要记录：

- 工具名称、版本和来源。
- 传感器名称及单位。
- 采样周期和时间戳精度。
- 能量是硬件累计计数器、功率积分还是模型估计。
- 计数器回绕、缺失样本和异常值处理方法。
- 工具是否需要管理员权限或内核驱动。
- 驱动签名、Secure Boot 和 HVCI 兼容情况。

HWiNFO、Intel Power Gadget、Windows 性能计数器、主板监控接口和外部功率计的测量边界不同，不能在未经验证时统一称为 RAPL CPU Package 能量。

### 6.3 建议保留的实验协议

为保持与现有延迟测试尽量可比较，建议继续保留：

- 原生 Windows 11。
- ONNX Runtime `CPUExecutionProvider`。
- batch=1、固定 `400x400`。
- FP32/W8A8 使用同一 affinity 和线程配置。
- 每个模型预热至少 30 秒。
- 每次能耗窗口至少 60 秒。
- FP32/W8A8 交替执行。
- 每个模型至少 5 次有效试验。
- 至少 3 次空闲基线。
- 后台 CPU 占用超过 10% 时试验无效。
- 可获取温度或热余量时，定义明确的热稳定阈值。
- CV 超过 5% 时最多补测 3 次，仍超标则报告不稳定。
- 保留所有试验，不选择性删除较慢或能耗较高的结果。

### 6.4 建议输出字段

无论使用何种工具，建议输出：

- 原始采样文件。
- 测量窗口起止时间。
- 推理次数。
- 测量窗口实际时长。
- 平均功率。
- 总能量。
- 每图能量。
- 空闲基线功率。
- 扣除空闲后的动态每图能量。
- 有效/无效状态和无效原因。
- 温度或热余量统计。
- 后台 CPU 占用。
- 重复试验均值、中位数、标准差和 CV。

功率积分型方案应明确计算：

```text
Total energy = integral(power over measurement time)
Energy per image = total energy / completed inference count
Dynamic energy per image =
    (total energy - idle baseline power * measurement duration)
    / completed inference count
```

若使用外部功率计，字段应命名为整机或 AC 输入能耗，不能使用 `CPU Package J/image`。

## 7. 需要修改的项目文件

确定新方案后，预计需要审查或修改：

```text
PCM_SETUP_CN.md
instruction.md
check_pcm.ps1
scripts/pcm_tools.py
scripts/run_energy.py
scripts/generate_report.py
scripts/self_test.py
```

同时需要决定是否继续生成：

```text
results/energy_cpu.json
results/energy_trials_cpu.csv
results/raw_pcm/cpu/
```

若采集工具不再是 PCM，建议将 `raw_pcm` 和 PCM 专用字段重命名，避免报告产生错误的测量来源暗示。

## 8. 已保留的证据

- 能耗阻塞状态：`results/energy_status_cpu.json`
- 系统信息：`results/system_info.json`
- 延迟结果：`results/latency_cpu.json`
- 延迟稳定性：`results/latency_stability_cpu.json`
- 中文结果摘要：`results/result_summary.md`
- Intel PCM artifact：`tools/pcm-appveyor-1.0.2317/`

在新方案确定之前，当前能耗结论应继续保持：

```text
CPU energy: blocked/not measured
No estimated energy values were reported.
```


## 9. 已批准并实现的安全替代路径

上述 PCM 状态作为历史证据保留：在当前 Secure Boot/HVCI 策略下，PCM 仍是 blocked。用户已批准采用 HWiNFO `CPU Package Power [W]` 连续日志积分作为新的正式测量方法，且不修改 Windows 安全设置。

新路径的测量名称是 **HWiNFO CPU Package 功率积分能耗**，不是 PCM/RAPL 计数器差值。实现和执行约束见：

```text
HWINFO_SETUP_CN.md
check_hwinfo.ps1
run_hwinfo_energy.ps1
scripts/hwinfo_tools.py
scripts/run_hwinfo_energy.py
```

新结果写入独立文件：

```text
results/energy_hwinfo_cpu.json
results/energy_trials_hwinfo_cpu.csv
results/raw_hwinfo/cpu/
results/hwinfo_trial_markers_cpu.json
```

因此，当前状态应区分为：

```text
Intel PCM/RAPL energy: blocked under current security policy
HWiNFO integrated CPU Package energy: implementation ready, measurement pending
```
