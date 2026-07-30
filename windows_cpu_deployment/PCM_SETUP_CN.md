# Intel PCM（Windows 11）准备说明

若当前 Windows 因 PCM 驱动签名而受阻，请保持 Secure Boot/HVCI 开启，改用 [HWINFO_SETUP_CN.md](HWINFO_SETUP_CN.md)；本文件只描述仍可合法加载 PCM 驱动时的可选路径。

本部署包使用 Intel Performance Counter Monitor（PCM）的 `pcm.exe` 读取 RAPL CPU Package 能量。正式能耗测试必须在原生 Windows 11 的管理员 PowerShell 中运行，WSL 不参与测量。

## 获取与安装

1. 从 Intel 官方 PCM 仓库获取 Windows x64 Release 构建：<https://github.com/intel/pcm>。
2. 按官方 `doc/WINDOWS_HOWTO.md` 准备 Windows MSR 驱动及 Visual C++ Runtime。
3. 将 `pcm.exe`、其依赖 DLL 和签名驱动保持在官方构建目录中。
4. 使用管理员 PowerShell，按官方说明安装/启动驱动。部分发行版支持：

   ```powershell
   .\pcm.exe --installDriver
   ```

5. 关闭可能占用 PMU 的 VTune、其他 PCM 实例或硬件监控工具。

不要从不明镜像下载 `msr.sys` 或 `pcm.exe`。PCM 需要内核驱动权限，应只使用 Intel 官方源码或官方 CI 构建。

## 验证

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\check_pcm.ps1 -PcmExe "C:\Tools\pcm\pcm.exe"
```

成功条件是脚本能解析 `Proc Energy (Joules)`。如果仅能读取利用率而没有该列，则不能生成 `J/image`，正式流程会终止。

常见错误：

- `MSRAccessDenied`：未使用管理员终端或驱动未正确安装。
- `PMUBusy`：其他监控工具占用了 PMU；退出相关程序后重试。
- Hyper-V/虚拟化提示：正式测试仍应由原生 Windows 的 `pcm.exe` 运行，而不是 WSL 内的工具。
