[CmdletBinding()]
param(
    [string]$PcmExe = "",
    [string]$HwinfoExe = "",
    [ValidateSet("Pcm", "Hwinfo")]
    [string]$EnergyBackend = "Pcm",
    [switch]$IncludeOpenVino,
    [switch]$SkipEnergy
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "请先运行 .\setup_windows.ps1" }
if (-not $SkipEnergy -and $EnergyBackend -eq "Pcm" -and -not $PcmExe) {
    throw "PCM 正式流程需要 -PcmExe C:\path\to\pcm.exe；也可改用 -EnergyBackend Hwinfo。"
}
if ($EnergyBackend -eq "Pcm" -and $PcmExe -and -not (Test-Path $PcmExe -PathType Leaf)) {
    throw "找不到 pcm.exe: $PcmExe"
}
if (-not $SkipEnergy -and $EnergyBackend -eq "Hwinfo" -and -not $HwinfoExe) {
    throw "HWiNFO 正式流程需要 -HwinfoExe C:\path\to\HWiNFO64.exe。"
}
if ($HwinfoExe -and -not (Test-Path $HwinfoExe -PathType Leaf)) {
    throw "找不到 HWiNFO64.exe: $HwinfoExe"
}
if (-not $SkipEnergy -and $EnergyBackend -eq "Hwinfo") {
    & ".\check_hwinfo.ps1" -HwinfoExe $HwinfoExe
}

$providers = @("cpu")
if ($IncludeOpenVino) { $providers += "openvino" }

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python 命令失败：$($Arguments -join ' ')" }
}

$activeOutput = powercfg /GETACTIVESCHEME
$match = [regex]::Match(($activeOutput -join " "), "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
if (-not $match.Success) { throw "无法读取当前 Windows 电源计划。" }
$originalScheme = $match.Value

try {
    powercfg /SETACTIVE SCHEME_MIN
    if ($LASTEXITCODE -ne 0) { throw "无法切换到高性能电源计划。" }

    Invoke-Python scripts\system_info.py
    Invoke-Python scripts\inspect_models.py

    $prepareArguments = @("scripts\prepare_local_models.py", "--providers") + $providers
    Invoke-Python @prepareArguments

    foreach ($provider in $providers) {
        Invoke-Python scripts\benchmark.py tune --provider $provider
        Invoke-Python scripts\benchmark.py latency --provider $provider
        Invoke-Python scripts\benchmark.py profile --provider $provider
        if ($provider -eq "openvino") {
            $tuning = Get-Content -Raw "results\tuning_openvino.json" | ConvertFrom-Json
            $threads = $tuning.selected.threads
            Invoke-Python scripts\inspect_openvino.py --threads $threads
        }
        if (-not $SkipEnergy) {
            if ($EnergyBackend -eq "Pcm") {
                Invoke-Python scripts\run_energy.py --provider $provider --pcm-exe $PcmExe
            } else {
                if (-not (Get-Process -Name "HWiNFO64" -ErrorAction SilentlyContinue)) {
                    throw "未检测到 HWiNFO64。请先启动 Sensors-only 并开始 CSV 日志。"
                }
                Invoke-Python scripts\run_hwinfo_energy.py record --provider $provider
            }
        }
    }
    if ($SkipEnergy -or $EnergyBackend -eq "Pcm") {
        $reportArguments = @("scripts\generate_report.py", "--providers") + $providers
        Invoke-Python @reportArguments
    }
} finally {
    powercfg /SETACTIVE $originalScheme | Out-Null
}

if (-not $SkipEnergy -and $EnergyBackend -eq "Hwinfo") {
    Write-Host "时间窗记录完成。停止 HWiNFO 日志后运行 .\run_hwinfo_energy.ps1 -Mode Analyze -HwinfoLog <CSV>。" -ForegroundColor Yellow
} else {
    Write-Host "全部流程完成。结果位于 results\result_summary.md" -ForegroundColor Green
}
