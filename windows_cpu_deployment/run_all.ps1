[CmdletBinding()]
param(
    [string]$PcmExe = "",
    [switch]$IncludeOpenVino,
    [switch]$SkipEnergy
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "请先运行 .\setup_windows.ps1" }
if (-not $SkipEnergy -and -not $PcmExe) { throw "正式流程需要 -PcmExe C:\path\to\pcm.exe；或显式使用 -SkipEnergy。" }
if ($PcmExe -and -not (Test-Path $PcmExe -PathType Leaf)) { throw "找不到 pcm.exe: $PcmExe" }

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
            $threads = (& $python -c "import json; print(json.load(open(r'results/tuning_openvino.json'))['selected']['threads'])").Trim()
            if ($LASTEXITCODE -ne 0) { throw "无法读取 OpenVINO 线程配置。" }
            Invoke-Python scripts\inspect_openvino.py --threads $threads
        }
        if (-not $SkipEnergy) {
            Invoke-Python scripts\run_energy.py --provider $provider --pcm-exe $PcmExe
        }
    }
    $reportArguments = @("scripts\generate_report.py", "--providers") + $providers
    Invoke-Python @reportArguments
} finally {
    powercfg /SETACTIVE $originalScheme | Out-Null
}

Write-Host "全部流程完成。结果位于 results\result_summary.md" -ForegroundColor Green
