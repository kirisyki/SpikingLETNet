[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Record", "Analyze")]
    [string]$Mode,
    [string]$HwinfoLog = "",
    [Parameter(Mandatory = $true)]
    [string]$HwinfoExe,
    [switch]$IncludeOpenVino,
    [ValidateSet("auto", "ymd", "mdy", "dmy")]
    [string]$DateOrder = "auto",
    [string]$PowerColumn = "",
    [double]$DurationSeconds = 180.0,
    [double]$WarmupSeconds = 30.0,
    [int]$Trials = 5,
    [int]$IdleTrials = 3,
    [int]$MaxExtraTrials = 3
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python -PathType Leaf)) {
    throw "请先运行 .\setup_windows.ps1"
}

$providers = @("cpu")
if ($IncludeOpenVino) {
    $providers += "openvino"
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python 命令失败：$($Arguments -join ' ')"
    }
}

& ".\check_hwinfo.ps1" -HwinfoExe $HwinfoExe

if ($Mode -eq "Record") {
    if (-not (Get-Process -Name "HWiNFO64" -ErrorAction SilentlyContinue)) {
        throw "未检测到 HWiNFO64 进程。请先启动 Sensors-only 并开始 CSV 日志。"
    }

    $activeOutput = powercfg /GETACTIVESCHEME
    $match = [regex]::Match(
        ($activeOutput -join " "),
        "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    if (-not $match.Success) {
        throw "无法读取当前 Windows 电源计划。"
    }
    $originalScheme = $match.Value
    try {
        powercfg /SETACTIVE SCHEME_MIN
        if ($LASTEXITCODE -ne 0) {
            throw "无法切换到高性能电源计划。"
        }
        foreach ($provider in $providers) {
            $arguments = @(
                "scripts\run_hwinfo_energy.py", "record",
                "--provider", $provider,
                "--duration-seconds", $DurationSeconds,
                "--warmup-seconds", $WarmupSeconds,
                "--trials", $Trials,
                "--idle-trials", $IdleTrials,
                "--max-extra-trials", $MaxExtraTrials
            )
            Invoke-Python @arguments
        }
    } finally {
        powercfg /SETACTIVE $originalScheme | Out-Null
    }

    Write-Host "时间窗记录完成。现在停止 HWiNFO 日志，确认 CSV 已写入磁盘，然后运行 Analyze 阶段。" -ForegroundColor Yellow
    return
}

if (-not $HwinfoLog) {
    throw "Analyze 模式需要 -HwinfoLog D:\path\to\log.csv"
}
if (-not (Test-Path $HwinfoLog -PathType Leaf)) {
    throw "找不到 HWiNFO 日志：$HwinfoLog"
}

$inspectArguments = @(
    "scripts\hwinfo_tools.py", "inspect",
    "--log", $HwinfoLog,
    "--date-order", $DateOrder,
    "--output", "results\hwinfo_log_inspection.json"
)
if ($PowerColumn) {
    $inspectArguments += @("--power-column", $PowerColumn)
}
Invoke-Python @inspectArguments

$hwinfoVersion = (Get-Item $HwinfoExe).VersionInfo.FileVersion
foreach ($provider in $providers) {
    $arguments = @(
        "scripts\run_hwinfo_energy.py", "analyze",
        "--provider", $provider,
        "--hwinfo-log", $HwinfoLog,
        "--hwinfo-version", $hwinfoVersion,
        "--date-order", $DateOrder
    )
    if ($PowerColumn) {
        $arguments += @("--power-column", $PowerColumn)
    }
    Invoke-Python @arguments
}

$reportArguments = @("scripts\generate_report.py", "--providers") + $providers
Invoke-Python @reportArguments
Write-Host "HWiNFO 能耗分析完成：results\result_summary.md" -ForegroundColor Green
