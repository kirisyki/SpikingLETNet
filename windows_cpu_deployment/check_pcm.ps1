[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PcmExe
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "请在管理员 PowerShell 中运行本脚本。"
}
if (-not (Test-Path $PcmExe -PathType Leaf)) { throw "找不到 pcm.exe: $PcmExe" }
if (-not (Test-Path ".venv\Scripts\python.exe")) { throw "请先运行 setup_windows.ps1。" }

$output = Join-Path $PSScriptRoot "results\pcm_smoke.csv"
$client = Join-Path $PSScriptRoot "results\pcm_sleep.py"
"import time; time.sleep(3)" | Set-Content -Path $client -Encoding ascii
try {
    & $PcmExe 0 "-csv=$output" -nc --no-color -- ".venv\Scripts\python.exe" $client
    if ($LASTEXITCODE -ne 0) { throw "pcm.exe 返回 $LASTEXITCODE。检查驱动、管理员权限和 PMU 占用。" }
    & ".venv\Scripts\python.exe" -c "from pathlib import Path; import sys; sys.path.insert(0, 'scripts'); from pcm_tools import parse_pcm_csv; print(parse_pcm_csv(Path(r'$output')))"
    if ($LASTEXITCODE -ne 0) { throw "PCM CSV 中没有可解析的 CPU Package 能量。" }
} finally {
    Remove-Item $client -ErrorAction SilentlyContinue
}
Write-Host "PCM 检查通过。" -ForegroundColor Green
