[CmdletBinding()]
param(
    [switch]$InstallPython
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Python311 {
    try {
        & py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" 2>$null
        if ($LASTEXITCODE -eq 0) { return @("py", "-3.11") }
    } catch {}
    try {
        & python -c "import sys; assert sys.version_info[:2] == (3, 11)" 2>$null
        if ($LASTEXITCODE -eq 0) { return @("python") }
    } catch {}
    return $null
}

$pythonCommand = Find-Python311
if (-not $pythonCommand -and $InstallPython) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "未找到 Python 3.11 或 winget。请从 https://www.python.org/downloads/ 安装 Python 3.11 x64。"
    }
    winget install --id Python.Python.3.11 -e --scope user --accept-package-agreements --accept-source-agreements
    $pythonCommand = Find-Python311
}
if (-not $pythonCommand) {
    throw "需要 Python 3.11 x64。安装后重试，或使用 .\setup_windows.ps1 -InstallPython。"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    if ($pythonCommand.Count -eq 2) {
        & $pythonCommand[0] $pythonCommand[1] -m venv .venv
    } else {
        & $pythonCommand[0] -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败。" }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade "pip==25.1.1"
if ($LASTEXITCODE -ne 0) { throw "pip 更新失败。" }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python 依赖安装失败。" }

& $venvPython -c "import onnxruntime as ort; print('ORT', ort.__version__); print('providers', ort.get_available_providers()); assert {'CPUExecutionProvider', 'OpenVINOExecutionProvider'}.issubset(ort.get_available_providers())"
if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime CPU EP 验证失败。" }
& $venvPython scripts\self_test.py
if ($LASTEXITCODE -ne 0) { throw "部署包自检失败。" }

Write-Host "环境安装完成。下一步请阅读 PCM_SETUP_CN.md，然后运行 run_all.ps1。" -ForegroundColor Green
